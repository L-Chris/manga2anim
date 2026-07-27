import { describe, it, expect, vi } from "vitest";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Mock the browser wasm runtime so importing the PURE pipeline functions works
// in node; real inference below uses onnxruntime-node.
vi.mock("onnxruntime-web", () => ({}));

import {
  OCR_MIN_CONFIDENCE,
  parseDictionary,
  runOcrPipeline,
} from "../ppocr";
import {
  buildClassNames,
  DEFAULT_SEG_CLASS_NAMES,
  DEFAULT_SEG_INPUT_SIZE,
  letterbox,
  postprocessYolo,
} from "./onnxProvider";
import type { BBox, OcrLine } from "../../types";

const ROOT = process.cwd();
const SEG = join(ROOT, "models/yolo26s-manga-seg.onnx");
const DET = join(ROOT, "models/ppocrv6-det.onnx");
const REC = join(ROOT, "models/ppocrv6-rec.onnx");
const DICT = join(ROOT, "models/ppocrv6_dict.txt");
const DATA = join(ROOT, "data");

const ready =
  existsSync(SEG) &&
  existsSync(DET) &&
  existsSync(REC) &&
  existsSync(DICT) &&
  existsSync(DATA);
const webp = ready
  ? readdirSync(DATA).filter((f) => /\.(webp|png|jpe?g)$/i.test(f)).sort()
  : [];
const describeIf = webp.length > 0 ? describe : describe.skip;

async function decode(path: string) {
  const sharp = (await import("sharp")).default;
  const { data, info } = await sharp(readFileSync(path))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    width: info.width,
    height: info.height,
    data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
  };
}

const OCR_GROUND_TRUTH: Record<
  string,
  Array<{ text: string; bbox: BBox }>
> = {
  "1.webp": [
    { text: "2017年", bbox: { x: 904, y: 147, w: 48, h: 198 } },
    { text: "今天也讓我", bbox: { x: 949, y: 596, w: 38, h: 140 } },
    { text: "在這裡工作吧～", bbox: { x: 916, y: 599, w: 39, h: 190 } },
    { text: "你最近", bbox: { x: 160, y: 621, w: 40, h: 92 } },
    { text: "來得挺勤啊", bbox: { x: 129, y: 622, w: 39, h: 137 } },
    { text: "馬上就要發售", bbox: { x: 971, y: 1026, w: 38, h: 163 } },
    { text: "要忙的事情", bbox: { x: 831, y: 1175, w: 38, h: 137 } },
    { text: "店鋪獨家特典", bbox: { x: 896, y: 1176, w: 35, h: 160 } },
  ],
};

function intersectionOverUnion(a: BBox, b: BBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - intersection;
  return union > 0 ? intersection / union : 0;
}

function editDistance(expected: string, actual: string): number {
  const a = [...expected.replace(/\s/g, "")];
  const b = [...actual.replace(/\s/g, "")];
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(
        current[j - 1] + 1,
        previous[j] + 1,
        previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
    previous.splice(0, previous.length, ...current);
  }
  return previous[b.length];
}

function measureCharacterAccuracy(
  groundTruth: Array<{ text: string; bbox: BBox }>,
  lines: OcrLine[]
) {
  const used = new Set<number>();
  let errors = 0;
  let expectedCharacters = 0;
  const matches = groundTruth.map((expected) => {
    let bestIndex = -1;
    let bestOverlap = 0;
    for (let index = 0; index < lines.length; index++) {
      if (used.has(index)) continue;
      const overlap = intersectionOverUnion(expected.bbox, lines[index].bbox);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestIndex = index;
      }
    }
    const actual = bestOverlap >= 0.2 && bestIndex >= 0 ? lines[bestIndex].text : "";
    if (actual) used.add(bestIndex);
    const distance = editDistance(expected.text, actual);
    expectedCharacters += [...expected.text].length;
    errors += distance;
    return { expected: expected.text, actual, distance };
  });

  const characterAccuracy = Math.max(0, 1 - errors / expectedCharacters);
  return { characterAccuracy, errors, expectedCharacters, matches };
}

describeIf("REAL PP-OCRv6 det+rec ONNX executes and decodes (headless)", () => {
  it(
    "filters OCR through YOLO regions and reports character accuracy",
    async () => {
      const ortNode = (await import("onnxruntime-node")).default;
      const segSess = await ortNode.InferenceSession.create(SEG, {
        executionProviders: ["cpu"],
      });
      const detSess = await ortNode.InferenceSession.create(DET, {
        executionProviders: ["cpu"],
      });
      const recSess = await ortNode.InferenceSession.create(REC, {
        executionProviders: ["cpu"],
      });
      const dictionary = parseDictionary(readFileSync(DICT, "utf8"));
      const classNames = buildClassNames(DEFAULT_SEG_CLASS_NAMES);
      // The v6 rec head emits 18710 classes = 18708 dict lines + blank + space.
      // Sanity-check the dict matches the model's vocabulary size.
      expect(dictionary.length + 2).toBe(18710);

      const runDet = async (tensor: Float32Array, h: number, w: number) => {
        const res = await detSess.run({
          [detSess.inputNames[0]]: new ortNode.Tensor("float32", tensor, [
            1,
            3,
            h,
            w,
          ]),
        });
        return res[detSess.outputNames[0]].data as Float32Array;
      };
      const runRec = async (tensor: Float32Array, _width: number) => {
        const res = await recSess.run({
          [recSess.inputNames[0]]: new ortNode.Tensor("float32", tensor, [
            1,
            3,
            48,
            320,
          ]),
        });
        const out = res[recSess.outputNames[0]];
        const dims = out.dims as number[];
        return {
          logits: out.data as Float32Array,
          T: dims[1],
          numClasses: dims[2],
        };
      };

      const summary: Array<Record<string, unknown>> = [];
      let measuredPages = 0;

      for (const file of webp) {
        const img = await decode(join(DATA, file));
        const segInput = letterbox(img, DEFAULT_SEG_INPUT_SIZE);
        const segResult = await segSess.run({
          [segSess.inputNames[0]]: new ortNode.Tensor("float32", segInput.tensor, [
            1,
            3,
            DEFAULT_SEG_INPUT_SIZE,
            DEFAULT_SEG_INPUT_SIZE,
          ]),
        });
        const detOutput = segResult[segSess.outputNames[0]];
        const protoOutput = segResult[segSess.outputNames[1]];
        const detections = postprocessYolo(
          detOutput.data as Float32Array,
          detOutput.dims as number[],
          protoOutput.data as Float32Array,
          protoOutput.dims as number[],
          segInput.scale,
          segInput.padX,
          segInput.padY,
          DEFAULT_SEG_INPUT_SIZE,
          { width: img.width, height: img.height },
          classNames
        );
        const regions = detections
          .filter(
            (detection) =>
              detection.className === "text" || detection.className === "bubble"
          )
          .map((detection) => detection.bbox);
        const lines = await runOcrPipeline(
          img.data,
          img.width,
          img.height,
          runDet,
          runRec,
          dictionary,
          { regions, minConfidence: OCR_MIN_CONFIDENCE }
        );
        expect(regions.length).toBeGreaterThan(0);
        expect(lines.every((line) => line.text.trim().length > 0)).toBe(true);
        expect(
          lines.every((line) => line.confidence >= OCR_MIN_CONFIDENCE)
        ).toBe(true);
        expect(lines.some((line) => line.text === "20")).toBe(false);

        const groundTruth = OCR_GROUND_TRUTH[file];
        const metrics = groundTruth
          ? measureCharacterAccuracy(groundTruth, lines)
          : undefined;
        if (metrics) {
          measuredPages++;
          // Keep a measured floor for the checked-in vertical Traditional
          // Chinese page so model/preprocessing changes cannot silently return
          // to the old non-empty-only baseline.
          expect(metrics.characterAccuracy).toBeGreaterThanOrEqual(0.98);
        }
        summary.push({
          file,
          yoloRegions: regions.length,
          lines: lines.map((l) => ({
            text: l.text,
            conf: Math.round(l.confidence * 1000) / 1000,
            bbox: l.bbox,
          })),
          metrics,
        });
      }

      writeFileSync(
        join(ROOT, "scripts/real-ocr-summary.json"),
        JSON.stringify(summary, null, 2)
      );

      expect(measuredPages).toBeGreaterThan(0);
      console.info("OCR character-accuracy summary", JSON.stringify(summary));
    },
    300_000
  );
});
