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
import { isOtherText } from "../../types";
import type { BBox, OcrLine } from "../../types";
import { parsePage } from "../pipeline";

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
    { text: "已经够了…", bbox: { x: 791, y: 64, w: 43, h: 157 } },
    { text: "我不想看见", bbox: { x: 332, y: 266, w: 47, h: 162 } },
    { text: "优诺", bbox: { x: 295, y: 268, w: 46, h: 77 } },
    { text: "再受更多伤了！", bbox: { x: 188, y: 355, w: 44, h: 200 } },
    { text: "切", bbox: { x: 646, y: 509, w: 55, h: 55 } },
    { text: "已经…", bbox: { x: 411, y: 837, w: 55, h: 120 } },
    { text: "够了…", bbox: { x: 285, y: 1016, w: 57, h: 125 } },
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
          expect(metrics.characterAccuracy).toBeGreaterThanOrEqual(0.95);
        }

        // Reuse the already-computed real detections/OCR lines to verify the
        // final assembly. Every text region must occur in exactly one visible
        // group: either one panel or the unassigned-text section, never both.
        const parsed = await parsePage(
          img,
          {
            id: "real-assembly",
            label: "Real assembly fixture",
            segmenter: {
              name: "Captured real detections",
              async segment() {
                return detections;
              },
            },
            ocr: {
              name: "Captured real OCR",
              async recognize() {
                return lines;
              },
            },
          },
          {
            pageId: file,
            name: file,
            readingDirection: "rtl",
          }
        );
        const panelTextIds = parsed.panels.flatMap((panel) => panel.textIds);
        const otherTextIds = parsed.textRegions.filter(isOtherText).map((r) => r.id);
        const visibleTextIds = [...panelTextIds, ...otherTextIds];
        expect(new Set(visibleTextIds).size).toBe(visibleTextIds.length);
        expect(new Set(visibleTextIds)).toEqual(
          new Set(parsed.textRegions.map((region) => region.id))
        );

        summary.push({
          file,
          yoloRegions: regions.length,
          lines: lines.map((l) => ({
            text: l.text,
            conf: Math.round(l.confidence * 1000) / 1000,
            bbox: l.bbox,
          })),
          metrics,
          assembledRegions: parsed.textRegions.map((region) => ({
            text: region.text,
            kind: region.kind,
            fromBubble: region.fromBubble,
            panelId: region.panelId,
          })),
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
