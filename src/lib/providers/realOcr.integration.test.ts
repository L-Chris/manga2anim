import { describe, it, expect, vi } from "vitest";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Mock the browser wasm runtime so importing the PURE pipeline functions works
// in node; real inference below uses onnxruntime-node.
vi.mock("onnxruntime-web", () => ({}));

import { parseDictionary, runOcrPipeline } from "../ppocr";

const ROOT = process.cwd();
const DET = join(ROOT, "models/ppocrv6-det.onnx");
const REC = join(ROOT, "models/ppocrv6-rec.onnx");
const DICT = join(ROOT, "models/ppocrv6_dict.txt");
const DATA = join(ROOT, "data");

const ready =
  existsSync(DET) && existsSync(REC) && existsSync(DICT) && existsSync(DATA);
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

describeIf("REAL PP-OCRv6 det+rec ONNX executes and decodes (headless)", () => {
  it(
    "produces non-empty recognized text on data/ pages",
    async () => {
      const ortNode = (await import("onnxruntime-node")).default;
      const detSess = await ortNode.InferenceSession.create(DET, {
        executionProviders: ["cpu"],
      });
      const recSess = await ortNode.InferenceSession.create(REC, {
        executionProviders: ["cpu"],
      });
      const dictionary = parseDictionary(readFileSync(DICT, "utf8"));
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

      const summary: Array<{
        file: string;
        lines: Array<{ text: string; conf: number }>;
      }> = [];
      let totalLines = 0;

      for (const file of webp) {
        const img = await decode(join(DATA, file));
        const lines = await runOcrPipeline(
          img.data,
          img.width,
          img.height,
          runDet,
          runRec,
          dictionary
        );
        totalLines += lines.length;
        summary.push({
          file,
          lines: lines.map((l) => ({
            text: l.text,
            conf: Math.round(l.confidence * 1000) / 1000,
          })),
        });
      }

      writeFileSync(
        join(ROOT, "scripts/real-ocr-summary.json"),
        JSON.stringify(summary, null, 2)
      );

      // The whole point: the real model + fixed decode must yield real text.
      expect(totalLines).toBeGreaterThan(0);
      const anyReadable = summary.some((p) =>
        p.lines.some((l) => l.text.trim().length >= 1)
      );
      expect(anyReadable).toBe(true);
    },
    300_000
  );
});
