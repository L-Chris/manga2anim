import { describe, it, expect, vi } from "vitest";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// The provider module imports onnxruntime-web (browser wasm) at top level; we
// never use it here. Mock it so importing the PURE decode functions works in
// node, while real inference below uses onnxruntime-node.
vi.mock("onnxruntime-web", () => ({}));

import {
  buildClassNames,
  DEFAULT_SEG_CLASS_NAMES,
  DEFAULT_SEG_INPUT_SIZE,
  letterbox,
  postprocessYolo,
} from "./onnxProvider";
import { sortByReadingOrder } from "../readingOrder";

const ROOT = process.cwd();
const SEG = join(ROOT, "models/yolo26s-manga-seg.onnx");
const DATA = join(ROOT, "data");
const INPUT = DEFAULT_SEG_INPUT_SIZE;

const hasModel = existsSync(SEG);
const webp = hasModel && existsSync(DATA)
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

describeIf("REAL YOLO26s-seg ONNX executes and decodes (headless)", () => {
  it(
    "produces structured panels/text/bubbles on every data/ page",
    async () => {
      // Real native runtime — proves the exported model actually runs locally.
      const ortNode = (await import("onnxruntime-node")).default;
      const session = await ortNode.InferenceSession.create(SEG, {
        executionProviders: ["cpu"],
      });
      const classNames = buildClassNames(DEFAULT_SEG_CLASS_NAMES);

      const summary: Array<{
        file: string;
        panels: number;
        text: number;
        bubble: number;
        topConf: number;
        readingOrder: number[];
      }> = [];

      for (const file of webp) {
        const img = await decode(join(DATA, file));
        const { tensor, scale, padX, padY } = letterbox(
          { width: img.width, height: img.height, data: img.data },
          INPUT
        );
        const res = await session.run({
          [session.inputNames[0]]: new ortNode.Tensor("float32", tensor, [
            1,
            3,
            INPUT,
            INPUT,
          ]),
        });
        const det = res[session.outputNames[0]];
        const proto =
          session.outputNames.length > 1 ? res[session.outputNames[1]] : undefined;

        // Feed raw tensors into the PRODUCTION decode path.
        const dets = postprocessYolo(
          det.data as Float32Array,
          det.dims as number[],
          proto ? (proto.data as Float32Array) : undefined,
          proto ? (proto.dims as number[]) : undefined,
          scale,
          padX,
          padY,
          INPUT,
          { width: img.width, height: img.height },
          classNames
        );

        // Structural assertions: real model + fixed decode must yield sane output.
        expect(dets.length).toBeGreaterThan(0);
        const counts = { panel: 0, text: 0, bubble: 0 } as Record<string, number>;
        let topConf = 0;
        for (const d of dets) {
          expect(["panel", "text", "bubble"]).toContain(d.className);
          expect(d.confidence).toBeGreaterThan(0);
          expect(d.confidence).toBeLessThanOrEqual(1);
          expect(d.bbox.x).toBeGreaterThanOrEqual(0);
          expect(d.bbox.y).toBeGreaterThanOrEqual(0);
          expect(d.bbox.x + d.bbox.w).toBeLessThanOrEqual(img.width + 1);
          expect(d.bbox.y + d.bbox.h).toBeLessThanOrEqual(img.height + 1);
          counts[d.className]++;
          topConf = Math.max(topConf, d.confidence);
        }
        // A manga page must contain at least one panel.
        expect(counts.panel).toBeGreaterThanOrEqual(1);

        // Reading-order reconstruction runs on the real panels without error.
        const panels = dets.filter((d) => d.className === "panel");
        const ordered = sortByReadingOrder(panels, "rtl");
        expect(ordered.length).toBe(panels.length);

        summary.push({
          file,
          panels: counts.panel,
          text: counts.text,
          bubble: counts.bubble,
          topConf: Math.round(topConf * 100) / 100,
          readingOrder: ordered.map((p) => Math.round(p.bbox.y)),
        });
      }

      // Persist a human-readable summary so the run is auditable.
      writeFileSync(
        join(ROOT, "scripts/real-yolo-summary.json"),
        JSON.stringify(summary, null, 2)
      );
      expect(summary.length).toBe(webp.length);
    },
    300_000
  );
});
