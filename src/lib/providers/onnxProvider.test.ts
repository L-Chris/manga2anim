import { describe, it, expect, vi } from "vitest";

// The provider module imports onnxruntime-web at top level; we never exercise
// the session/ort.Tensor paths here (only the pure decode functions), so stub
// the module to avoid pulling the WASM runtime into the unit test.
vi.mock("onnxruntime-web", () => ({}));

import {
  buildClassNames,
  decodeMask,
  DEFAULT_SEG_CLASS_NAMES,
  DEFAULT_SEG_INPUT_SIZE,
  letterbox,
  normalizeClassName,
  postprocessYolo,
  REQUIRED_MODEL_FILES,
  createOnnxProvider,
} from "./onnxProvider";
import type { DecodedImage } from "./types";

const NM = 32; // mask coefficients (Ultralytics default)
// Verified end2end row layout: [xyxy(4)] + [conf(1)] + [classId(1)] + [mask(32)].
const C = 4 + 1 + 1 + NM; // = 38, matches the real exported model

/**
 * Build a [1, N, C] row-major end2end detection tensor (the layout the real
 * Ultralytics seg export produces — verified via scripts/probe-yolo.mjs). The
 * per-class `scores` are collapsed to (conf = max, classId = argmax), matching
 * how the end2end head reports a single decided class per candidate.
 */
function makeDet(
  cands: Array<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    scores: number[]; // length NUM_CLASSES
    coeffs?: number[]; // length NM
  }>,
  inputSize: number
): { data: Float32Array; dims: number[] } {
  const N = cands.length;
  const data = new Float32Array(C * N);
  for (let i = 0; i < N; i++) {
    const c = cands[i];
    const row = i * C;
    let classId = 0;
    let conf = -1;
    for (let k = 0; k < c.scores.length; k++) {
      if (c.scores[k] > conf) { conf = c.scores[k]; classId = k; }
    }
    data[row + 0] = c.x1;
    data[row + 1] = c.y1;
    data[row + 2] = c.x2;
    data[row + 3] = c.y2;
    data[row + 4] = conf;
    data[row + 5] = classId;
    const coeffs = c.coeffs ?? new Array(NM).fill(0);
    for (let m = 0; m < NM; m++) data[row + 6 + m] = coeffs[m];
  }
  void inputSize;
  return { data, dims: [1, N, C] };
}

describe("normalizeClassName / buildClassNames", () => {
  it("maps the reference HF model's labels onto our schema", () => {
    // Reference model order: 0:frame, 1:text, 2:balloon
    expect(buildClassNames(["frame", "text", "balloon"])).toEqual([
      "panel",
      "text",
      "bubble",
    ]);
  });

  it("accepts common spelling variants (balloon/ballon/bubble, frame/panel)", () => {
    expect(normalizeClassName("Frame")).toBe("panel");
    expect(normalizeClassName("PANEL")).toBe("panel");
    expect(normalizeClassName("balloon")).toBe("bubble");
    expect(normalizeClassName("ballon")).toBe("bubble"); // the repo-name typo
    expect(normalizeClassName("Bubble")).toBe("bubble");
    expect(normalizeClassName("text")).toBe("text");
    expect(normalizeClassName("  TEXT  ")).toBe("text");
  });

  it("maps unknown labels to text", () => {
    expect(normalizeClassName("mystery")).toBe("text");
  });

  it("exposes the reference defaults and training input size", () => {
    expect(DEFAULT_SEG_CLASS_NAMES).toEqual(["frame", "text", "balloon"]);
    expect(DEFAULT_SEG_INPUT_SIZE).toBe(1280);
  });
});

describe("createOnnxProvider", () => {
  it("selects the real provider only when the complete model manifest exists", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ files: [...REQUIRED_MODEL_FILES] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const provider = await createOnnxProvider({ modelsDirUrl: "/models" });

    expect(fetchMock).toHaveBeenCalledWith("/models/manifest.json", {
      cache: "no-store",
    });
    expect(provider.id).toBe("onnx");
    expect(provider.ocr.name).toBe("PP-OCRv6 medium");
    vi.unstubAllGlobals();
  });

  it("rejects provider creation when any required model asset is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ files: ["yolo26s-manga-seg.onnx"] }),
      })
    );

    await expect(createOnnxProvider({ modelsDirUrl: "/models" })).rejects.toThrow(
      "Required model assets are unavailable"
    );
    vi.unstubAllGlobals();
  });
});

describe("letterbox", () => {
  it("produces a CHW tensor of size*size*3 with 114/255 padding", () => {
    const img: DecodedImage = {
      width: 4,
      height: 6,
      data: new Uint8ClampedArray(4 * 6 * 4), // all black
    };
    const size = 8;
    const { tensor, scale, padX, padY } = letterbox(img, size);
    expect(tensor.length).toBe(3 * size * size);
    expect(scale).toBeCloseTo(8 / 6, 4); // min(8/4, 8/6)
    expect(padY).toBe(0);
    expect(padX).toBe(1); // floor((8 - round(4*scale))/2)
    // Unwritten padding cell stays at the Ultralytics gray value.
    expect(tensor[0]).toBeCloseTo(114 / 255, 4);
  });

  it("samples source pixels into the letterboxed canvas correctly", () => {
    const w = 4;
    const h = 6;
    const data = new Uint8ClampedArray(w * h * 4);
    // Source pixel (0,0) = pure red.
    data[0] = 255;
    data[1] = 0;
    data[2] = 0;
    data[3] = 255;
    const size = 8;
    const { tensor, padX, padY } = letterbox({ width: w, height: h, data }, size);
    const plane = size * size;
    // Destination of source (0,0) is (padX, padY) = (1, 0) → di = 0*size + 1 = 1.
    const di = padY * size + padX;
    expect(tensor[di]).toBeCloseTo(1, 4); // R
    expect(tensor[plane + di]).toBeCloseTo(0, 4); // G
    expect(tensor[2 * plane + di]).toBeCloseTo(0, 4); // B
  });
});

describe("postprocessYolo", () => {
  const classNames = buildClassNames(["frame", "text", "balloon"]);

  it("decodes end-to-end xyxy coordinates and reverses letterbox padding", () => {
    const { data, dims } = makeDet(
      [{ x1: 150, y1: 50, x2: 350, y2: 250, scores: [0.9, 0.1, 0.1] }],
      1280
    );

    const dets = postprocessYolo(
      data,
      dims,
      undefined,
      undefined,
      0.5,
      100,
      0,
      1280,
      { width: 1000, height: 1000 },
      classNames
    );

    expect(dets[0].bbox).toEqual({ x: 100, y: 100, w: 400, h: 400 });
  });

  it("filters low confidence, applies per-class NMS, and maps classes", () => {
    const { data, dims } = makeDet(
      [
        // A: strong panel
        { x1: 80, y1: 80, x2: 120, y2: 120, scores: [0.9, 0.1, 0.1] },
        // B: panel almost identical to A but weaker → suppressed by NMS
        // (box (85,85,40,40), IoU with A's (80,80,40,40) ≈ 0.62 > 0.6)
        { x1: 85, y1: 85, x2: 125, y2: 125, scores: [0.5, 0.1, 0.1] },
        // C: text region, separate location → kept
        { x1: 490, y1: 490, x2: 510, y2: 510, scores: [0.1, 0.8, 0.1] },
        // D: below threshold → filtered
        { x1: 690, y1: 690, x2: 710, y2: 710, scores: [0.05, 0.1, 0.05] },
      ],
      1280
    );

    const dets = postprocessYolo(
      data,
      dims,
      undefined,
      undefined,
      1, // scale
      0, // padX
      0, // padY
      1280,
      { width: 1280, height: 1280 },
      classNames
    );

    // A kept, B suppressed, C kept, D filtered → exactly 2.
    expect(dets).toHaveLength(2);
    const byClass = Object.fromEntries(dets.map((d) => [d.className, d]));
    expect(byClass.panel).toBeDefined();
    expect(byClass.text).toBeDefined();
    expect(byClass.bubble).toBeUndefined();

    // Letterbox inverse with scale=1,pad=0 preserves the xyxy box.
    expect(byClass.panel.bbox).toEqual({ x: 80, y: 80, w: 40, h: 40 });
    expect(byClass.panel.confidence).toBeCloseTo(0.9, 5);
    expect(byClass.panel.classId).toBe(0);
    expect(byClass.text.bbox).toEqual({ x: 490, y: 490, w: 20, h: 20 });
    expect(byClass.text.classId).toBe(1);
  });

  it("returns undefined polygon when mask coefficients produce no active pixels", () => {
    // All-zero coefficients → sigmoid(0)=0.5 → not > 0.5 → no points.
    const { data, dims } = makeDet(
      [{ x1: 54, y1: 54, x2: 74, y2: 74, scores: [0.9, 0, 0] }],
      1280
    );
    const proto = new Float32Array(1 * NM * 4 * 4); // all zero
    const dets = postprocessYolo(
      data,
      dims,
      proto,
      [1, NM, 4, 4],
      1,
      0,
      0,
      1280,
      { width: 1280, height: 1280 },
      classNames
    );
    expect(dets).toHaveLength(1);
    expect(dets[0].polygon).toBeUndefined();
  });

  it("clamps boxes to the image bounds", () => {
    const { data, dims } = makeDet(
      // Box crosses the top-left corner and must be clipped to the image.
      [{ x1: -15, y1: -15, x2: 25, y2: 25, scores: [0.9, 0, 0] }],
      1280
    );
    const dets = postprocessYolo(
      data,
      dims,
      undefined,
      undefined,
      1,
      0,
      0,
      1280,
      { width: 1280, height: 1280 },
      classNames
    );
    const b = dets[0].bbox;
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.y).toBeGreaterThanOrEqual(0);
  });
});

describe("decodeMask", () => {
  it("produces image-space points when proto+coeffs activate the mask", () => {
    const mh = 4;
    const mw = 4;
    const proto = new Float32Array(1 * NM * mh * mw);
    // Plane m=0 all = +10; with coeff[0]=1 the logit is +10 → sigmoid ≈ 1.
    for (let p = 0; p < mh * mw; p++) proto[0 * mh * mw + p] = 10;
    const coeffs = new Float32Array(NM);
    coeffs[0] = 1;

    const pts = decodeMask(proto, [1, NM, mh, mw], coeffs, { x: 0, y: 0, w: 1280, h: 1280 }, 1, 0, 0, 1280);
    expect(pts).toBeDefined();
    expect(pts!.length).toBeGreaterThanOrEqual(3);
    // First sampled cell maps to (0*(1280/4), 0*(1280/4)) = (0,0).
    expect(pts![0]).toEqual({ x: 0, y: 0 });
  });
});
