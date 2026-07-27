import * as ort from "onnxruntime-web";
import type { BBox, DetectionClass, OcrLine, Point, RawDetection } from "../../types";
import { clampBBox, nms } from "../geometry";
import { parseDictionary, runOcrPipeline } from "../ppocr";
import type {
  DecodedImage,
  InferenceProvider,
  OcrEngine,
  ProgressFn,
  Segmenter,
} from "./types";

/**
 * ONNX inference provider — the production path.
 *
 * Loads two ONNX models from a directory the caller supplies (the Tauri app
 * config dir /models):
 *
 *   yolo26s-manga-seg.onnx   YOLO26s manga instance segmentation
 *   ppocrv6-det.onnx         PP-OCRv6 small text detector
 *   ppocrv6-rec.onnx         PP-OCRv6 small text recognizer
 *
 * Class mapping is **model-driven**, not hardcoded. The reference model
 * (ShadowB/Manga109-panel-balloon-text-yolov26-segmentation) stores its labels
 * as `0:frame, 1:text, 2:balloon` and was trained at `imgsz=1280`. We normalize
 * whatever names the model reports into our internal 3-class schema
 * (panel / text / bubble) via {@link normalizeClassName}, so a model that calls
 * the panel class `frame` and the bubble class `balloon`/`ballon` still maps
 * correctly. The default name order below matches that reference model exactly
 * and is used when the ONNX file carries no readable `names` metadata.
 *
 * Provider creation requires a complete generated model manifest. Missing
 * assets are reported as an error instead of selecting a fallback algorithm.
 *
 * Post-processing follows the standard Ultralytics YOLO-seg layout:
 *   output0: [1, 4 + numClasses + 32, numDetections]
 *   output1 (proto): [1, 32, maskH, maskW]
 * and the standard PaddleOCR det (DB) / rec (CTC) pipelines.
 */

/**
 * Default class names, in model class-id order, matching the reference HF
 * model's `data.yaml`. Normalized to our schema at construction time.
 */
export const DEFAULT_SEG_CLASS_NAMES = ["frame", "text", "balloon"];

/** Default segmentation input size — matches the reference model's training. */
export const DEFAULT_SEG_INPUT_SIZE = 1280;

/** Files that must be present before the production provider is selected. */
export const REQUIRED_MODEL_FILES = [
  "yolo26s-manga-seg.onnx",
  "ppocrv6-det.onnx",
  "ppocrv6-rec.onnx",
  "ppocrv6_dict.txt",
] as const;

/**
 * Map a raw model label (any casing/spelling) onto our internal 3-class schema.
 * `frame`/`panel` → panel; `balloon`/`ballon`/`bubble` → bubble; `text` → text.
 * Unrecognized labels fall back to `text` (visible content); the model's true
 * class id is preserved on the detection for any manual correction.
 */
export function normalizeClassName(raw: string): DetectionClass {
  const s = raw.trim().toLowerCase();
  if (s === "frame" || s === "panel" || s === "panels" || s === "frames") return "panel";
  if (s === "balloon" || s === "ballon" || s === "balloons" || s === "bubble" || s === "bubbles")
    return "bubble";
  if (s === "text" || s === "texts" || s === "text_region") return "text";
  return "text";
}

/** Build the normalized, model-ordered class list used during decode. */
export function buildClassNames(rawNames: readonly string[]): DetectionClass[] {
  return rawNames.map(normalizeClassName);
}

export interface OnnxProviderOptions {
  modelsDirUrl: string; // base URL (directory) the weights are served from
  segModel?: string;
  detModel?: string;
  recModel?: string;
  /** Override the model's class names (model class-id order). */
  segClassNames?: string[];
  /** Override the segmentation input size (must match how the model was exported). */
  segInputSize?: number;
  /** Dictionary filename (default: ppocrv6_dict.txt). */
  dictModel?: string;
}

// ---- image preprocessing ---------------------------------------------------

/** Letterbox-resize RGBA image to `size`×`size`, returning a CHW float tensor. */
export function letterbox(
  img: DecodedImage,
  size: number
): { tensor: Float32Array; scale: number; padX: number; padY: number } {
  const { width, height, data } = img;
  const scale = Math.min(size / width, size / height);
  const newW = Math.round(width * scale);
  const newH = Math.round(height * scale);
  const padX = Math.floor((size - newW) / 2);
  const padY = Math.floor((size - newH) / 2);

  // 114 gray padding (Ultralytics convention), CHW, normalized to [0,1].
  const tensor = new Float32Array(3 * size * size).fill(114 / 255);
  const plane = size * size;

  // Bilinear-ish nearest sampling from source into the letterboxed canvas.
  for (let y = 0; y < newH; y++) {
    const sy = Math.min(height - 1, Math.floor(y / scale));
    for (let x = 0; x < newW; x++) {
      const sx = Math.min(width - 1, Math.floor(x / scale));
      const sp = (sy * width + sx) * 4;
      const dx = padX + x;
      const dy = padY + y;
      const di = dy * size + dx;
      tensor[di] = data[sp] / 255;
      tensor[plane + di] = data[sp + 1] / 255;
      tensor[2 * plane + di] = data[sp + 2] / 255;
    }
  }
  return { tensor, scale, padX, padY };
}

// ---- YOLO26s segmentation --------------------------------------------------

class YoloSegmenter implements Segmenter {
  readonly name = "YOLO26s Manga Instance Segmentation";
  private session: ort.InferenceSession | null = null;
  private readonly inputSize: number;
  private readonly classNames: DetectionClass[];

  constructor(
    private modelUrl: string,
    classNames: DetectionClass[] = buildClassNames(DEFAULT_SEG_CLASS_NAMES),
    inputSize: number = DEFAULT_SEG_INPUT_SIZE
  ) {
    this.classNames = classNames;
    this.inputSize = inputSize;
  }

  private async ensureSession(): Promise<ort.InferenceSession> {
    if (!this.session) {
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ["wasm"],
      });
    }
    return this.session;
  }

  async segment(img: DecodedImage, onProgress?: ProgressFn): Promise<RawDetection[]> {
    onProgress?.(0.05, "Loading YOLO26s model");
    const session = await this.ensureSession();
    const { tensor, scale, padX, padY } = letterbox(img, this.inputSize);

    onProgress?.(0.35, "Running segmentation");
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: new ort.Tensor("float32", tensor, [
        1,
        3,
        this.inputSize,
        this.inputSize,
      ]),
    };
    const results = await session.run(feeds);
    const outNames = session.outputNames;
    const detOut = results[outNames[0]];
    const protoOut = outNames.length > 1 ? results[outNames[1]] : undefined;

    onProgress?.(0.7, "Post-processing detections");
    const dets = postprocessYolo(
      detOut.data as Float32Array,
      detOut.dims as number[],
      protoOut ? (protoOut.data as Float32Array) : undefined,
      protoOut ? (protoOut.dims as number[]) : undefined,
      scale,
      padX,
      padY,
      this.inputSize,
      { width: img.width, height: img.height },
      this.classNames
    );
    onProgress?.(1, "Segmentation complete");
    return dets;
  }
}

// ---- pure post-processing (unit-testable without onnxruntime) --------------

/**
 * Decode a YOLO-seg detection tensor into RawDetections. Pure function so the
 * decode path (confidence filter, class map, letterbox inverse, per-class NMS,
 * mask decode) can be verified with synthetic tensors — no model weights needed.
 *
 * det layout (Ultralytics **end2end** seg export, verified empirically against
 * the reference model — see scripts/probe-yolo.mjs): [1, N, C] **row-major**,
 * one row per top-k candidate:
 *   cols 0..3           = x1, y1, x2, y2 (letterboxed input space)
 *   col  4              = confidence (already sigmoid, [0,1])
 *   col  5              = class id (integer 0..nc-1)
 *   cols 6..6+nm-1      = mask coefficients (nm = 32)
 * proto layout: [1, nm, mh, mw].
 *
 * NOTE: this is NOT the raw [1, 4+nc+nm, N] column-major layout — the end2end
 * export collapses the per-class scores into a single (conf, class_id) pair.
 */
export function postprocessYolo(
  detData: Float32Array,
  detDims: number[],
  protoData: Float32Array | undefined,
  protoDims: number[] | undefined,
  scale: number,
  padX: number,
  padY: number,
  inputSize: number,
  img: { width: number; height: number },
  classNames: readonly DetectionClass[] = buildClassNames(DEFAULT_SEG_CLASS_NAMES),
  confThresh = 0.25
): RawDetection[] {
  // Row-major end2end layout: [1, N, C], C = 4 (xyxy) + 1 (conf) + 1 (class) + nm.
  const N = detDims[1];
  const C = detDims[2];
  const nm = Math.max(0, C - 6);

  const boxes: BBox[] = [];
  const scores: number[] = [];
  const classIds: number[] = [];
  const maskCoeffs: Float32Array[] = [];

  for (let i = 0; i < N; i++) {
    const row = i * C;
    const conf = detData[row + 4]; // already sigmoid
    if (conf < confThresh) continue;

    let classId = Math.round(detData[row + 5]);
    if (classId < 0 || classId >= classNames.length) continue;

    const x1 = (detData[row + 0] - padX) / scale;
    const y1 = (detData[row + 1] - padY) / scale;
    const x2 = (detData[row + 2] - padX) / scale;
    const y2 = (detData[row + 3] - padY) / scale;
    boxes.push(
      clampBBox(
        { x: x1, y: y1, w: x2 - x1, h: y2 - y1 },
        img.width,
        img.height
      )
    );
    scores.push(conf);
    classIds.push(classId);
    if (nm > 0) {
      const coeffs = new Float32Array(nm);
      for (let m = 0; m < nm; m++) coeffs[m] = detData[row + 6 + m];
      maskCoeffs.push(coeffs);
    }
  }

  // Per-class NMS (class ids are now explicit, not one-hot).
  const keep = new Set<number>();
  const present = new Set(classIds);
  for (const c of present) {
    const idx = boxes.map((_, i) => i).filter((i) => classIds[i] === c);
    const subBoxes = idx.map((i) => boxes[i]);
    const subScores = idx.map((i) => scores[i]);
    const kept = nms(subBoxes, subScores, 0.6);
    for (const k of kept) keep.add(idx[k]);
  }

  const detections: RawDetection[] = [];
  for (const i of keep) {
    const polygon =
      protoData && protoDims && maskCoeffs[i]
        ? decodeMask(
            protoData,
            protoDims,
            maskCoeffs[i],
            boxes[i],
            scale,
            padX,
            padY,
            inputSize
          )
        : undefined;
    detections.push({
      classId: classIds[i],
      className: classNames[classIds[i]] ?? "text",
      confidence: scores[i],
      bbox: boxes[i],
      polygon,
    });
  }
  return detections;
}

/** Combine mask coefficients with proto masks, threshold, map to image space. */
export function decodeMask(
  protoData: Float32Array,
  protoDims: number[],
  coeffs: Float32Array,
  bbox: BBox,
  scale: number,
  padX: number,
  padY: number,
  inputSize: number
): Point[] | undefined {
  const [, nm, mh, mw] = protoDims;
  const mask = new Float32Array(mh * mw);
  for (let p = 0; p < mh * mw; p++) {
    let v = 0;
    for (let m = 0; m < nm; m++) v += coeffs[m] * protoData[m * mh * mw + p];
    mask[p] = 1 / (1 + Math.exp(-v)); // sigmoid
  }
  const pts: Point[] = [];
  const step = Math.max(1, Math.floor(mh / 80));
  for (let y = 0; y < mh; y += step) {
    for (let x = 0; x < mw; x += step) {
      if (mask[y * mw + x] > 0.5) {
        pts.push({
          x: (x * (inputSize / mw) - padX) / scale,
          y: (y * (inputSize / mh) - padY) / scale,
        });
      }
    }
  }
  if (pts.length < 3) return undefined;
  return pts
    .filter(
      (p) =>
        p.x >= bbox.x &&
        p.x <= bbox.x + bbox.w &&
        p.y >= bbox.y &&
        p.y <= bbox.y + bbox.h
    )
    .slice(0, 200);
}

// ---- PP-OCRv6 --------------------------------------------------------------

class PpOcrEngine implements OcrEngine {
  readonly name = "PP-OCRv6 small";
  private detSession: ort.InferenceSession | null = null;
  private recSession: ort.InferenceSession | null = null;
  private dictionary: string[] | null = null;

  constructor(
    private detUrl: string,
    private recUrl: string,
    private dictUrl: string
  ) {}

  private async ensure(): Promise<void> {
    if (!this.detSession || !this.recSession || !this.dictionary) {
      this.detSession = await ort.InferenceSession.create(this.detUrl, {
        executionProviders: ["wasm"],
      });
      this.recSession = await ort.InferenceSession.create(this.recUrl, {
        executionProviders: ["wasm"],
      });
      // Load dictionary (one character per line).
      const dictRes = await fetch(this.dictUrl);
      if (!dictRes.ok) throw new Error(`Dictionary fetch failed: ${dictRes.status}`);
      const dictText = await dictRes.text();
      this.dictionary = parseDictionary(dictText);
    }
  }

  async recognize(
    img: DecodedImage,
    regions: readonly BBox[],
    onProgress?: ProgressFn
  ): Promise<OcrLine[]> {
    await this.ensure();
    const dict = this.dictionary!;
    const detSession = this.detSession!;
    const recSession = this.recSession!;

    onProgress?.(0.1, "Running PP-OCRv6 detection");

    const runDet = async (tensor: Float32Array, h: number, w: number): Promise<Float32Array> => {
      const feeds: Record<string, ort.Tensor> = {
        [detSession.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, h, w]),
      };
      const results = await detSession.run(feeds);
      const out = results[detSession.outputNames[0]];
      return out.data as Float32Array;
    };

    const runRec = async (tensor: Float32Array, width: number): Promise<{ logits: Float32Array; T: number; numClasses: number }> => {
      // `width` is the unpadded crop width; the rec model takes a fixed 320-wide
      // padded tensor. Kept in the signature for future dynamic-width rec models.
      void width;
      const feeds: Record<string, ort.Tensor> = {
        [recSession.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, 48, 320]),
      };
      const results = await recSession.run(feeds);
      const out = results[recSession.outputNames[0]];
      const dims = out.dims as number[]; // [1, T, numClasses]
      const T = dims[1];
      const numClasses = dims[2];
      return { logits: out.data as Float32Array, T, numClasses };
    };

    const lines = await runOcrPipeline(
      img.data,
      img.width,
      img.height,
      runDet,
      runRec,
      dict,
      { regions }
    );

    onProgress?.(1, "OCR complete");
    return lines;
  }
}

// ---- provider --------------------------------------------------------------

export async function createOnnxProvider(
  opts: OnnxProviderOptions
): Promise<InferenceProvider> {
  const segUrl = `${opts.modelsDirUrl}/${opts.segModel ?? "yolo26s-manga-seg.onnx"}`;
  const detUrl = `${opts.modelsDirUrl}/${opts.detModel ?? "ppocrv6-det.onnx"}`;
  const recUrl = `${opts.modelsDirUrl}/${opts.recModel ?? "ppocrv6-rec.onnx"}`;
  const dictUrl = `${opts.modelsDirUrl}/${opts.dictModel ?? "ppocrv6_dict.txt"}`;

  // Probe a tiny generated manifest instead of downloading the segmentation
  // model twice (once for availability and once for the ONNX session).
  try {
    const res = await fetch(`${opts.modelsDirUrl}/manifest.json`, {
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`model manifest request failed with HTTP ${res.status}`);
    }
    const manifest = (await res.json()) as { files?: unknown };
    const files = manifest.files;
    if (
      !Array.isArray(files) ||
      !REQUIRED_MODEL_FILES.every((file) => files.includes(file))
    ) {
      throw new Error("model manifest does not contain every required asset");
    }
  } catch (error) {
    throw new Error(`Required model assets are unavailable: ${String(error)}`);
  }

  const classNames = buildClassNames(
    opts.segClassNames ?? DEFAULT_SEG_CLASS_NAMES
  );
  const inputSize = opts.segInputSize ?? DEFAULT_SEG_INPUT_SIZE;

  return {
    id: "onnx",
    label: "YOLO26s + PP-OCRv6 (ONNX)",
    segmenter: new YoloSegmenter(segUrl, classNames, inputSize),
    ocr: new PpOcrEngine(detUrl, recUrl, dictUrl),
  };
}
