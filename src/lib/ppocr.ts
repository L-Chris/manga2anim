/**
 * PP-OCRv6 det + rec inference pipeline (pure TypeScript, no OpenCV).
 *
 * Implements the standard PaddleOCR post-processing:
 *   Det: DBNet probability map → threshold → connected components → bounding polygons
 *   Rec: crop → resize to h=48 → CTC greedy decode with dictionary
 *
 * All functions are pure and unit-testable without onnxruntime.
 */

import type { BBox, OcrLine } from "../types";

// ---- constants -------------------------------------------------------------

/** ImageNet normalization (PaddleOCR convention). */
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** PP-OCRv6 small detector thresholds from its published inference config. */
const DET_THRESH = 0.2;
const DET_BOX_THRESH = 0.45;
const DET_MIN_SIDE = 3;
/** Polygon expansion ratio (simplified pyclipper unclip). */
const DET_EXPAND_RATIO = 1.4;

/** Recognition fixed height. */
const REC_HEIGHT = 48;
/** Maximum recognition width (padded). */
const REC_MAX_WIDTH = 320;
/** Empirically rejects the short numeric/Latin garbage produced on manga art. */
export const OCR_MIN_CONFIDENCE = 0.8;

/** Only columns this tall relative to their width use vertical segmentation. */
const VERTICAL_MIN_ASPECT_RATIO = 1.8;
/** Prevents an internal gap in a glyph (for example 三) becoming a character cut. */
const VERTICAL_MIN_CHARACTER_PITCH = 0.9;

export interface OcrPipelineOptions {
  /** YOLO text/bubble regions used to reject detector hits on artwork/borders. */
  regions?: readonly BBox[];
  /** Minimum mean CTC confidence required for a line to reach the UI. */
  minConfidence?: number;
}

// ---- dictionary ------------------------------------------------------------

/**
 * Load the PP-OCR dictionary (one character per line).
 * Index 0 = CTC blank (not in file); dictionary[i] corresponds to class i+1.
 *
 * A newline-terminated file (the norm, e.g. ppocr_keys_v1.txt with 6623 lines)
 * splits into a trailing empty element; we drop exactly that one so the array
 * length equals the number of real characters, matching PaddleOCR's loader.
 */
export function parseDictionary(text: string): string[] {
  const lines = text.split("\n").map((line) => line.replace(/\r$/, ""));
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

// ---- det preprocessing -----------------------------------------------------

/**
 * Resize RGBA image for det model: scale so longest side ≤ maxSide and both
 * dims are multiples of 32. Returns CHW float tensor + scale info.
 */
export function detPreprocess(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  maxSide = 960
): { tensor: Float32Array; newW: number; newH: number; scaleW: number; scaleH: number } {
  // Compute target size (multiple of 32).
  let ratio = 1.0;
  const maxDim = Math.max(width, height);
  if (maxDim > maxSide) ratio = maxSide / maxDim;
  let newH = Math.round(height * ratio);
  let newW = Math.round(width * ratio);
  newH = Math.max(32, Math.round(newH / 32) * 32);
  newW = Math.max(32, Math.round(newW / 32) * 32);

  const scaleW = newW / width;
  const scaleH = newH / height;

  // Nearest-neighbor resize + normalize → CHW.
  const plane = newH * newW;
  const tensor = new Float32Array(3 * plane);
  for (let y = 0; y < newH; y++) {
    const sy = Math.min(height - 1, Math.floor(y / scaleH));
    for (let x = 0; x < newW; x++) {
      const sx = Math.min(width - 1, Math.floor(x / scaleW));
      const sp = (sy * width + sx) * 4;
      const di = y * newW + x;
      // Paddle's published detector decodes BGR input before normalization.
      tensor[di] = (data[sp + 2] / 255 - MEAN[0]) / STD[0];
      tensor[plane + di] = (data[sp + 1] / 255 - MEAN[1]) / STD[1];
      tensor[2 * plane + di] = (data[sp] / 255 - MEAN[2]) / STD[2];
    }
  }
  return { tensor, newW, newH, scaleW, scaleH };
}

// ---- det postprocessing ----------------------------------------------------

/**
 * Post-process the DBNet probability map into text bounding boxes.
 * @param probMap flat array of shape [H*W] (row-major), values in [0,1]
 * @param mapH height of the probability map
 * @param mapW width of the probability map
 * @param scaleW horizontal scale from original image to det input
 * @param scaleH vertical scale from original image to det input
 * @param origW original image width
 * @param origH original image height
 */
export function detPostprocess(
  probMap: Float32Array,
  mapH: number,
  mapW: number,
  scaleW: number,
  scaleH: number,
  origW: number,
  origH: number
): BBox[] {
  // 1. Binarize.
  const binary = new Uint8Array(mapH * mapW);
  for (let i = 0; i < probMap.length; i++) {
    binary[i] = probMap[i] > DET_THRESH ? 1 : 0;
  }

  // 2. Connected components (two-pass labeling).
  const labels = new Int32Array(mapH * mapW);
  let nextLabel = 1;
  const parent = new Int32Array(mapH * mapW + 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;

  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  function union(a: number, b: number) {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  }

  for (let y = 0; y < mapH; y++) {
    for (let x = 0; x < mapW; x++) {
      const idx = y * mapW + x;
      if (binary[idx] === 0) continue;
      const top = y > 0 ? labels[(y - 1) * mapW + x] : 0;
      const left = x > 0 ? labels[y * mapW + x - 1] : 0;
      if (top === 0 && left === 0) {
        labels[idx] = nextLabel++;
      } else if (top !== 0 && left === 0) {
        labels[idx] = top;
      } else if (top === 0 && left !== 0) {
        labels[idx] = left;
      } else {
        labels[idx] = Math.min(top, left);
        union(top, left);
      }
    }
  }

  // 3. Compute bounding boxes per component.
  const boxes: BBox[] = [];
  const compBounds = new Map<
    number,
    { x1: number; y1: number; x2: number; y2: number; area: number; probSum: number }
  >();

  for (let y = 0; y < mapH; y++) {
    for (let x = 0; x < mapW; x++) {
      const idx = y * mapW + x;
      if (labels[idx] === 0) continue;
      const root = find(labels[idx]);
      let b = compBounds.get(root);
      if (!b) {
        b = { x1: x, y1: y, x2: x, y2: y, area: 0, probSum: 0 };
        compBounds.set(root, b);
      }
      b.x1 = Math.min(b.x1, x);
      b.y1 = Math.min(b.y1, y);
      b.x2 = Math.max(b.x2, x);
      b.y2 = Math.max(b.y2, y);
      b.area++;
      b.probSum += probMap[idx];
    }
  }

  // 4. Filter by area and expand (simplified unclip).
  for (const [, b] of compBounds) {
    const w = b.x2 - b.x1 + 1;
    const h = b.y2 - b.y1 + 1;
    if (w < DET_MIN_SIDE || h < DET_MIN_SIDE) continue;
    if (b.probSum / b.area < DET_BOX_THRESH) continue;
    const perimeter = 2 * (w + h);
    const expand = (DET_EXPAND_RATIO * b.area) / Math.max(1, perimeter);
    // Map back to original image coords.
    const x1 = Math.max(0, (b.x1 - expand) / scaleW);
    const y1 = Math.max(0, (b.y1 - expand) / scaleH);
    const x2 = Math.min(origW, (b.x2 + 1 + expand) / scaleW);
    const y2 = Math.min(origH, (b.y2 + 1 + expand) / scaleH);
    boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
  }

  return boxes;
}

/** Keep OCR detector boxes that lie mostly inside a YOLO text/bubble region. */
export function filterOcrBoxesByRegions(
  boxes: readonly BBox[],
  regions: readonly BBox[],
  minContainment = 0.5
): BBox[] {
  if (regions.length === 0) return [];
  return boxes.filter((box) => {
    const area = Math.max(0, box.w) * Math.max(0, box.h);
    if (area === 0) return false;
    return regions.some((region) => {
      const centerX = box.x + box.w / 2;
      const centerY = box.y + box.h / 2;
      if (
        centerX >= region.x &&
        centerX <= region.x + region.w &&
        centerY >= region.y &&
        centerY <= region.y + region.h
      ) {
        return true;
      }
      const x1 = Math.max(box.x, region.x);
      const y1 = Math.max(box.y, region.y);
      const x2 = Math.min(box.x + box.w, region.x + region.w);
      const y2 = Math.min(box.y + box.h, region.y + region.h);
      const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      return intersection / area >= minContainment;
    });
  });
}

// ---- rec preprocessing -----------------------------------------------------

/**
 * Crop a text region from the RGBA image, resize to height 48 preserving aspect
 * ratio, pad width to REC_MAX_WIDTH, normalize → CHW tensor.
 * Returns null if the crop is degenerate.
 */
export function recPreprocess(
  data: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  bbox: BBox
): { tensor: Float32Array; width: number } | null {
  const cx = Math.max(0, Math.floor(bbox.x));
  const cy = Math.max(0, Math.floor(bbox.y));
  const cw = Math.min(imgW - cx, Math.ceil(bbox.w));
  const ch = Math.min(imgH - cy, Math.ceil(bbox.h));
  if (cw < 2 || ch < 2) return null;

  // Target width preserving aspect ratio, capped at REC_MAX_WIDTH.
  const ratio = REC_HEIGHT / ch;
  const targetW = Math.min(REC_MAX_WIDTH, Math.max(1, Math.round(cw * ratio)));

  const plane = REC_HEIGHT * REC_MAX_WIDTH;
  // RecResizeImg normalizes BGR pixels to [-1, 1] and then inserts them into a
  // zero-filled [3,48,320] tensor. The untouched right padding therefore stays
  // at normalized value 0 (mid-gray), matching PaddleOCR exactly.
  const tensor = new Float32Array(3 * plane);

  for (let y = 0; y < REC_HEIGHT; y++) {
    const sy = Math.min(ch - 1, Math.floor(y / ratio));
    for (let x = 0; x < targetW; x++) {
      const sx = Math.min(cw - 1, Math.floor(x / ratio));
      const sp = ((cy + sy) * imgW + (cx + sx)) * 4;
      const di = y * REC_MAX_WIDTH + x;
      tensor[di] = data[sp + 2] / 127.5 - 1;
      tensor[plane + di] = data[sp + 1] / 127.5 - 1;
      tensor[2 * plane + di] = data[sp] / 127.5 - 1;
    }
  }
  return { tensor, width: targetW };
}

/**
 * Split a narrow, upright text column into one-character (occasionally
 * two-character) crops using horizontal bands of background pixels.
 *
 * The recognizer expects a horizontal line. Feeding it a whole vertical
 * column shrinks every glyph to a few pixels, while rotating the column turns
 * each Han character sideways. Keeping each returned crop upright preserves
 * the glyph shape and gives it most of the recognizer's 48-pixel input height.
 *
 * Returns null for horizontal/square boxes or when no reliable character gap
 * exists, allowing the caller to keep the normal one-crop recognition path.
 */
export function splitVerticalTextBox(
  data: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  bbox: BBox
): BBox[] | null {
  const x = Math.max(0, Math.floor(bbox.x));
  const y = Math.max(0, Math.floor(bbox.y));
  const w = Math.min(imgW - x, Math.ceil(bbox.w));
  const h = Math.min(imgH - y, Math.ceil(bbox.h));
  if (w < 3 || h < 3 || h / w < VERTICAL_MIN_ASPECT_RATIO) return null;

  // Estimate the light background from the upper luminance quartile instead
  // of assuming pure white. This still works on lightly screened speech
  // bubbles while keeping gray antialiasing around black glyphs as foreground.
  const histogram = new Uint32Array(256);
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      const p = ((y + yy) * imgW + x + xx) * 4;
      const luminance = Math.round(
        0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]
      );
      histogram[luminance]++;
    }
  }
  const target = Math.ceil(w * h * 0.8);
  let seen = 0;
  let background = 255;
  for (let value = 0; value < histogram.length; value++) {
    seen += histogram[value];
    if (seen >= target) {
      background = value;
      break;
    }
  }
  const darkThreshold = Math.max(80, Math.min(210, background - 40));

  const rowInk = new Uint16Array(h);
  const columnInk = new Uint16Array(w);
  for (let yy = 0; yy < h; yy++) {
    let ink = 0;
    for (let xx = 0; xx < w; xx++) {
      const p = ((y + yy) * imgW + x + xx) * 4;
      const luminance =
        0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      if (luminance <= darkThreshold) {
        ink++;
        columnInk[xx]++;
      }
    }
    rowInk[yy] = ink;
  }

  // DB's polygon expansion deliberately leaves generous horizontal padding.
  // Character pitch must be based on the actual ink span, otherwise a 23 px
  // glyph inside a 47 px detector box would be mistaken for half a character.
  // Ignore isolated antialias/bubble pixels that occur in only a few rows;
  // otherwise one outlier can make the inferred glyph width almost as wide as
  // the detector's padded box and cause several characters to be merged.
  const minColumnInk = Math.max(2, Math.floor(h * 0.03));
  let inkLeft = 0;
  while (inkLeft < w && columnInk[inkLeft] < minColumnInk) inkLeft++;
  let inkRight = w - 1;
  while (inkRight >= inkLeft && columnInk[inkRight] < minColumnInk) inkRight--;
  let inkWidth = inkRight - inkLeft + 1;
  if (inkWidth < Math.max(3, Math.round(w * 0.2))) {
    inkLeft = 0;
    inkRight = w - 1;
    inkWidth = w;
  }
  const horizontalPadding = Math.max(1, Math.round(inkWidth * 0.08));
  const cropLeft = Math.max(0, inkLeft - horizontalPadding);
  const cropRight = Math.min(w, inkRight + 1 + horizontalPadding);

  // A one-pixel bubble edge can run through the crop. Requiring more than a
  // small fraction of the width to be dark keeps such an edge from hiding the
  // whitespace between characters.
  const maxGapInk = Math.max(1, Math.floor(inkWidth * 0.08));
  const minGapHeight = Math.max(3, Math.round(inkWidth * 0.12));
  const gapCenters: number[] = [];
  let gapStart = -1;
  for (let yy = 0; yy <= h; yy++) {
    const isGap = yy < h && rowInk[yy] <= maxGapInk;
    if (isGap && gapStart < 0) gapStart = yy;
    if (!isGap && gapStart >= 0) {
      // Outer padding is not a character boundary.
      if (yy - gapStart >= minGapHeight && gapStart > 0 && yy < h) {
        gapCenters.push(Math.round((gapStart + yy) / 2));
      }
      gapStart = -1;
    }
  }
  if (gapCenters.length === 0) return null;

  // Select well-spaced gaps. This deliberately ignores large blank bands
  // inside sparse glyphs such as 三 until roughly one character pitch has
  // elapsed, so a returned crop contains one or at most two upright glyphs.
  const minPitch = Math.max(5, Math.round(inkWidth * VERTICAL_MIN_CHARACTER_PITCH));
  const boundaries = [0];
  for (const center of gapCenters) {
    if (center - boundaries[boundaries.length - 1] >= minPitch) {
      boundaries.push(center);
    }
  }
  if (boundaries.length === 1) return null;

  const segments: BBox[] = [];
  for (let index = 0; index < boundaries.length; index++) {
    const top = boundaries[index];
    const bottom = index + 1 < boundaries.length ? boundaries[index + 1] : h;
    // A short tail is normally an ellipsis/exclamation dot. Passing it to the
    // recognizer often produces high-confidence "1"/"20" garbage, so leave it
    // out rather than exposing it as dialogue text.
    if (bottom - top < minPitch) continue;
    segments.push({
      x: x + cropLeft,
      y: y + top,
      w: cropRight - cropLeft,
      h: bottom - top,
    });
  }

  return segments.length >= 2 ? segments : null;
}

/**
 * Rec-preprocess a *standalone* RGBA crop buffer (whole buffer = the crop), as
 * opposed to {@link recPreprocess} which extracts a bbox from a full image.
 * Used for the orientation-aware region path where we rotate a crop first.
 */
export function recPreprocessBuffer(
  data: Uint8ClampedArray,
  w: number,
  h: number
): { tensor: Float32Array; width: number } | null {
  return recPreprocess(data, w, h, { x: 0, y: 0, w, h });
}

/**
 * Rotate an RGBA buffer 90° clockwise. Pure + unit-tested. k applications give
 * k*90° CW. Used to feed vertical manga text to the horizontal rec model: a
 * vertical column rotated 90° becomes a horizontal line the model can read.
 */
export function rotateRgba90cw(
  data: Uint8ClampedArray,
  w: number,
  h: number
): { data: Uint8ClampedArray; w: number; h: number } {
  const nw = h;
  const nh = w;
  const out = new Uint8ClampedArray(nw * nh * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const X = h - 1 - y;
      const Y = x;
      const si = (y * w + x) * 4;
      const di = (X * nw + Y) * 4;
      out[di] = data[si];
      out[di + 1] = data[si + 1];
      out[di + 2] = data[si + 2];
      out[di + 3] = data[si + 3];
    }
  }
  return { data: out, w: nw, h: nh };
}

function rotateRgba(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  k: number
): { data: Uint8ClampedArray; w: number; h: number } {
  let cur = { data, w, h };
  for (let i = 0; i < ((k % 4) + 4) % 4; i++) cur = rotateRgba90cw(cur.data, cur.w, cur.h);
  return cur;
}

/** Extract an RGBA crop from a full image (clamped to bounds). */
export function cropRgba(
  data: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  bbox: BBox
): { data: Uint8ClampedArray; w: number; h: number } | null {
  const cx = Math.max(0, Math.floor(bbox.x));
  const cy = Math.max(0, Math.floor(bbox.y));
  const cw = Math.min(imgW - cx, Math.ceil(bbox.w));
  const ch = Math.min(imgH - cy, Math.ceil(bbox.h));
  if (cw < 2 || ch < 2) return null;
  const out = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const si = ((cy + y) * imgW + (cx + x)) * 4;
      const di = (y * cw + x) * 4;
      out[di] = data[si];
      out[di + 1] = data[si + 1];
      out[di + 2] = data[si + 2];
      out[di + 3] = 255;
    }
  }
  return { data: out, w: cw, h: ch };
}

// ---- rec postprocessing (CTC greedy decode) --------------------------------

/**
 * CTC greedy decode: given logits [T, numClasses], produce the decoded string.
 * @param logits flat array of shape [T * numClasses], row-major
 * @param T number of timesteps
 * @param numClasses vocabulary size (including blank at index 0)
 * @param dictionary array of characters (index 0 in dictionary = class 1)
 */
export function ctcDecode(
  logits: Float32Array,
  T: number,
  numClasses: number,
  dictionary: string[]
): { text: string; confidence: number } {
  const indices: number[] = [];
  const confidences: number[] = [];
  let prevIdx = -1;

  for (let t = 0; t < T; t++) {
    // Argmax over numClasses.
    let bestIdx = 0;
    let bestVal = -Infinity;
    const offset = t * numClasses;
    for (let c = 0; c < numClasses; c++) {
      const v = logits[offset + c];
      if (v > bestVal) { bestVal = v; bestIdx = c; }
    }
    // CTC: collapse repeats and remove blank (index 0).
    if (bestIdx !== 0 && bestIdx !== prevIdx) {
      indices.push(bestIdx);
      confidences.push(bestVal);
    }
    prevIdx = bestIdx;
  }

  // Map indices to characters. PaddleOCR convention: class 0 = blank (removed
  // above); classes 1..N map to dictionary[0..N-1]; class N+1 (i.e. charIdx ===
  // dictionary.length) is the appended trailing space character.
  let text = "";
  for (const idx of indices) {
    const charIdx = idx - 1;
    if (charIdx >= 0 && charIdx < dictionary.length) text += dictionary[charIdx];
    else if (charIdx === dictionary.length) text += " ";
  }

  const confidence = confidences.length > 0
    ? confidences.reduce((s, v) => s + v, 0) / confidences.length
    : 0;

  return { text, confidence };
}

// ---- full pipeline helper --------------------------------------------------

/**
 * Run det + rec on a full image. This is the high-level entry point called by
 * the ONNX provider's PpOcrEngine.
 *
 * @param runDet  callback that takes [1,3,H,W] tensor and returns [1,1,H,W] prob map
 * @param runRec  callback that takes [1,3,48,W] tensor and returns [1,T,C] logits
 * @param dictionary  parsed ppocrv6_dict.txt
 * @param options  YOLO region constraints and recognizer confidence threshold
 */
export async function runOcrPipeline(
  imgData: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  runDet: (tensor: Float32Array, h: number, w: number) => Promise<Float32Array>,
  runRec: (tensor: Float32Array, width: number) => Promise<{ logits: Float32Array; T: number; numClasses: number }>,
  dictionary: string[],
  options: OcrPipelineOptions = {}
): Promise<OcrLine[]> {
  // 1. Detection.
  const { tensor: detTensor, newW, newH, scaleW, scaleH } = detPreprocess(imgData, imgW, imgH);
  const probMap = await runDet(detTensor, newH, newW);
  const detectedBoxes = detPostprocess(
    probMap,
    newH,
    newW,
    scaleW,
    scaleH,
    imgW,
    imgH
  );
  const boxes = options.regions === undefined
    ? detectedBoxes
    : filterOcrBoxesByRegions(detectedBoxes, options.regions);
  const minConfidence = options.minConfidence ?? OCR_MIN_CONFIDENCE;

  // 2. Recognition per box.
  const lines: OcrLine[] = [];
  for (const bbox of boxes) {
    const verticalSegments = splitVerticalTextBox(imgData, imgW, imgH, bbox);
    const recognitionBoxes = verticalSegments ?? [bbox];
    let text = "";
    let confidenceSum = 0;
    let confidenceCharacters = 0;

    // Vertical segments are already ordered from top to bottom. Horizontal
    // boxes contain just the original bbox, so both paths share decoding and
    // confidence filtering here.
    for (const recognitionBox of recognitionBoxes) {
      const crop = recPreprocess(imgData, imgW, imgH, recognitionBox);
      if (!crop) continue;
      const { logits, T, numClasses } = await runRec(crop.tensor, crop.width);
      const decoded = ctcDecode(logits, T, numClasses, dictionary);
      const decodedText = decoded.text.trim();
      if (decodedText.length === 0 || decoded.confidence < minConfidence) continue;
      text += decodedText;
      const characterCount = [...decodedText].length;
      confidenceSum += decoded.confidence * characterCount;
      confidenceCharacters += characterCount;
    }

    if (text.length === 0 || confidenceCharacters === 0) continue;
    lines.push({
      text,
      confidence: confidenceSum / confidenceCharacters,
      // The UI associates the reconstructed column with the detector's
      // original box rather than exposing one line per character.
      bbox,
    });
  }

  return lines;
}

/**
 * Region-mode OCR (the spec's intended flow): the segmentation model already
 * located text/bubble regions, so we skip the scene-text *detector* (which on
 * manga art fires on panel borders, not dialogue) and run the *recognizer*
 * directly on each region crop.
 *
 * Because PP-OCR's rec head is a horizontal-line model and manga dialogue is
 * vertical, each crop is tried in all four 90° rotations and the best non-empty
 * decode (by text length, then confidence) is kept. Empirically (see
 * scripts/probe-crop-rec.mjs) this recovers the horizontal text the model CAN
 * read (chapter markers, page numbers, labels) and returns "" on vertical
 * dialogue / empty art — the latter is filtered out, so unlike full-page det+rec
 * it does NOT emit border-garbage. Vertical-column reading is outside this
 * lightweight horizontal model's capability and is reported honestly as empty.
 *
 * @param regions  text/bubble bboxes (image coords), e.g. from YOLO class 1/2.
 */
export async function runRegionOcr(
  imgData: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  regions: readonly BBox[],
  runRec: (tensor: Float32Array, width: number) => Promise<{ logits: Float32Array; T: number; numClasses: number }>,
  dictionary: string[]
): Promise<OcrLine[]> {
  const lines: OcrLine[] = [];
  for (const region of regions) {
    const crop = cropRgba(imgData, imgW, imgH, region);
    if (!crop) continue;

    let best: { text: string; confidence: number } | null = null;
    for (let k = 0; k < 4; k++) {
      const rot = rotateRgba(crop.data, crop.w, crop.h, k);
      const prep = recPreprocessBuffer(rot.data, rot.w, rot.h);
      if (!prep) continue;
      const { logits, T, numClasses } = await runRec(prep.tensor, prep.width);
      const dec = ctcDecode(logits, T, numClasses, dictionary);
      const t = dec.text.trim();
      if (t.length === 0) continue;
      if (
        !best ||
        t.length > best.text.length ||
        (t.length === best.text.length && dec.confidence > best.confidence)
      ) {
        best = { text: dec.text, confidence: dec.confidence };
      }
    }
    if (best) lines.push({ text: best.text, confidence: best.confidence, bbox: region });
  }
  return lines;
}
