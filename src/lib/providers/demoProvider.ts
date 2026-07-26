import type { BBox, OcrLine, RawDetection } from "../../types";
import { clampBBox } from "../geometry";
import type {
  DecodedImage,
  InferenceProvider,
  OcrEngine,
  ProgressFn,
  Segmenter,
} from "./types";

/**
 * Demo inference provider — no model weights required.
 *
 * It performs genuine lightweight image analysis so the whole pipeline is
 * exercisable end-to-end:
 *
 *  - Segmentation: detects the near-white "gutters" that separate manga panels
 *    (horizontal runs of light pixels spanning most of the width, and vertical
 *    runs spanning most of the height), then reconstructs panel rectangles from
 *    the resulting grid. Bubble/text regions are synthesized inside each panel.
 *  - OCR: emits placeholder text lines positioned on the synthesized bubbles.
 *
 * When real YOLO26s / PP-OCRv6 ONNX weights are dropped into the models dir, the
 * ONNX provider takes over and this one is only a fallback.
 */

// ---- grayscale helpers -----------------------------------------------------

function toGray(img: DecodedImage): Float32Array {
  const { width, height, data } = img;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    // Rec. 601 luma
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  return gray;
}

/**
 * For each row, the fraction of pixels that are "light" (above threshold).
 * Gutters between panels are rows that are almost entirely light.
 */
function rowLightFraction(gray: Float32Array, w: number, h: number, thresh: number): Float32Array {
  const out = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let light = 0;
    const base = y * w;
    for (let x = 0; x < w; x++) {
      if (gray[base + x] >= thresh) light++;
    }
    out[y] = light / w;
  }
  return out;
}

/**
 * Column-light fraction computed only over rows [y0, y1). Used by the row-first
 * XY-cut: vertical gutters are detected *per row band*, so a full-width panel
 * (whose band has no vertical gutter) is never split.
 */
function colLightFractionInBand(
  gray: Float32Array,
  w: number,
  y0: number,
  y1: number,
  thresh: number
): Float32Array {
  const bandH = Math.max(1, y1 - y0);
  const out = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let light = 0;
    for (let y = y0; y < y1; y++) {
      if (gray[y * w + x] >= thresh) light++;
    }
    out[x] = light / bandH;
  }
  return out;
}

/**
 * Collapse a per-line "is gutter" boolean array into gap intervals
 * [start, end) in pixels, ignoring gaps thinner than minGap.
 */
function gutterIntervals(frac: Float32Array, gutterFrac: number, minGap: number): Array<[number, number]> {
  const gaps: Array<[number, number]> = [];
  let start = -1;
  for (let i = 0; i < frac.length; i++) {
    const isGutter = frac[i] >= gutterFrac;
    if (isGutter && start === -1) start = i;
    if ((!isGutter || i === frac.length - 1) && start !== -1) {
      const end = isGutter ? i + 1 : i;
      if (end - start >= minGap) gaps.push([start, end]);
      start = -1;
    }
  }
  return gaps;
}

/** Turn gutter gaps into content segments (the spaces between gaps). */
function segmentsFromGaps(length: number, gaps: Array<[number, number]>, margin: number): Array<[number, number]> {
  const segs: Array<[number, number]> = [];
  let cursor = margin;
  for (const [g0, g1] of gaps) {
    if (g0 - cursor > margin) segs.push([cursor, g0]);
    cursor = g1;
  }
  if (length - cursor > margin) segs.push([cursor, length - margin]);
  if (segs.length === 0) segs.push([margin, length - margin]);
  return segs;
}

// ---- demo segmenter --------------------------------------------------------

class DemoSegmenter implements Segmenter {
  readonly name = "Demo gutter-based segmenter";

  async segment(img: DecodedImage, onProgress?: ProgressFn): Promise<RawDetection[]> {
    onProgress?.(0.1, "Analyzing page structure");
    const { width, height } = img;
    const gray = toGray(img);
    const lightThresh = 235; // near-white
    const gutterFrac = 0.97; // a gutter row is ≥97% light
    const minGap = Math.max(4, Math.round(Math.min(width, height) * 0.006));
    const margin = Math.max(2, Math.round(Math.min(width, height) * 0.01));

    onProgress?.(0.4, "Detecting panel gutters");
    const rowFrac = rowLightFraction(gray, width, height, lightThresh);
    const hGaps = gutterIntervals(rowFrac, gutterFrac, minGap);
    const rowSegs = segmentsFromGaps(height, hGaps, margin);

    onProgress?.(0.7, "Reconstructing panels (row-first XY-cut)");
    const detections: RawDetection[] = [];

    // Row-first XY-cut: horizontal gutters are detected globally (rows read
    // top→bottom regardless of layout), but vertical gutters are detected
    // *within each row band only*. A full-width panel (e.g. a banner spanning
    // both columns) has no vertical gutter inside its band, so it stays whole
    // instead of being sliced by a global column grid.
    for (const [y0, y1] of rowSegs) {
      const bandColFrac = colLightFractionInBand(gray, width, y0, y1, lightThresh);
      const vGaps = gutterIntervals(bandColFrac, gutterFrac, minGap);
      const colSegs = segmentsFromGaps(width, vGaps, margin);
      for (const [x0, x1] of colSegs) {
        const bbox = clampBBox(
          { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
          width,
          height
        );
        if (bbox.w < 8 || bbox.h < 8) continue;
        if (!hasContent(gray, width, bbox, lightThresh)) continue;
        detections.push({
          classId: 0,
          className: "panel",
          confidence: 0.9,
          bbox,
        });
      }
    }

    // If gutter detection found nothing (e.g. borderless page), fall back to a
    // single full-page panel so downstream stages still have something to do.
    if (detections.length === 0) {
      detections.push({
        classId: 0,
        className: "panel",
        confidence: 0.5,
        bbox: { x: margin, y: margin, w: width - 2 * margin, h: height - 2 * margin },
      });
    }

    // Synthesize one speech bubble + one text region inside each panel so OCR
    // and classification have inputs to work with.
    onProgress?.(0.9, "Synthesizing text regions");
    for (const panel of detections.filter((d) => d.className === "panel")) {
      const b = panel.bbox;
      const bubble: BBox = clampBBox(
        {
          x: b.x + b.w * 0.5,
          y: b.y + b.h * 0.08,
          w: b.w * 0.42,
          h: b.h * 0.22,
        },
        width,
        height
      );
      detections.push({
        classId: 2,
        className: "bubble",
        confidence: 0.7,
        bbox: bubble,
      });
      const text: BBox = clampBBox(
        {
          x: b.x + b.w * 0.08,
          y: b.y + b.h * 0.7,
          w: b.w * 0.4,
          h: b.h * 0.18,
        },
        width,
        height
      );
      detections.push({
        classId: 1,
        className: "text",
        confidence: 0.6,
        bbox: text,
      });
    }

    onProgress?.(1, "Segmentation complete");
    return detections;
  }
}

/** True if a region contains a meaningful amount of dark (ink) pixels. */
function hasContent(gray: Float32Array, w: number, b: BBox, lightThresh: number): boolean {
  const x0 = Math.floor(b.x);
  const y0 = Math.floor(b.y);
  const x1 = Math.floor(b.x + b.w);
  const y1 = Math.floor(b.y + b.h);
  let dark = 0;
  let total = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      total++;
      if (gray[y * w + x] < lightThresh - 40) dark++;
    }
  }
  return total > 0 && dark / total > 0.01;
}

// ---- demo OCR --------------------------------------------------------------

const PLACEHOLDER_LINES = [
  "…",
  "!!",
  "Hmm…",
  "Wait—",
  "Let's go.",
  "No way!",
  "Over here!",
  "……",
];

class DemoOcr implements OcrEngine {
  readonly name = "Demo OCR (placeholder)";

  async recognize(img: DecodedImage, onProgress?: ProgressFn): Promise<OcrLine[]> {
    // The demo OCR doesn't read pixels; the pipeline pairs it with the demo
    // segmenter's synthesized regions. Return nothing here — the pipeline fills
    // placeholder text for demo regions. Kept as a real no-op engine so the
    // interface stays honest.
    void img;
    onProgress?.(1, "OCR complete");
    return [];
  }
}

export function placeholderText(index: number): string {
  return PLACEHOLDER_LINES[index % PLACEHOLDER_LINES.length];
}

// ---- provider --------------------------------------------------------------

export function createDemoProvider(): InferenceProvider {
  return {
    id: "demo",
    label: "Demo (no model weights)",
    available: true,
    segmenter: new DemoSegmenter(),
    ocr: new DemoOcr(),
  };
}
