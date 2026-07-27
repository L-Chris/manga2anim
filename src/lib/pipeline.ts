import type {
  OcrLine,
  PageResult,
  Panel,
  RawDetection,
  ReadingDirection,
  TextRegion,
} from "../types";
import { classifyText } from "./classify";
import { panelColor } from "./colors";
import { containment, unionBBox } from "./geometry";
import type { DecodedImage, InferenceProvider, ProgressFn } from "./providers/types";
import { sortByReadingOrder } from "./readingOrder";

let idCounter = 0;
function uid(prefix: string): string {
  idCounter += 1;
  return `${prefix}_${idCounter.toString(36)}_${Math.floor(idCounter / 1000)}`;
}

export interface AssembleOptions {
  pageId: string;
  name: string;
  sourcePath?: string;
  imageDataUrl?: string;
  readingDirection: ReadingDirection;
}

/**
 * Run the full parse pipeline for one page:
 *   segmentation → OCR → geometric sort → panel/text assignment → classification
 *
 * `provider` supplies the segmenter + OCR engine. Progress is reported through
 * `onProgress` as an overall 0..1 fraction.
 */
export async function parsePage(
  image: DecodedImage,
  provider: InferenceProvider,
  opts: AssembleOptions,
  onProgress?: ProgressFn
): Promise<PageResult> {
  const { width, height } = image;

  // 1. Segmentation ---------------------------------------------------------
  const detections = await provider.segmenter.segment(image, (f, stage) =>
    onProgress?.(f * 0.45, stage)
  );

  const panels = detections.filter((d) => d.className === "panel");
  const bubbles = detections.filter((d) => d.className === "bubble");
  const texts = detections.filter((d) => d.className === "text");

  // 2. Geometric reading-order sort of panels (core logic) ------------------
  onProgress?.(0.5, "Reconstructing reading order");
  const orderedPanels = sortByReadingOrder(panels, opts.readingDirection);

  // 3. OCR over the whole page (one pass) -----------------------------------
  onProgress?.(0.55, "Running OCR");
  const ocrLines = await provider.ocr.recognize(image, (f, stage) =>
    onProgress?.(0.55 + f * 0.25, stage)
  );

  // 4. Assemble text regions from bubbles + text detections + OCR lines -----
  onProgress?.(0.85, "Assigning text to panels");
  const textRegions = assembleTextRegions(
    bubbles,
    texts,
    ocrLines,
    width,
    height
  );

  // 5. Build panel objects and assign text regions by containment -----------
  const panelObjects: Panel[] = orderedPanels.map((det, i) => ({
    id: uid("panel"),
    order: i + 1,
    confidence: det.confidence,
    bbox: det.bbox,
    polygon: det.polygon,
    color: panelColor(i),
    textIds: [],
  }));

  for (const region of textRegions) {
    const owner = findOwningPanel(region.bbox, panelObjects);
    region.panelId = owner ? owner.id : null;
  }

  // Order text within each panel by reading direction, then link ids.
  for (const panel of panelObjects) {
    const owned = textRegions.filter((r) => r.panelId === panel.id);
    const ordered = sortByReadingOrder(owned, opts.readingDirection);
    panel.textIds = ordered.map((r) => r.id);
  }

  onProgress?.(1, "Done");

  return {
    pageId: opts.pageId,
    name: opts.name,
    imageDataUrl: opts.imageDataUrl,
    sourcePath: opts.sourcePath,
    imageWidth: width,
    imageHeight: height,
    readingDirection: opts.readingDirection,
    panels: panelObjects,
    textRegions,
    status: "done",
  };
}

/**
 * Merge bubble detections, text detections, and OCR lines into TextRegions.
 *
 * Strategy: each bubble/text detection becomes a region. The region's text is
 * the concatenation of OCR lines whose centers fall inside it; regions without
 * recognized text remain empty for manual correction.
 */
function assembleTextRegions(
  bubbles: RawDetection[],
  texts: RawDetection[],
  ocrLines: OcrLine[],
  pageWidth: number,
  pageHeight: number
): TextRegion[] {
  const regions: TextRegion[] = [];

  const makeRegion = (det: RawDetection, fromBubble: boolean): TextRegion => {
    const contained = ocrLines.filter((l) =>
      pointInBBox(l.bbox.x + l.bbox.w / 2, l.bbox.y + l.bbox.h / 2, det.bbox)
    );
    const text = contained.map((l) => l.text).join(" ");
    const confidence = contained.length > 0
      ? contained.reduce((s, l) => s + l.confidence, 0) / contained.length
      : det.confidence;

    const kind = classifyText({
      text,
      bbox: det.bbox,
      fromBubble,
      pageWidth,
      pageHeight,
    });

    return {
      id: uid("text"),
      kind,
      text,
      confidence,
      bbox: det.bbox,
      polygon: det.polygon,
      fromBubble,
      panelId: null,
    };
  };

  for (const b of bubbles) regions.push(makeRegion(b, true));
  for (const t of texts) regions.push(makeRegion(t, false));

  // Any OCR lines not captured by a detection become standalone "other text".
  if (ocrLines.length > 0) {
    for (const line of ocrLines) {
      const captured = regions.some((r) =>
        pointInBBox(line.bbox.x + line.bbox.w / 2, line.bbox.y + line.bbox.h / 2, r.bbox)
      );
      if (captured) continue;
      const kind = classifyText({
        text: line.text,
        bbox: line.bbox,
        fromBubble: false,
        pageWidth,
        pageHeight,
      });
      regions.push({
        id: uid("text"),
        kind,
        text: line.text,
        confidence: line.confidence,
        bbox: line.bbox,
        polygon: line.polygon,
        fromBubble: false,
        panelId: null,
      });
    }
  }

  return regions;
}

/** Find the panel that best contains a text region (≥50% containment, largest overlap). */
function findOwningPanel(bbox: TextRegion["bbox"], panels: Panel[]): Panel | null {
  let best: Panel | null = null;
  let bestScore = 0;
  for (const p of panels) {
    const c = containment(bbox, p.bbox);
    if (c >= 0.5 && c > bestScore) {
      bestScore = c;
      best = p;
    }
  }
  return best;
}

function pointInBBox(px: number, py: number, b: { x: number; y: number; w: number; h: number }): boolean {
  return px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h;
}

// Re-export for callers that build panels from manual edits.
export { unionBBox };
