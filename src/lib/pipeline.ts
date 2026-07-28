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
import { bboxArea, containment, unionBBox } from "./geometry";
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

  // 3. OCR constrained to YOLO text/bubble regions --------------------------
  onProgress?.(0.55, "Running OCR");
  const ocrRegions = [...bubbles, ...texts].map((detection) => detection.bbox);
  const ocrLines = await provider.ocr.recognize(
    image,
    ocrRegions,
    (f, stage) => onProgress?.(0.55 + f * 0.25, stage)
  );

  // 4. Assemble text regions from bubbles + text detections + OCR lines -----
  onProgress?.(0.85, "Assigning text to panels");
  const textRegions = assembleTextRegions(
    bubbles,
    texts,
    ocrLines,
    width,
    height,
    opts.readingDirection
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
 * Strategy: every OCR line is assigned to exactly one detection, with bubbles
 * taking priority over raw text boxes. This prevents the same recognized line
 * from becoming both a dialogue region and an "unknown" text region when YOLO
 * emits overlapping bubble + text detections. Detections without recognized
 * text remain empty for manual correction unless another detection already
 * owns the OCR content they cover.
 */
function assembleTextRegions(
  bubbles: RawDetection[],
  texts: RawDetection[],
  ocrLines: OcrLine[],
  pageWidth: number,
  pageHeight: number,
  readingDirection: ReadingDirection
): TextRegion[] {
  const regions: TextRegion[] = [];

  type RegionBucket = {
    det: RawDetection;
    fromBubble: boolean;
    lines: OcrLine[];
  };

  const bubbleBuckets: RegionBucket[] = bubbles.map((det) => ({
    det,
    fromBubble: true,
    lines: [],
  }));
  const textBuckets: RegionBucket[] = texts.map((det) => ({
    det,
    fromBubble: false,
    lines: [],
  }));

  const lineMatchesDetection = (line: OcrLine, det: RawDetection): boolean => {
    const centerX = line.bbox.x + line.bbox.w / 2;
    const centerY = line.bbox.y + line.bbox.h / 2;
    return (
      pointInBBox(centerX, centerY, det.bbox) ||
      containment(line.bbox, det.bbox) >= 0.5
    );
  };

  const findBestBucket = (
    line: OcrLine,
    buckets: RegionBucket[]
  ): RegionBucket | null => {
    let best: RegionBucket | null = null;
    let bestContainment = -1;
    let bestArea = Infinity;
    let bestConfidence = -1;
    for (const bucket of buckets) {
      if (!lineMatchesDetection(line, bucket.det)) continue;
      const covered = containment(line.bbox, bucket.det.bbox);
      const area = bboxArea(bucket.det.bbox);
      if (
        covered > bestContainment ||
        (covered === bestContainment && area < bestArea) ||
        (covered === bestContainment &&
          area === bestArea &&
          bucket.det.confidence > bestConfidence)
      ) {
        best = bucket;
        bestContainment = covered;
        bestArea = area;
        bestConfidence = bucket.det.confidence;
      }
    }
    return best;
  };

  const assignedLines = new Set<OcrLine>();
  for (const line of ocrLines) {
    // A bubble is the semantic container for dialogue. Only fall back to a raw
    // text detection when no bubble covers the OCR line.
    const owner =
      findBestBucket(line, bubbleBuckets) ??
      findBestBucket(line, textBuckets);
    if (!owner) continue;
    owner.lines.push(line);
    assignedLines.add(line);
  }

  const makeRegion = (bucket: RegionBucket): TextRegion => {
    const contained = sortByReadingOrder(
      bucket.lines,
      readingDirection
    );
    const text = contained.map((l) => l.text).join(" ");
    const confidence = contained.length > 0
      ? contained.reduce((s, l) => s + l.confidence, 0) / contained.length
      : bucket.det.confidence;

    const kind = classifyText({
      text,
      bbox: bucket.det.bbox,
      fromBubble: bucket.fromBubble,
      pageWidth,
      pageHeight,
    });

    return {
      id: uid("text"),
      kind,
      text,
      confidence,
      bbox: bucket.det.bbox,
      polygon: bucket.det.polygon,
      fromBubble: bucket.fromBubble,
      panelId: null,
    };
  };

  const hasCoveredLine = (bucket: RegionBucket): boolean =>
    ocrLines.some((line) => lineMatchesDetection(line, bucket.det));

  for (const bucket of bubbleBuckets) {
    // Suppress duplicate bubble detections whose OCR was assigned to a better
    // (usually tighter) bubble, while preserving genuinely empty bubbles.
    if (bucket.lines.length === 0 && hasCoveredLine(bucket)) continue;
    regions.push(makeRegion(bucket));
  }
  for (const bucket of textBuckets) {
    const shadowedByBubble = bubbles.some(
      (bubble) => containment(bucket.det.bbox, bubble.bbox) >= 0.5
    );
    // Text boxes inside bubbles are structural duplicates. Likewise, if a
    // different text box already owns every OCR line covered by this box, an
    // empty duplicate card provides no useful manual-correction target.
    if (
      bucket.lines.length === 0 &&
      (shadowedByBubble || hasCoveredLine(bucket))
    ) {
      continue;
    }
    regions.push(makeRegion(bucket));
  }

  // Any OCR lines not captured by a detection become standalone "other text".
  if (ocrLines.length > 0) {
    for (const line of ocrLines) {
      if (assignedLines.has(line)) continue;
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
