/**
 * Core data model for the manga panel parser.
 *
 * Coordinate convention: all geometry is stored in **image pixel coordinates**
 * (origin top-left, x right, y down). The page carries imageWidth/imageHeight so
 * any consumer (canvas overlay, JSON export) can resolve positions regardless of
 * how the image is scaled for display.
 */

/** Reading direction of a manga page. Drives geometric sort. */
export type ReadingDirection = "rtl" | "ltr" | "vertical";

/** A 2D point in image pixel coordinates. */
export interface Point {
  x: number;
  y: number;
}

/** Axis-aligned bounding box in image pixel coordinates. */
export interface BBox {
  x: number; // left
  y: number; // top
  w: number;
  h: number;
}

/**
 * The three instance classes the YOLO26s manga segmentation model is expected
 * to emit. Kept as a fixed union so the pipeline can rely on them.
 */
export type DetectionClass = "panel" | "text" | "bubble";

/**
 * A single raw instance-segmentation detection straight from the model, before
 * any geometric reasoning. The ONNX provider normalizes its output into this
 * shape.
 */
export interface RawDetection {
  classId: number;
  className: DetectionClass;
  confidence: number;
  bbox: BBox;
  /** Optional instance polygon (pixel coords). Falls back to bbox if absent. */
  polygon?: Point[];
}

/** A single OCR text line with its location. */
export interface OcrLine {
  text: string;
  confidence: number;
  bbox: BBox;
  polygon?: Point[];
}

/**
 * Fine-grained classification of a piece of text. `dialogue`/`thought` live in
 * speech bubbles and attach to panels; everything else is surfaced in the
 * "other text" section per the product spec.
 */
export type TextKind =
  | "dialogue"
  | "thought"
  | "narration"
  | "sfx"
  | "interjection"
  | "unknown";

/** A text region after OCR + classification + panel assignment. */
export interface TextRegion {
  id: string;
  kind: TextKind;
  /** Recognized text (user-editable). */
  text: string;
  confidence: number;
  bbox: BBox;
  polygon?: Point[];
  /** Whether this came from a speech-bubble detection (vs. raw text region). */
  fromBubble: boolean;
  /** Id of the panel this region was assigned to, or null if unassigned. */
  panelId: string | null;
  /** True if the user has manually edited this region. */
  manual?: boolean;
}

/** A manga panel (分镜) after geometric reading-order reconstruction. */
export interface Panel {
  id: string;
  /** 1-based reading order within the page. */
  order: number;
  confidence: number;
  bbox: BBox;
  polygon?: Point[];
  /** Hex color used to draw this panel's border + badge. */
  color: string;
  /** Ids of TextRegions assigned to this panel, in reading order. */
  textIds: string[];
  manual?: boolean;
}

/** Per-page parse result. This is what gets retained and exported. */
export interface PageResult {
  pageId: string;
  /** Display name (file basename). */
  name: string;
  /** Source image as a data URL (for rendering) — not serialized to export. */
  imageDataUrl?: string;
  /** Absolute path on disk (Tauri) — serialized to export. */
  sourcePath?: string;
  imageWidth: number;
  imageHeight: number;
  readingDirection: ReadingDirection;
  /** Panels sorted by `order`. */
  panels: Panel[];
  /** All text regions on the page (dialogue + other). */
  textRegions: TextRegion[];
  status: PageStatus;
  /** Human-readable error if status === "error". */
  error?: string;
}

export type PageStatus =
  | "pending"
  | "segmenting"
  | "ocr"
  | "sorting"
  | "done"
  | "error";

/** Top-level export document (what "Export JSON" writes). */
export interface ExportDocument {
  schema: "manga-panel-parser/v1";
  exportedAt: string;
  readingDirection: ReadingDirection;
  pages: ExportPage[];
}

export interface ExportPage {
  pageId: string;
  name: string;
  sourcePath?: string;
  imageWidth: number;
  imageHeight: number;
  readingDirection: ReadingDirection;
  panels: Panel[];
  textRegions: TextRegion[];
  /** Convenience: text regions not assigned to any panel. */
  otherTexts: TextRegion[];
}

/** True if a text kind belongs in the dialogue flow of a panel. */
export function isDialogueKind(kind: TextKind): boolean {
  return kind === "dialogue" || kind === "thought";
}

/** True if a text region is not assigned to a panel. */
export function isOtherText(region: TextRegion): boolean {
  return region.panelId === null;
}
