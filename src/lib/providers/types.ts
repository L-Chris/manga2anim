import type { OcrLine, RawDetection } from "../../types";

/**
 * Decoded image handed to providers. Pixel data is RGBA (Uint8ClampedArray),
 * row-major, length = width * height * 4.
 */
export interface DecodedImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** Progress callback (0..1) with a human-readable stage label. */
export type ProgressFn = (fraction: number, stage: string) => void;

/**
 * A segmentation model (YOLO26s manga instance segmentation). Emits panel,
 * text, and bubble instances.
 */
export interface Segmenter {
  readonly name: string;
  segment(image: DecodedImage, onProgress?: ProgressFn): Promise<RawDetection[]>;
}

/** An OCR engine (PP-OCRv6 small). Recognizes text lines in an image. */
export interface OcrEngine {
  readonly name: string;
  /** Recognize all text lines in the image. */
  recognize(image: DecodedImage, onProgress?: ProgressFn): Promise<OcrLine[]>;
}

/**
 * A provider bundles a segmenter + OCR engine and reports whether its model
 * weights are actually available. The pipeline picks the ONNX provider when
 * weights are present and otherwise falls back to the demo provider so the full
 * flow remains runnable end-to-end.
 */
export interface InferenceProvider {
  readonly id: string;
  readonly label: string;
  readonly available: boolean;
  readonly segmenter: Segmenter;
  readonly ocr: OcrEngine;
}
