import type { BBox, OcrLine, RawDetection } from "../../types";

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
  /** Recognize text lines inside YOLO text/bubble regions. */
  recognize(
    image: DecodedImage,
    regions: readonly BBox[],
    onProgress?: ProgressFn
  ): Promise<OcrLine[]>;
}

/** A provider bundles the required segmentation and OCR engines. */
export interface InferenceProvider {
  readonly id: string;
  readonly label: string;
  readonly segmenter: Segmenter;
  readonly ocr: OcrEngine;
}
