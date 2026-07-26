import { createDemoProvider } from "./demoProvider";
import { createOnnxProvider } from "./onnxProvider";
import type { InferenceProvider } from "./types";

export type { InferenceProvider, DecodedImage, ProgressFn } from "./types";

/**
 * Resolve the best available inference provider. Prefers the ONNX (YOLO26s +
 * PP-OCRv6) provider when its weights are reachable, otherwise falls back to the
 * demo provider so the full pipeline always runs.
 *
 * @param modelsDirUrl base URL serving the ONNX weights (e.g. a Tauri
 *   asset-protocol URL or a dev-server path). Optional — omit to force demo.
 */
export async function resolveProvider(
  modelsDirUrl?: string
): Promise<InferenceProvider> {
  if (modelsDirUrl) {
    const onnx = await createOnnxProvider({ modelsDirUrl });
    if (onnx) return onnx;
  }
  return createDemoProvider();
}
