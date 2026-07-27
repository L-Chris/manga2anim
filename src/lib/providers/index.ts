import { createOnnxProvider } from "./onnxProvider";
import type { InferenceProvider } from "./types";

export type { InferenceProvider, DecodedImage, ProgressFn } from "./types";

/** Load the required YOLO26s + PP-OCRv6 provider. */
export async function resolveProvider(
  modelsDirUrl: string
): Promise<InferenceProvider> {
  return createOnnxProvider({ modelsDirUrl });
}
