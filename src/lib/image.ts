import type { DecodedImage } from "./providers/types";

/** Load an image element from a URL/data URL. */
export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src.slice(0, 64)}`));
    img.src = src;
  });
}

/** Decode an image source into raw RGBA pixels for the inference providers. */
export async function decodeImage(src: string): Promise<DecodedImage> {
  const img = await loadImage(src);
  const width = img.naturalWidth;
  const height = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas context unavailable");
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, width, height);
  return { width, height, data };
}

/** Read a File/Blob as a data URL. */
export function fileToDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
