import type { BBox, TextKind } from "../types";
import { bboxArea } from "./geometry";

/**
 * Heuristic classification of a recognized text line into a TextKind.
 *
 * The product spec asks that onomatopoeia (拟声词), interjections (语气词),
 * narration (旁白) and anything that can't be attributed to a speech bubble be
 * grouped into "other text". This function makes a first-pass guess; the user
 * can always reclassify in the UI.
 *
 * Signals used:
 *  - fromBubble: text inside a detected speech bubble → dialogue/thought.
 *  - length & punctuation: very short, punctuation-heavy → interjection/sfx.
 *  - area-per-character: large inked area for little text → sound effect (SFX).
 *  - position: wide bars hugging the top/bottom edge → narration box.
 */
export function classifyText(params: {
  text: string;
  bbox: BBox;
  fromBubble: boolean;
  pageWidth: number;
  pageHeight: number;
}): TextKind {
  const { text, bbox, fromBubble, pageWidth, pageHeight } = params;
  const trimmed = text.trim();

  if (fromBubble) {
    // Thought bubbles are usually drawn with a cloud/scalloped edge — we can't
    // see that here, so default to dialogue and let the user switch to thought.
    return "dialogue";
  }

  if (trimmed.length === 0) return "unknown";

  // Punctuation-only or very short bursts → interjection (语气词), e.g. "!", "!!", "…".
  const letters = trimmed.replace(/[^\p{L}\p{N}]/gu, "");
  if (letters.length <= 1 && trimmed.length <= 3) return "interjection";

  // Sound effects: few characters but a large bounding box (big inked lettering).
  const area = bboxArea(bbox);
  const pageArea = Math.max(1, pageWidth * pageHeight);
  const areaPerChar = area / Math.max(1, letters.length);
  if (letters.length <= 6 && areaPerChar > pageArea * 0.0009) return "sfx";

  // Narration: a wide, thin box hugging the top or bottom edge of the page.
  const widthRatio = bbox.w / Math.max(1, pageWidth);
  const aspect = bbox.w / Math.max(1, bbox.h);
  const nearTop = bbox.y < pageHeight * 0.12;
  const nearBottom = bbox.y + bbox.h > pageHeight * 0.88;
  if (widthRatio > 0.5 && aspect > 3 && (nearTop || nearBottom)) return "narration";

  return "unknown";
}
