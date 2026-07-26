import type { BBox, Point } from "../types";

/** Center point of a bounding box. */
export function bboxCenter(b: BBox): Point {
  return { x: b.x + b.w / 2, y: b.y + b.h / 2 };
}

/** Area of a bounding box. */
export function bboxArea(b: BBox): number {
  return Math.max(0, b.w) * Math.max(0, b.h);
}

/** Intersection area of two boxes. */
export function intersectionArea(a: BBox, b: BBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

/** Intersection-over-union of two boxes. */
export function iou(a: BBox, b: BBox): number {
  const inter = intersectionArea(a, b);
  const union = bboxArea(a) + bboxArea(b) - inter;
  return union <= 0 ? 0 : inter / union;
}

/**
 * Fraction of `inner` covered by `outer` (0..1). Used to decide whether a text
 * region "belongs to" a panel or bubble.
 */
export function containment(inner: BBox, outer: BBox): number {
  const inter = intersectionArea(inner, outer);
  const area = bboxArea(inner);
  return area <= 0 ? 0 : inter / area;
}

/** Union bounding box of a list of boxes. */
export function unionBBox(boxes: BBox[]): BBox {
  if (boxes.length === 0) return { x: 0, y: 0, w: 0, h: 0 };
  let x1 = Infinity;
  let y1 = Infinity;
  let x2 = -Infinity;
  let y2 = -Infinity;
  for (const b of boxes) {
    x1 = Math.min(x1, b.x);
    y1 = Math.min(y1, b.y);
    x2 = Math.max(x2, b.x + b.w);
    y2 = Math.max(y2, b.y + b.h);
  }
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/** Clamp a box to lie within [0, maxW] x [0, maxH]. */
export function clampBBox(b: BBox, maxW: number, maxH: number): BBox {
  const x = Math.max(0, Math.min(b.x, maxW));
  const y = Math.max(0, Math.min(b.y, maxH));
  const w = Math.max(0, Math.min(b.x + b.w, maxW) - x);
  const h = Math.max(0, Math.min(b.y + b.h, maxH) - y);
  return { x, y, w, h };
}

/**
 * Standard non-maximum suppression on bounding boxes.
 * Returns the indices of the detections to keep, sorted by descending score.
 */
export function nms(
  boxes: BBox[],
  scores: number[],
  iouThreshold = 0.5
): number[] {
  const order = boxes
    .map((_, i) => i)
    .sort((a, b) => scores[b] - scores[a]);
  const keep: number[] = [];
  const suppressed = new Set<number>();
  for (const i of order) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) suppressed.add(j);
    }
  }
  return keep;
}
