import type { BBox, ReadingDirection } from "../types";
import { bboxCenter } from "./geometry";

/**
 * Geometric reading-order reconstruction (the app's core logic).
 *
 * Approach: axis-banding (a greedy 1-D XY-cut).
 *
 *  1. Every manga reading direction progresses **top → bottom** across rows, so
 *     the primary axis is always vertical (Y). RTL/LTR only change how items are
 *     ordered *within* a row; vertical (webtoon) is effectively one item per row.
 *  2. Items are grouped into horizontal "bands" by greedily merging any item
 *     whose vertical extent overlaps the running band. This tolerates panels of
 *     different heights sitting on the same visual row.
 *  3. Bands are emitted top → bottom; within each band items are ordered along
 *     the secondary (X) axis, ascending for LTR/vertical and descending for RTL.
 *
 * The algorithm is pure and deterministic — no model involved — and is reused
 * both for ordering panels on a page and for ordering text lines inside a panel.
 */

export interface Orderable {
  bbox: BBox;
}

/** A band is a set of items sharing a visual row. */
interface Band<T> {
  items: T[];
  top: number;
  bottom: number;
}

/**
 * Fraction of the smaller item's height that must overlap the running band for
 * it to be considered part of that row. Keeps merely-adjacent panels from
 * merging while still grouping genuinely co-row panels.
 */
const ROW_OVERLAP_TOLERANCE = 0.35;

/**
 * Sort items into reading order for the given direction.
 * Returns a new array; the input is not mutated.
 */
export function sortByReadingOrder<T extends Orderable>(
  items: readonly T[],
  direction: ReadingDirection
): T[] {
  if (items.length <= 1) return items.slice();

  const bands = buildBands(items);

  // Bands top → bottom (by their vertical midpoint).
  bands.sort((a, b) => (a.top + a.bottom) / 2 - (b.top + b.bottom) / 2);

  const rtl = direction === "rtl";
  const xSign = rtl ? -1 : 1; // rtl → rightmost first

  const ordered: T[] = [];
  for (const band of bands) {
    band.items.sort((a, b) => {
      const ax = bboxCenter(a.bbox).x;
      const bx = bboxCenter(b.bbox).x;
      const dx = (ax - bx) * xSign;
      if (Math.abs(dx) > 1e-6) return dx;
      // Tie-break on vertical center so ordering is total & stable.
      return bboxCenter(a.bbox).y - bboxCenter(b.bbox).y;
    });
    ordered.push(...band.items);
  }
  return ordered;
}

/**
 * Greedy interval-overlap banding on the vertical axis.
 * Items are processed top → bottom; an item joins the current band if it
 * vertically overlaps the band by at least ROW_OVERLAP_TOLERANCE of its own
 * height, otherwise it starts a new band.
 */
function buildBands<T extends Orderable>(items: readonly T[]): Band<T>[] {
  const sorted = items
    .slice()
    .sort((a, b) => a.bbox.y - b.bbox.y || a.bbox.x - b.bbox.x);

  const bands: Band<T>[] = [];
  let current: Band<T> | null = null;

  for (const item of sorted) {
    const top = item.bbox.y;
    const bottom = item.bbox.y + item.bbox.h;

    if (current && overlapsBand(top, bottom, current, item.bbox.h)) {
      current.items.push(item);
      current.top = Math.min(current.top, top);
      current.bottom = Math.max(current.bottom, bottom);
    } else {
      current = { items: [item], top, bottom };
      bands.push(current);
    }
  }
  return bands;
}

function overlapsBand(
  top: number,
  bottom: number,
  band: { top: number; bottom: number },
  itemHeight: number
): boolean {
  const overlap = Math.min(bottom, band.bottom) - Math.max(top, band.top);
  if (overlap <= 0) return false;
  const ref = Math.max(1, Math.min(itemHeight, band.bottom - band.top));
  return overlap / ref >= ROW_OVERLAP_TOLERANCE;
}

/**
 * Convenience: return the input indices in reading order. Useful when callers
 * need to reorder a parallel array without re-wrapping objects.
 */
export function readingOrderIndices(
  boxes: readonly BBox[],
  direction: ReadingDirection
): number[] {
  const wrapped = boxes.map((bbox, index) => ({ bbox, index }));
  return sortByReadingOrder(wrapped, direction).map((w) => w.index);
}
