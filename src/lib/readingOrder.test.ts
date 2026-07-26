import { describe, it, expect } from "vitest";
import { sortByReadingOrder, readingOrderIndices } from "./readingOrder";
import type { BBox } from "../types";

function box(x: number, y: number, w: number, h: number): BBox {
  return { x, y, w, h };
}

interface Item {
  id: string;
  bbox: BBox;
}

function ids(items: Item[]): string[] {
  return items.map((i) => i.id);
}

describe("sortByReadingOrder", () => {
  it("orders a simple 2x2 grid RTL (right-to-left, top-to-bottom)", () => {
    // Classic manga: top-right first, then top-left, then bottom-right, bottom-left.
    const items: Item[] = [
      { id: "TL", bbox: box(0, 0, 100, 100) },
      { id: "TR", bbox: box(110, 0, 100, 100) },
      { id: "BL", bbox: box(0, 110, 100, 100) },
      { id: "BR", bbox: box(110, 110, 100, 100) },
    ];
    expect(ids(sortByReadingOrder(items, "rtl"))).toEqual([
      "TR",
      "TL",
      "BR",
      "BL",
    ]);
  });

  it("orders a simple 2x2 grid LTR", () => {
    const items: Item[] = [
      { id: "TL", bbox: box(0, 0, 100, 100) },
      { id: "TR", bbox: box(110, 0, 100, 100) },
      { id: "BL", bbox: box(0, 110, 100, 100) },
      { id: "BR", bbox: box(110, 110, 100, 100) },
    ];
    expect(ids(sortByReadingOrder(items, "ltr"))).toEqual([
      "TL",
      "TR",
      "BL",
      "BR",
    ]);
  });

  it("groups panels of different heights into the same visual row", () => {
    // A tall panel on the left spanning two rows, two stacked panels on the right.
    // Because the tall panel's vertical extent overlaps both right panels, all
    // three land in a single band; RTL then orders them right→left, tie-breaking
    // top→bottom: right column first (top then bottom), then the tall left panel.
    const tall: Item = { id: "TALL", bbox: box(0, 0, 100, 210) };
    const rTop: Item = { id: "RT", bbox: box(110, 0, 100, 100) };
    const rBot: Item = { id: "RB", bbox: box(110, 110, 100, 100) };
    const out = ids(sortByReadingOrder([tall, rTop, rBot], "rtl"));
    expect(out).toEqual(["RT", "RB", "TALL"]);
  });

  it("keeps genuinely separate rows apart (no full-height panel)", () => {
    // Two panels top row, one panel bottom row — must not merge across the gap.
    const items: Item[] = [
      { id: "TR", bbox: box(110, 0, 100, 90) },
      { id: "TL", bbox: box(0, 0, 100, 90) },
      { id: "B", bbox: box(55, 200, 100, 90) },
    ];
    expect(ids(sortByReadingOrder(items, "rtl"))).toEqual(["TR", "TL", "B"]);
  });

  it("vertical direction reads strictly top-to-bottom", () => {
    const items: Item[] = [
      { id: "B", bbox: box(50, 300, 100, 80) },
      { id: "A", bbox: box(50, 0, 100, 80) },
      { id: "C", bbox: box(50, 600, 100, 80) },
    ];
    expect(ids(sortByReadingOrder(items, "vertical"))).toEqual([
      "A",
      "B",
      "C",
    ]);
  });

  it("is stable and total for a single item and empty input", () => {
    const one: Item[] = [{ id: "only", bbox: box(0, 0, 10, 10) }];
    expect(ids(sortByReadingOrder(one, "rtl"))).toEqual(["only"]);
    expect(sortByReadingOrder([], "rtl")).toEqual([]);
  });

  it("does not mutate the input array", () => {
    const items: Item[] = [
      { id: "B", bbox: box(0, 100, 50, 50) },
      { id: "A", bbox: box(0, 0, 50, 50) },
    ];
    const snapshot = items.map((i) => i.id);
    sortByReadingOrder(items, "ltr");
    expect(items.map((i) => i.id)).toEqual(snapshot);
  });

  it("readingOrderIndices returns indices in reading order (LTR)", () => {
    const boxes: BBox[] = [
      box(110, 0, 100, 100), // index 0 = top-right
      box(0, 0, 100, 100), // index 1 = top-left
    ];
    expect(readingOrderIndices(boxes, "ltr")).toEqual([1, 0]);
    expect(readingOrderIndices(boxes, "rtl")).toEqual([0, 1]);
  });
});
