import { describe, it, expect } from "vitest";
import {
  ctcDecode,
  detPostprocess,
  detPreprocess,
  parseDictionary,
  recPreprocess,
} from "./ppocr";

describe("parseDictionary", () => {
  it("splits lines and strips \\r", () => {
    const dict = parseDictionary("a\r\nb\r\nc\n");
    expect(dict).toEqual(["a", "b", "c"]);
  });

  it("handles empty input", () => {
    expect(parseDictionary("")).toEqual([]);
  });

  it("drops the trailing empty element of a newline-terminated file", () => {
    // Real ppocr_keys_v1.txt: 6623 lines each ending in \n → naive split = 6624.
    expect(parseDictionary("a\nb\nc\n")).toEqual(["a", "b", "c"]);
  });
});

describe("detPreprocess", () => {
  it("produces a CHW tensor with dims that are multiples of 32", () => {
    const w = 100;
    const h = 200;
    const data = new Uint8ClampedArray(w * h * 4).fill(128);
    const { tensor, newW, newH } = detPreprocess(data, w, h, 960);
    expect(newW % 32).toBe(0);
    expect(newH % 32).toBe(0);
    expect(tensor.length).toBe(3 * newW * newH);
  });

  it("scales down large images to maxSide", () => {
    const w = 4000;
    const h = 3000;
    const data = new Uint8ClampedArray(w * h * 4);
    const { newW, newH } = detPreprocess(data, w, h, 960);
    expect(Math.max(newW, newH)).toBeLessThanOrEqual(960 + 32);
  });

  it("normalizes pixel values with ImageNet mean/std", () => {
    // Pure white pixel (255,255,255) → (1-0.485)/0.229 ≈ 2.249
    const w = 32;
    const h = 32;
    const data = new Uint8ClampedArray(w * h * 4).fill(255);
    const { tensor } = detPreprocess(data, w, h, 960);
    expect(tensor[0]).toBeCloseTo((1 - 0.485) / 0.229, 2);
  });
});

describe("detPostprocess", () => {
  it("finds a single rectangular text region in the probability map", () => {
    const mapH = 100;
    const mapW = 100;
    const probMap = new Float32Array(mapH * mapW);
    // Draw a rectangle of high probability at (20,20)-(40,40).
    for (let y = 20; y <= 40; y++) {
      for (let x = 20; x <= 40; x++) {
        probMap[y * mapW + x] = 0.9;
      }
    }
    const boxes = detPostprocess(probMap, mapH, mapW, 1, 1, 100, 100);
    expect(boxes.length).toBe(1);
    // The box should roughly cover the rectangle (with expansion).
    const b = boxes[0];
    expect(b.x).toBeLessThan(20);
    expect(b.y).toBeLessThan(20);
    expect(b.x + b.w).toBeGreaterThan(40);
    expect(b.y + b.h).toBeGreaterThan(40);
  });

  it("finds two separate text regions", () => {
    const mapH = 100;
    const mapW = 200;
    const probMap = new Float32Array(mapH * mapW);
    // Region 1: (10,10)-(20,30)
    for (let y = 10; y <= 20; y++)
      for (let x = 10; x <= 30; x++) probMap[y * mapW + x] = 0.8;
    // Region 2: (60,60)-(70,80)
    for (let y = 60; y <= 70; y++)
      for (let x = 60; x <= 80; x++) probMap[y * mapW + x] = 0.8;

    const boxes = detPostprocess(probMap, mapH, mapW, 1, 1, 200, 100);
    expect(boxes.length).toBe(2);
  });

  it("filters out tiny noise regions", () => {
    const mapH = 200;
    const mapW = 200;
    const probMap = new Float32Array(mapH * mapW);
    // A 2x2 blob — too small.
    probMap[100 * mapW + 100] = 0.9;
    probMap[100 * mapW + 101] = 0.9;
    probMap[101 * mapW + 100] = 0.9;
    probMap[101 * mapW + 101] = 0.9;
    const boxes = detPostprocess(probMap, mapH, mapW, 1, 1, 2000, 2000);
    expect(boxes.length).toBe(0);
  });

  it("maps coordinates back to original image space via scale", () => {
    const mapH = 50;
    const mapW = 50;
    const probMap = new Float32Array(mapH * mapW);
    // Rectangle spanning map pixels (10..20) inclusive → continuous [10,21).
    for (let y = 10; y <= 20; y++)
      for (let x = 10; x <= 20; x++) probMap[y * mapW + x] = 0.9;
    // scale = 0.5 means the original image is 2x the map; the un-expanded
    // mapped rectangle is [10,21)/0.5 = [20,42). Expansion only grows outward,
    // so the returned box must contain [20,42) on both axes.
    const boxes = detPostprocess(probMap, mapH, mapW, 0.5, 0.5, 100, 100);
    expect(boxes.length).toBe(1);
    const b = boxes[0];
    expect(b.x).toBeLessThanOrEqual(20);
    expect(b.y).toBeLessThanOrEqual(20);
    expect(b.x + b.w).toBeGreaterThanOrEqual(42);
    expect(b.y + b.h).toBeGreaterThanOrEqual(42);
  });
});

describe("recPreprocess", () => {
  it("returns null for degenerate crops", () => {
    const data = new Uint8ClampedArray(100 * 100 * 4);
    expect(recPreprocess(data, 100, 100, { x: 50, y: 50, w: 0, h: 0 })).toBeNull();
    expect(recPreprocess(data, 100, 100, { x: 50, y: 50, w: 1, h: 1 })).toBeNull();
  });

  it("produces a [3, 48, 320] tensor", () => {
    const w = 200;
    const h = 200;
    const data = new Uint8ClampedArray(w * h * 4).fill(100);
    const result = recPreprocess(data, w, h, { x: 10, y: 10, w: 80, h: 40 });
    expect(result).not.toBeNull();
    expect(result!.tensor.length).toBe(3 * 48 * 320);
    expect(result!.width).toBeGreaterThan(0);
    expect(result!.width).toBeLessThanOrEqual(320);
  });
});

describe("ctcDecode", () => {
  const dict = ["a", "b", "c", "d", "e"]; // class 1→'a', 2→'b', ...

  it("decodes a simple sequence with blank removal", () => {
    // T=5, numClasses=6 (blank=0, a=1..e=5)
    // Sequence: blank, a, a, blank, b → decoded "ab"
    const T = 5;
    const C = 6;
    const logits = new Float32Array(T * C).fill(-10);
    logits[0 * C + 0] = 10; // blank
    logits[1 * C + 1] = 10; // a
    logits[2 * C + 1] = 10; // a (repeat → collapsed)
    logits[3 * C + 0] = 10; // blank
    logits[4 * C + 2] = 10; // b
    const { text } = ctcDecode(logits, T, C, dict);
    expect(text).toBe("ab");
  });

  it("handles all-blank sequence", () => {
    const T = 3;
    const C = 6;
    const logits = new Float32Array(T * C).fill(-10);
    for (let t = 0; t < T; t++) logits[t * C + 0] = 10;
    const { text } = ctcDecode(logits, T, C, dict);
    expect(text).toBe("");
  });

  it("decodes repeated chars separated by blank", () => {
    // a, blank, a → "aa" (blank separates repeats)
    const T = 3;
    const C = 6;
    const logits = new Float32Array(T * C).fill(-10);
    logits[0 * C + 1] = 10; // a
    logits[1 * C + 0] = 10; // blank
    logits[2 * C + 1] = 10; // a
    const { text } = ctcDecode(logits, T, C, dict);
    expect(text).toBe("aa");
  });

  it("returns average confidence of decoded chars", () => {
    const T = 2;
    const C = 6;
    const logits = new Float32Array(T * C).fill(-10);
    logits[0 * C + 1] = 0.8; // a with logit 0.8
    logits[1 * C + 2] = 0.6; // b with logit 0.6
    const { confidence } = ctcDecode(logits, T, C, dict);
    expect(confidence).toBeCloseTo((0.8 + 0.6) / 2, 5);
  });
});
