import { describe, expect, it } from "vitest";
import type { OcrLine, RawDetection } from "../types";
import { parsePage } from "./pipeline";
import type { DecodedImage, InferenceProvider } from "./providers/types";

const image: DecodedImage = {
  width: 400,
  height: 200,
  data: new Uint8ClampedArray(400 * 200 * 4),
};

function createTestProvider(): InferenceProvider {
  const detections: RawDetection[] = [
    {
      classId: 0,
      className: "panel",
      confidence: 0.9,
      bbox: { x: 10, y: 10, w: 180, h: 180 },
    },
    {
      classId: 0,
      className: "panel",
      confidence: 0.9,
      bbox: { x: 210, y: 10, w: 180, h: 180 },
    },
    {
      classId: 2,
      className: "bubble",
      confidence: 0.8,
      bbox: { x: 230, y: 30, w: 100, h: 60 },
    },
    {
      classId: 1,
      className: "text",
      confidence: 0.8,
      bbox: { x: 30, y: 120, w: 100, h: 30 },
    },
    {
      classId: 1,
      className: "text",
      confidence: 0.7,
      bbox: { x: 250, y: 140, w: 80, h: 30 },
    },
  ];
  const lines: OcrLine[] = [
    {
      text: "right dialogue",
      confidence: 0.95,
      bbox: { x: 250, y: 45, w: 50, h: 20 },
    },
    {
      text: "left caption",
      confidence: 0.9,
      bbox: { x: 50, y: 125, w: 50, h: 15 },
    },
    {
      text: "standalone",
      confidence: 0.85,
      bbox: { x: 192, y: 170, w: 12, h: 12 },
    },
  ];

  return {
    id: "test",
    label: "Test provider",
    segmenter: {
      name: "Test segmenter",
      async segment() {
        return detections;
      },
    },
    ocr: {
      name: "Test OCR",
      async recognize() {
        return lines;
      },
    },
  };
}

describe("pipeline integration", () => {
  it("assembles model detections and OCR without synthesizing text", async () => {
    const result = await parsePage(image, createTestProvider(), {
      pageId: "page",
      name: "page.png",
      readingDirection: "rtl",
    });

    expect(result.status).toBe("done");
    expect(result.panels).toHaveLength(2);
    expect(result.panels.map((panel) => panel.order)).toEqual([1, 2]);
    expect(result.textRegions.map((region) => region.text)).toEqual(
      expect.arrayContaining(["right dialogue", "left caption", "standalone", ""])
    );
    expect(result.textRegions.find((region) => region.text === "")?.confidence).toBe(
      0.7
    );
    expect(
      result.textRegions.find((region) => region.text === "standalone")?.panelId
    ).toBeNull();
  });

  it("orders same-row panels according to the configured reading direction", async () => {
    const rtl = await parsePage(image, createTestProvider(), {
      pageId: "rtl",
      name: "page.png",
      readingDirection: "rtl",
    });
    const ltr = await parsePage(image, createTestProvider(), {
      pageId: "ltr",
      name: "page.png",
      readingDirection: "ltr",
    });

    expect(rtl.panels[0].bbox.x).toBe(210);
    expect(ltr.panels[0].bbox.x).toBe(10);
  });
});
