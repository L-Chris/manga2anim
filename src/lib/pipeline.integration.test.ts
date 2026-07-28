import { describe, expect, it } from "vitest";
import { isOtherText } from "../types";
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
      async recognize(_image, regions) {
        expect(regions).toEqual(
          detections
            .filter((detection) =>
              detection.className === "bubble" || detection.className === "text"
            )
            .map((detection) => detection.bbox)
        );
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

  it("keeps assigned unknown text in its panel instead of duplicating it as other text", async () => {
    const result = await parsePage(image, createTestProvider(), {
      pageId: "exclusive-groups",
      name: "page.png",
      readingDirection: "rtl",
    });
    const assignedUnknown = result.textRegions.find(
      (region) => region.text === "left caption"
    );
    const standalone = result.textRegions.find(
      (region) => region.text === "standalone"
    );

    expect(assignedUnknown?.kind).toBe("unknown");
    expect(assignedUnknown?.panelId).not.toBeNull();
    expect(isOtherText(assignedUnknown!)).toBe(false);
    expect(standalone?.panelId).toBeNull();
    expect(isOtherText(standalone!)).toBe(true);
  });

  it("assigns overlapping bubble and text OCR once with bubble priority", async () => {
    const detections: RawDetection[] = [
      {
        classId: 0,
        className: "panel",
        confidence: 0.95,
        bbox: { x: 0, y: 0, w: 200, h: 200 },
      },
      {
        classId: 2,
        className: "bubble",
        confidence: 0.9,
        bbox: { x: 20, y: 20, w: 140, h: 140 },
      },
      {
        classId: 1,
        className: "text",
        confidence: 0.92,
        bbox: { x: 90, y: 40, w: 30, h: 80 },
      },
      {
        classId: 1,
        className: "text",
        confidence: 0.91,
        bbox: { x: 50, y: 40, w: 30, h: 80 },
      },
    ];
    const lines: OcrLine[] = [
      {
        text: "second column",
        confidence: 0.97,
        bbox: { x: 55, y: 45, w: 20, h: 70 },
      },
      {
        text: "first column",
        confidence: 0.98,
        bbox: { x: 95, y: 45, w: 20, h: 70 },
      },
    ];
    const provider: InferenceProvider = {
      id: "overlap",
      label: "Overlap provider",
      segmenter: {
        name: "Overlap segmenter",
        async segment() {
          return detections;
        },
      },
      ocr: {
        name: "Overlap OCR",
        async recognize() {
          return lines;
        },
      },
    };

    const result = await parsePage(image, provider, {
      pageId: "overlap",
      name: "page.png",
      readingDirection: "rtl",
    });

    expect(result.textRegions).toHaveLength(1);
    expect(result.textRegions[0]).toMatchObject({
      text: "first column second column",
      kind: "dialogue",
      fromBubble: true,
    });
    expect(result.panels[0].textIds).toEqual([result.textRegions[0].id]);
    expect(result.textRegions.filter(isOtherText)).toEqual([]);
  });

  it("deduplicates overlapping raw text detections without a bubble", async () => {
    const detections: RawDetection[] = [
      {
        classId: 0,
        className: "panel",
        confidence: 0.95,
        bbox: { x: 0, y: 0, w: 200, h: 200 },
      },
      {
        classId: 1,
        className: "text",
        confidence: 0.85,
        bbox: { x: 20, y: 20, w: 140, h: 140 },
      },
      {
        classId: 1,
        className: "text",
        confidence: 0.92,
        bbox: { x: 60, y: 40, w: 40, h: 80 },
      },
    ];
    const line: OcrLine = {
      text: "recognized once",
      confidence: 0.99,
      bbox: { x: 65, y: 45, w: 30, h: 70 },
    };
    const provider: InferenceProvider = {
      id: "text-overlap",
      label: "Text overlap provider",
      segmenter: {
        name: "Text overlap segmenter",
        async segment() {
          return detections;
        },
      },
      ocr: {
        name: "Text overlap OCR",
        async recognize() {
          return [line];
        },
      },
    };

    const result = await parsePage(image, provider, {
      pageId: "text-overlap",
      name: "page.png",
      readingDirection: "rtl",
    });

    expect(result.textRegions).toHaveLength(1);
    expect(result.textRegions[0]).toMatchObject({
      text: "recognized once",
      fromBubble: false,
      panelId: result.panels[0].id,
    });
    expect(result.panels[0].textIds).toEqual([result.textRegions[0].id]);
  });

  it("keeps columns in one bubble and joins them in reading order", async () => {
    const base = createTestProvider();
    const provider: InferenceProvider = {
      ...base,
      ocr: {
        name: base.ocr.name,
        async recognize() {
          // Deliberately return left before right to prove assembly does not
          // depend on detector traversal order.
          return [
            {
              text: "second column",
              confidence: 0.97,
              bbox: { x: 240, y: 40, w: 20, h: 40 },
            },
            {
              text: "first column",
              confidence: 0.98,
              bbox: { x: 300, y: 40, w: 20, h: 40 },
            },
          ];
        },
      },
    };

    const result = await parsePage(image, provider, {
      pageId: "bubble-columns",
      name: "page.png",
      readingDirection: "rtl",
    });
    const bubble = result.textRegions.find((region) => region.fromBubble);

    expect(bubble?.text).toBe("first column second column");
  });
});
