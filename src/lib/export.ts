import type {
  ExportDocument,
  ExportPage,
  PageResult,
  ReadingDirection,
  TextRegion,
} from "../types";
import { isOtherText } from "../types";

/** Build the export document from the retained per-page results. */
export function buildExportDocument(
  pages: PageResult[],
  readingDirection: ReadingDirection,
  exportedAt: string
): ExportDocument {
  return {
    schema: "manga-panel-parser/v1",
    exportedAt,
    readingDirection,
    pages: pages
      .filter((p) => p.status === "done")
      .map((p): ExportPage => {
        const otherTexts: TextRegion[] = p.textRegions.filter(isOtherText);
        return {
          pageId: p.pageId,
          name: p.name,
          sourcePath: p.sourcePath,
          imageWidth: p.imageWidth,
          imageHeight: p.imageHeight,
          readingDirection: p.readingDirection,
          panels: p.panels,
          textRegions: p.textRegions,
          otherTexts,
        };
      }),
  };
}

/** Serialize the export document to pretty JSON. */
export function exportToJson(pages: PageResult[], readingDirection: ReadingDirection): string {
  const doc = buildExportDocument(pages, readingDirection, new Date().toISOString());
  return JSON.stringify(doc, null, 2);
}

/** Trigger a browser/Tauri download of a JSON string. */
export function downloadJson(json: string, filename: string): void {
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
