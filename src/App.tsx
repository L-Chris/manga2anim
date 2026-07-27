import { useCallback, useEffect, useRef, useState } from "react";
import { Toolbar } from "./components/Toolbar";
import { ImageViewer } from "./components/ImageViewer";
import { DialoguePanel } from "./components/DialoguePanel";
import { PageStrip } from "./components/PageStrip";
import { EmptyState } from "./components/EmptyState";
import { decodeImage, fileToDataUrl } from "./lib/image";
import { parsePage } from "./lib/pipeline";
import { resolveProvider, type InferenceProvider } from "./lib/providers";
import { selectCurrentPage, useStore, type ImportedImage } from "./store";

let pageSeq = 0;

export default function App() {
  const pages = useStore((s) => s.pages);
  const currentPage = useStore(selectCurrentPage);
  const readingDirection = useStore((s) => s.readingDirection);
  const initPages = useStore((s) => s.initPages);
  const setPageResult = useStore((s) => s.setPageResult);
  const setPageStatus = useStore((s) => s.setPageStatus);
  const setProviderLabel = useStore((s) => s.setProviderLabel);

  const providerRef = useRef<InferenceProvider | null>(null);
  const [busy, setBusy] = useState(false);
  const [overall, setOverall] = useState<{ done: number; total: number } | null>(null);

  // Resolve the inference provider once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const provider = await resolveProvider(modelsDirUrl());
      if (cancelled) return;
      providerRef.current = provider;
      setProviderLabel(provider.label);
    })();
    return () => {
      cancelled = true;
    };
  }, [setProviderLabel]);

  const runPipeline = useCallback(
    async (images: ImportedImage[]) => {
      const provider = providerRef.current ?? (await resolveProvider(modelsDirUrl()));
      providerRef.current = provider;
      setProviderLabel(provider.label);

      setBusy(true);
      setOverall({ done: 0, total: images.length });
      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        setOverall({ done: i, total: images.length });
        try {
          setPageStatus(img.pageId, "segmenting");
          const decoded = await decodeImage(img.dataUrl);
          const result = await parsePage(decoded, provider, {
            pageId: img.pageId,
            name: img.name,
            sourcePath: img.sourcePath,
            imageDataUrl: img.dataUrl,
            readingDirection,
          });
          setPageResult(img.pageId, result);
        } catch (err) {
          setPageStatus(img.pageId, "error", String(err));
        }
      }
      setOverall({ done: images.length, total: images.length });
      setBusy(false);
    },
    [readingDirection, setPageResult, setPageStatus, setProviderLabel]
  );

  const handleImportFiles = useCallback(
    async (files: File[]) => {
      const imageFiles = files.filter((f) => f.type.startsWith("image/"));
      if (imageFiles.length === 0) return;
      const images: ImportedImage[] = [];
      for (const file of imageFiles) {
        pageSeq += 1;
        const dataUrl = await fileToDataUrl(file);
        images.push({
          pageId: `page_${pageSeq}`,
          name: file.name,
          dataUrl,
        });
      }
      initPages(images, readingDirection);
      await runPipeline(images);
    },
    [initPages, readingDirection, runPipeline]
  );

  const hasPages = pages.length > 0;

  return (
    <div className="flex h-full flex-col">
      <Toolbar onImportFiles={handleImportFiles} busy={busy} />

      {!hasPages ? (
        <EmptyState onImportFiles={handleImportFiles} busy={busy} />
      ) : (
        <div className="flex min-h-0 flex-1">
          {/* Left: image viewer */}
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="min-h-0 flex-1">
              {currentPage && <ImageViewer page={currentPage} />}
            </div>
            <PageStrip />
          </div>

          {/* Right: dialogue flow + other text */}
          <div className="w-[380px] shrink-0 border-l border-ink-700 bg-ink-900">
            {currentPage && <DialoguePanel page={currentPage} />}
          </div>
        </div>
      )}

      {overall && busy && (
        <div className="pointer-events-none fixed bottom-4 left-1/2 z-50 -translate-x-1/2 rounded-full bg-ink-800 px-4 py-2 text-sm shadow-lg ring-1 ring-ink-600">
          Processing page {Math.min(overall.done + 1, overall.total)} / {overall.total}…
        </div>
      )}
    </div>
  );
}

/** Base URL for model assets served by Vite and embedded in packaged builds. */
function modelsDirUrl(): string {
  return "/models";
}
