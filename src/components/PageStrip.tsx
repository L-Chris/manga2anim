import { useStore } from "../store";

/** Horizontal thumbnail strip for multi-page navigation. */
export function PageStrip() {
  const pages = useStore((s) => s.pages);
  const currentIndex = useStore((s) => s.currentPageIndex);
  const setCurrentPage = useStore((s) => s.setCurrentPage);

  if (pages.length <= 1) return null;

  return (
    <div className="flex items-center gap-2 overflow-x-auto scroll-thin border-t border-ink-700 bg-ink-900 px-3 py-2">
      {pages.map((page, i) => {
        const active = i === currentIndex;
        return (
          <button
            key={page.pageId}
            onClick={() => setCurrentPage(i)}
            title={page.name}
            className={`relative h-16 w-12 shrink-0 overflow-hidden rounded border transition ${
              active
                ? "border-accent ring-2 ring-accent/40"
                : "border-ink-700 hover:border-ink-600"
            }`}
          >
            {page.imageDataUrl ? (
              <img
                src={page.imageDataUrl}
                alt={page.name}
                className="h-full w-full object-cover"
                draggable={false}
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center bg-ink-800 text-[10px] text-slate-500">
                {i + 1}
              </div>
            )}
            <span className="absolute bottom-0 left-0 right-0 bg-black/60 text-center text-[9px] text-white">
              {i + 1}
              {page.status === "error" ? " ⚠" : page.status === "done" ? "" : " …"}
            </span>
          </button>
        );
      })}
    </div>
  );
}
