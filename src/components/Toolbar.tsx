import { useRef } from "react";
import { exportToJson, downloadJson } from "../lib/export";
import { useStore } from "../store";
import type { ReadingDirection } from "../types";

const DIRECTIONS: { value: ReadingDirection; label: string }[] = [
  { value: "rtl", label: "→ RTL (日漫)" },
  { value: "ltr", label: "← LTR (美漫)" },
  { value: "vertical", label: "↓ 条漫" },
];

export function Toolbar({
  onImportFiles,
  busy,
}: {
  onImportFiles: (files: File[]) => void;
  busy: boolean;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const readingDirection = useStore((s) => s.readingDirection);
  const setReadingDirection = useStore((s) => s.setReadingDirection);
  const providerLabel = useStore((s) => s.providerLabel);
  const pages = useStore((s) => s.pages);

  const handleExport = () => {
    const json = exportToJson(pages, readingDirection);
    downloadJson(json, `manga-parse-${pages.length}p.json`);
  };

  const doneCount = pages.filter((p) => p.status === "done").length;

  return (
    <header className="flex items-center gap-3 border-b border-ink-700 bg-ink-900 px-4 py-2.5">
      <div className="flex items-center gap-2">
        <span className="text-lg">📖</span>
        <h1 className="text-sm font-semibold tracking-tight text-slate-100">
          Manga Panel Parser
        </h1>
      </div>

      <div className="mx-2 h-5 w-px bg-ink-700" />

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onImportFiles(files);
          e.target.value = "";
        }}
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={busy}
        className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
      >
        {busy ? "解析中…" : "导入图片"}
      </button>

      <label className="flex items-center gap-2 text-xs text-slate-400">
        阅读方向
        <select
          value={readingDirection}
          onChange={(e) => setReadingDirection(e.target.value as ReadingDirection)}
          className="rounded-md border border-ink-600 bg-ink-800 px-2 py-1 text-xs text-slate-200 outline-none focus:border-accent"
        >
          {DIRECTIONS.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>
      </label>

      <div className="ml-auto flex items-center gap-3">
        <span className="rounded-full bg-ink-800 px-2.5 py-1 text-[11px] text-slate-400 ring-1 ring-ink-600">
          模型: {providerLabel || "加载中…"}
        </span>
        <button
          onClick={handleExport}
          disabled={doneCount === 0}
          className="rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-sm text-slate-200 transition hover:border-accent hover:text-white disabled:opacity-40"
        >
          导出 JSON
        </button>
      </div>
    </header>
  );
}
