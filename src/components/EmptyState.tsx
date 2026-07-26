import { useRef } from "react";

const PIPELINE_STEPS = [
  "图片导入",
  "YOLO26s 实例分割",
  "PP-OCRv6 文字识别",
  "几何排序（阅读顺序重建）",
  "彩色分镜标注",
  "右侧对话流",
  "其他文本分类",
  "手动修正",
  "JSON 导出",
];

export function EmptyState({
  onImportFiles,
  busy,
}: {
  onImportFiles: (files: File[]) => void;
  busy: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="max-w-lg text-center">
        <div className="mb-4 text-5xl">📖</div>
        <h2 className="text-xl font-semibold text-slate-100">
          漫画分镜自动解析
        </h2>
        <p className="mt-2 text-sm text-slate-400">
          导入一组漫画图片，自动识别分镜、对话气泡与文本区域，
          并按阅读方向重建阅读顺序。
        </p>

        <input
          ref={inputRef}
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
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="mt-6 rounded-lg bg-accent px-6 py-2.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
        >
          选择漫画图片…
        </button>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-1.5 text-[11px] text-slate-500">
          {PIPELINE_STEPS.map((step, i) => (
            <span key={step} className="flex items-center gap-1.5">
              <span className="rounded bg-ink-800 px-2 py-0.5 ring-1 ring-ink-700">
                {step}
              </span>
              {i < PIPELINE_STEPS.length - 1 && <span className="text-ink-600">→</span>}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
