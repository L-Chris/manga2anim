import { useStore } from "../store";
import { isDialogueKind, isOtherText } from "../types";
import type { PageResult, Panel, TextKind, TextRegion } from "../types";

const KIND_LABELS: Record<TextKind, string> = {
  dialogue: "对话",
  thought: "心理",
  narration: "旁白",
  sfx: "拟声词",
  interjection: "语气词",
  unknown: "未分类",
};

const KIND_OPTIONS: TextKind[] = [
  "dialogue",
  "thought",
  "narration",
  "sfx",
  "interjection",
  "unknown",
];

export function DialoguePanel({ page }: { page: PageResult }) {
  const panels = page.panels;
  const otherTexts = page.textRegions.filter(isOtherText);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-ink-700 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-100">对话流</h2>
        <p className="mt-0.5 text-[11px] text-slate-500">
          按阅读顺序 · {panels.length} 个分镜
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto scroll-thin px-3 py-3">
        {panels.length === 0 && (
          <p className="px-1 py-8 text-center text-xs text-slate-500">
            本页未检测到分镜。
          </p>
        )}

        <div className="space-y-3">
          {panels.map((panel) => (
            <PanelCard key={panel.id} page={page} panel={panel} />
          ))}
        </div>

        {/* Other text section */}
        <div className="mt-5">
          <div className="mb-2 flex items-center gap-2 px-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              其他文本区域
            </span>
            <span className="rounded-full bg-ink-800 px-1.5 text-[10px] text-slate-500">
              {otherTexts.length}
            </span>
          </div>
          {otherTexts.length === 0 ? (
            <p className="px-1 text-[11px] text-slate-600">无</p>
          ) : (
            <div className="space-y-2">
              {otherTexts.map((region) => (
                <TextRow key={region.id} page={page} region={region} compact />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PanelCard({ page, panel }: { page: PageResult; panel: Panel }) {
  const selectedPanelId = useStore((s) => s.selectedPanelId);
  const selectPanel = useStore((s) => s.selectPanel);
  const hoverPanel = useStore((s) => s.hoverPanel);
  const movePanel = useStore((s) => s.movePanel);
  const deletePanel = useStore((s) => s.deletePanel);

  const selected = panel.id === selectedPanelId;
  const dialogueRegions = panel.textIds
    .map((id) => page.textRegions.find((t) => t.id === id))
    .filter((t): t is TextRegion => Boolean(t) && isDialogueKind(t!.kind));

  return (
    <div
      onClick={() => selectPanel(panel.id)}
      onMouseEnter={() => hoverPanel(panel.id)}
      onMouseLeave={() => hoverPanel(null)}
      className={`cursor-pointer rounded-lg border p-2.5 transition ${
        selected
          ? "border-transparent bg-ink-800 ring-2"
          : "border-ink-700 bg-ink-850 hover:border-ink-600"
      }`}
      style={selected ? { boxShadow: `0 0 0 2px ${panel.color}` } : undefined}
    >
      <div className="mb-2 flex items-center gap-2">
        <span
          className="flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold text-white"
          style={{ backgroundColor: panel.color }}
        >
          {panel.order}
        </span>
        <span className="text-[11px] text-slate-500">
          分镜 · {Math.round(panel.confidence * 100)}%
        </span>
        <div className="ml-auto flex items-center gap-1">
          <IconBtn title="上移" onClick={(e) => { e.stopPropagation(); movePanel(page.pageId, panel.id, -1); }}>
            ↑
          </IconBtn>
          <IconBtn title="下移" onClick={(e) => { e.stopPropagation(); movePanel(page.pageId, panel.id, 1); }}>
            ↓
          </IconBtn>
          <IconBtn title="删除分镜" onClick={(e) => { e.stopPropagation(); deletePanel(page.pageId, panel.id); }}>
            ✕
          </IconBtn>
        </div>
      </div>

      {dialogueRegions.length === 0 ? (
        <p className="px-1 text-[11px] italic text-slate-600">（无对话）</p>
      ) : (
        <div className="space-y-2">
          {dialogueRegions.map((region) => (
            <TextRow key={region.id} page={page} region={region} />
          ))}
        </div>
      )}
    </div>
  );
}

function TextRow({
  page,
  region,
  compact,
}: {
  page: PageResult;
  region: TextRegion;
  compact?: boolean;
}) {
  const editText = useStore((s) => s.editText);
  const reclassifyText = useStore((s) => s.reclassifyText);
  const reassignText = useStore((s) => s.reassignText);

  return (
    <div
      className={`rounded-md border border-ink-700 bg-ink-900 p-2 ${
        compact ? "" : ""
      }`}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="mb-1 flex items-center gap-1.5">
        <span className="rounded bg-ink-800 px-1.5 py-0.5 text-[10px] text-slate-400">
          {KIND_LABELS[region.kind]}
        </span>
        {region.fromBubble && (
          <span className="rounded bg-indigo-950 px-1.5 py-0.5 text-[10px] text-indigo-300">
            气泡
          </span>
        )}
        {region.manual && (
          <span className="rounded bg-amber-950 px-1.5 py-0.5 text-[10px] text-amber-300">
            已修改
          </span>
        )}
        <select
          value={region.kind}
          onChange={(e) =>
            reclassifyText(page.pageId, region.id, e.target.value as TextKind)
          }
          className="ml-auto rounded border border-ink-600 bg-ink-800 px-1 py-0.5 text-[10px] text-slate-300 outline-none focus:border-accent"
        >
          {KIND_OPTIONS.map((k) => (
            <option key={k} value={k}>
              {KIND_LABELS[k]}
            </option>
          ))}
        </select>
      </div>

      <textarea
        value={region.text}
        onChange={(e) => editText(page.pageId, region.id, e.target.value)}
        rows={Math.min(4, Math.max(1, Math.ceil(region.text.length / 24)))}
        className="w-full resize-none rounded border border-transparent bg-transparent text-sm text-slate-100 outline-none focus:border-ink-600 focus:bg-ink-950"
      />

      {/* Panel reassignment */}
      <div className="mt-1 flex items-center gap-1.5">
        <span className="text-[10px] text-slate-500">归属分镜:</span>
        <select
          value={region.panelId ?? ""}
          onChange={(e) =>
            reassignText(page.pageId, region.id, e.target.value || null)
          }
          className="rounded border border-ink-600 bg-ink-800 px-1 py-0.5 text-[10px] text-slate-300 outline-none focus:border-accent"
        >
          <option value="">（其他文本）</option>
          {page.panels.map((p) => (
            <option key={p.id} value={p.id}>
              分镜 {p.order}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function IconBtn({
  children,
  title,
  onClick,
}: {
  children: React.ReactNode;
  title: string;
  onClick: (e: React.MouseEvent) => void;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className="flex h-5 w-5 items-center justify-center rounded text-xs text-slate-400 transition hover:bg-ink-700 hover:text-white"
    >
      {children}
    </button>
  );
}
