import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import type { PageResult, Panel } from "../types";

/**
 * Renders the manga page with an interactive SVG overlay:
 *  - each panel gets a colored border + a numbered badge (reading order)
 *  - clicking a panel (or its badge) selects it and highlights it
 *  - hovering syncs highlight with the dialogue panel
 */
export function ImageViewer({ page }: { page: PageResult }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [display, setDisplay] = useState({ w: 0, h: 0 });
  const selectedPanelId = useStore((s) => s.selectedPanelId);
  const hoveredPanelId = useStore((s) => s.hoveredPanelId);
  const selectPanel = useStore((s) => s.selectPanel);
  const hoverPanel = useStore((s) => s.hoverPanel);

  const { imageWidth, imageHeight } = page;

  // Fit the image into the available space, preserving aspect ratio.
  useEffect(() => {
    const el = containerRef.current;
    if (!el || !imageWidth || !imageHeight) return;
    const compute = () => {
      const cw = el.clientWidth - 32; // padding
      const ch = el.clientHeight - 32;
      const scale = Math.min(cw / imageWidth, ch / imageHeight, 1.5);
      setDisplay({ w: imageWidth * scale, h: imageHeight * scale });
    };
    compute();
    const ro = new ResizeObserver(compute);
    ro.observe(el);
    return () => ro.disconnect();
  }, [imageWidth, imageHeight]);

  const scale = useMemo(
    () => (imageWidth ? display.w / imageWidth : 1),
    [display.w, imageWidth]
  );

  if (page.status !== "done") {
    return (
      <div ref={containerRef} className="flex h-full items-center justify-center">
        <StatusBadge page={page} />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex h-full items-center justify-center overflow-auto scroll-thin p-4"
    >
      <div
        className="relative no-select"
        style={{ width: display.w, height: display.h }}
      >
        {page.imageDataUrl && (
          <img
            src={page.imageDataUrl}
            alt={page.name}
            className="absolute inset-0 h-full w-full select-none rounded-sm"
            draggable={false}
          />
        )}

        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          preserveAspectRatio="none"
        >
          {page.panels.map((panel) => (
            <PanelShape
              key={panel.id}
              panel={panel}
              selected={panel.id === selectedPanelId}
              hovered={panel.id === hoveredPanelId}
              onSelect={() => selectPanel(panel.id)}
              onHover={(h) => hoverPanel(h ? panel.id : null)}
            />
          ))}
        </svg>
      </div>
      {scale <= 0 && null}
    </div>
  );
}

function PanelShape({
  panel,
  selected,
  hovered,
  onSelect,
  onHover,
}: {
  panel: Panel;
  selected: boolean;
  hovered: boolean;
  onSelect: () => void;
  onHover: (hovering: boolean) => void;
}) {
  const { bbox, color, order } = panel;
  const active = selected || hovered;
  const strokeWidth = selected ? 5 : hovered ? 3.5 : 2.5;

  // Badge position: top-left of the panel (or top-right reads more naturally
  // for RTL, but top-left keeps badge placement consistent with LTR UIs).
  const badgeR = Math.min(bbox.w, bbox.h) * 0.12 + 8;
  const bx = bbox.x + badgeR + 2;
  const by = bbox.y + badgeR + 2;

  return (
    <g
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      style={{ cursor: "pointer" }}
    >
      {/* Clickable fill (transparent) + border */}
      <rect
        x={bbox.x}
        y={bbox.y}
        width={bbox.w}
        height={bbox.h}
        fill={active ? hexToRgba(color, 0.12) : "transparent"}
        stroke={color}
        strokeWidth={strokeWidth}
        vectorEffect="non-scaling-stroke"
      />
      {/* Numbered badge */}
      <circle cx={bx} cy={by} r={badgeR} fill={color} stroke="#0b0d12" strokeWidth={2} />
      <text
        x={bx}
        y={by}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={badgeR * 1.1}
        fontWeight={700}
        fill="#ffffff"
        style={{ pointerEvents: "none" }}
      >
        {order}
      </text>
    </g>
  );
}

function StatusBadge({ page }: { page: PageResult }) {
  const labels: Record<string, string> = {
    pending: "等待处理…",
    segmenting: "实例分割中…",
    ocr: "文字识别中…",
    sorting: "阅读顺序重建中…",
    error: `出错: ${page.error ?? "未知错误"}`,
  };
  const isError = page.status === "error";
  return (
    <div
      className={`rounded-lg px-5 py-3 text-sm ring-1 ${
        isError
          ? "bg-red-950/40 text-red-300 ring-red-800"
          : "bg-ink-800 text-slate-300 ring-ink-600"
      }`}
    >
      {labels[page.status] ?? page.status}
    </div>
  );
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
