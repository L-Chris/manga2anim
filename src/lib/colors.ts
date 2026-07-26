/**
 * Distinct, high-contrast colors for panel borders/badges. Cycles if there are
 * more panels than colors. Chosen to stay legible over both black-and-white and
 * toned manga art.
 */
export const PANEL_COLORS = [
  "#ef4444", // red
  "#f59e0b", // amber
  "#10b981", // emerald
  "#3b82f6", // blue
  "#a855f7", // purple
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
  "#84cc16", // lime
  "#06b6d4", // cyan
  "#eab308", // yellow
  "#8b5cf6", // violet
];

export function panelColor(index: number): string {
  return PANEL_COLORS[((index % PANEL_COLORS.length) + PANEL_COLORS.length) % PANEL_COLORS.length];
}
