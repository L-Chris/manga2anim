import { create } from "zustand";
import type {
  PageResult,
  PageStatus,
  Panel,
  ReadingDirection,
  TextKind,
} from "./types";

interface ImportedImage {
  pageId: string;
  name: string;
  dataUrl: string;
  sourcePath?: string;
}

interface AppState {
  // Data
  pages: PageResult[];
  currentPageIndex: number;
  readingDirection: ReadingDirection;
  providerLabel: string;

  // UI state
  selectedPanelId: string | null;
  hoveredPanelId: string | null;

  // Provider / import
  setProviderLabel: (label: string) => void;
  setReadingDirection: (dir: ReadingDirection) => void;
  initPages: (images: ImportedImage[], dir: ReadingDirection) => void;
  setPageResult: (pageId: string, result: PageResult) => void;
  setPageStatus: (pageId: string, status: PageStatus, error?: string) => void;

  // Navigation
  setCurrentPage: (index: number) => void;

  // Selection
  selectPanel: (panelId: string | null) => void;
  hoverPanel: (panelId: string | null) => void;

  // Manual editing
  reorderPanels: (pageId: string, panelIdsInOrder: string[]) => void;
  movePanel: (pageId: string, panelId: string, direction: -1 | 1) => void;
  editText: (pageId: string, textId: string, text: string) => void;
  reclassifyText: (pageId: string, textId: string, kind: TextKind) => void;
  reassignText: (pageId: string, textId: string, panelId: string | null) => void;
  deletePanel: (pageId: string, panelId: string) => void;
}

/** Recompute panel.order + color from array position, and relink textIds. */
function normalizePanels(panels: Panel[]): Panel[] {
  return panels.map((p, i) => ({ ...p, order: i + 1 }));
}

function updatePage(
  state: AppState,
  pageId: string,
  updater: (page: PageResult) => PageResult
): Partial<AppState> {
  return {
    pages: state.pages.map((p) => (p.pageId === pageId ? updater(p) : p)),
  };
}

export const useStore = create<AppState>((set) => ({
  pages: [],
  currentPageIndex: 0,
  readingDirection: "rtl",
  providerLabel: "",
  selectedPanelId: null,
  hoveredPanelId: null,

  setProviderLabel: (label) => set({ providerLabel: label }),
  setReadingDirection: (dir) =>
    set((state) => ({
      readingDirection: dir,
      pages: state.pages.map((p) => ({ ...p, readingDirection: dir })),
    })),

  initPages: (images, dir) =>
    set({
      readingDirection: dir,
      currentPageIndex: 0,
      selectedPanelId: null,
      pages: images.map(
        (img): PageResult => ({
          pageId: img.pageId,
          name: img.name,
          imageDataUrl: img.dataUrl,
          sourcePath: img.sourcePath,
          imageWidth: 0,
          imageHeight: 0,
          readingDirection: dir,
          panels: [],
          textRegions: [],
          status: "pending",
        })
      ),
    }),

  setPageResult: (pageId, result) =>
    set((state) => updatePage(state, pageId, () => result)),

  setPageStatus: (pageId, status, error) =>
    set((state) =>
      updatePage(state, pageId, (p) => ({ ...p, status, error }))
    ),

  setCurrentPage: (index) => set({ currentPageIndex: index, selectedPanelId: null }),
  selectPanel: (panelId) => set({ selectedPanelId: panelId }),
  hoverPanel: (panelId) => set({ hoveredPanelId: panelId }),

  reorderPanels: (pageId, panelIdsInOrder) =>
    set((state) =>
      updatePage(state, pageId, (page) => {
        const byId = new Map(page.panels.map((p) => [p.id, p]));
        const reordered = panelIdsInOrder
          .map((id) => byId.get(id))
          .filter((p): p is Panel => Boolean(p));
        return { ...page, panels: normalizePanels(reordered) };
      })
    ),

  movePanel: (pageId, panelId, direction) =>
    set((state) =>
      updatePage(state, pageId, (page) => {
        const panels = page.panels.slice();
        const idx = panels.findIndex((p) => p.id === panelId);
        if (idx === -1) return page;
        const target = idx + direction;
        if (target < 0 || target >= panels.length) return page;
        [panels[idx], panels[target]] = [panels[target], panels[idx]];
        panels[idx] = { ...panels[idx], manual: true };
        panels[target] = { ...panels[target], manual: true };
        return { ...page, panels: normalizePanels(panels) };
      })
    ),

  editText: (pageId, textId, text) =>
    set((state) =>
      updatePage(state, pageId, (page) => ({
        ...page,
        textRegions: page.textRegions.map((t) =>
          t.id === textId ? { ...t, text, manual: true } : t
        ),
      }))
    ),

  reclassifyText: (pageId, textId, kind) =>
    set((state) =>
      updatePage(state, pageId, (page) => ({
        ...page,
        textRegions: page.textRegions.map((t) =>
          t.id === textId ? { ...t, kind, manual: true } : t
        ),
      }))
    ),

  reassignText: (pageId, textId, panelId) =>
    set((state) =>
      updatePage(state, pageId, (page) => {
        const textRegions = page.textRegions.map((t) =>
          t.id === textId ? { ...t, panelId, manual: true } : t
        );
        // Rebuild each panel's textIds from the (possibly new) assignments.
        const panels = page.panels.map((panel) => ({
          ...panel,
          textIds: textRegions
            .filter((t) => t.panelId === panel.id)
            .map((t) => t.id),
        }));
        return { ...page, textRegions, panels };
      })
    ),

  deletePanel: (pageId, panelId) =>
    set((state) =>
      updatePage(state, pageId, (page) => {
        const panels = normalizePanels(page.panels.filter((p) => p.id !== panelId));
        const textRegions = page.textRegions.map((t) =>
          t.panelId === panelId ? { ...t, panelId: null } : t
        );
        return { ...page, panels, textRegions };
      })
    ),
}));

// Selectors
export function selectCurrentPage(state: AppState): PageResult | undefined {
  return state.pages[state.currentPageIndex];
}

export type { ImportedImage };
