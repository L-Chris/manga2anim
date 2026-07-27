# Manga Panel Parser (漫画分镜自动解析与可视化工具)

A local desktop app (**Tauri + TypeScript + React + Tailwind CSS**) that parses
manga pages into structured panels and dialogue. It runs a local
**YOLO26s Manga instance-segmentation** model for panels / text / speech
bubbles, **PP-OCRv6 small** for text recognition, and a deterministic
**geometric reading-order algorithm** to reconstruct the reading flow for
RTL / LTR / vertical manga.

```
图片导入 → YOLO26s 实例分割 → PP-OCRv6 OCR → 几何排序（阅读顺序重建）
        → 彩色分镜标注与序号 → 右侧对话流 → 其他文本分类 → 手动修正 → JSON 导出
```

## Features

- Import multiple manga pages at once; each page's full parse result is retained.
- Automatic panel reading order from geometry + reading direction (RTL / LTR / vertical).
- Colored panel borders + numbered badges drawn over the original image.
- Click a panel or its badge to highlight it (synced with the dialogue panel).
- Right-side dialogue flow shown in reading order.
- SFX / interjections / narration / unclassified text collected into an
  "其他文本区域" (other text) section.
- Manual correction: reorder panels, edit recognized text, reclassify text kind,
  reassign text to a different panel, delete panels.
- Export the final result as JSON (`manga-panel-parser/v1` schema).

## Architecture

```
src/
  types.ts                     Data model (Page, Panel, TextRegion, export schema)
  store.ts                     Zustand app state + manual-edit actions
  lib/
    geometry.ts                bbox math: IoU, containment, NMS, union
    readingOrder.ts            ★ core: axis-banding geometric reading-order sort
    classify.ts                heuristic text-kind classification
    colors.ts                  distinct panel palette
    pipeline.ts                orchestration: segment → OCR → sort → assign → classify
    export.ts                  JSON export document builder + download
    image.ts                   image decode (data URL → RGBA)
    providers/
      types.ts                 Segmenter / OcrEngine / InferenceProvider interfaces
      onnxProvider.ts          YOLO26s + PP-OCRv6 via onnxruntime-web
      index.ts                 resolveProvider() — loads the required ONNX models
  components/
    Toolbar.tsx                import / direction / export controls
    ImageViewer.tsx            image + SVG panel overlay (borders, badges, click)
    DialoguePanel.tsx          right-side dialogue flow + editing + other text
    PageStrip.tsx              multi-page thumbnail navigation
    EmptyState.tsx             landing / import prompt
src-tauri/                     Rust backend (window, dialog + fs plugins, models dir)
```

### Core logic: geometric reading order (`src/lib/readingOrder.ts`)

Manga reading always progresses top → bottom across rows; RTL/LTR only change
the order *within* a row. The algorithm:

1. Greedily groups items into horizontal **bands** by vertical-overlap
   (tolerant of panels with different heights on the same visual row).
2. Emits bands top → bottom.
3. Orders items within each band along X — descending for RTL, ascending for
   LTR/vertical — with a vertical tie-break for a total, stable order.

It is pure, deterministic, and unit-tested (`readingOrder.test.ts`). The same
function orders panels on a page and text lines inside a panel.

## Getting started

Prerequisites: Node 20+, Rust (stable), and the Tauri
[platform dependencies](https://tauri.app/start/prerequisites/).

```bash
npm install
npm run tauri dev      # launch the desktop app (dev)
npm run test           # run unit tests (vitest)
npm run build          # typecheck + production web build
npm run tauri build    # package a distributable desktop binary
```

> Windows note: if `npm install` fails in an esbuild postinstall with
> `'node' is not recognized`, your Node directory (e.g. nvm-windows) isn't on
> the Win32 PATH inherited by child `cmd.exe` processes. Add it to the system
> PATH and retry.

## Model assets

Release installers bundle the production model assets, so installed builds run
YOLO26s + PP-OCRv6 fully locally out of the box. Source builds require all model
assets listed below; when any asset is missing or cannot be loaded, processing
stops with an explicit error.

### Reference segmentation model

[`ShadowB/Manga109-panel-balloon-text-yolov26-segmentation`](https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation)
— an Ultralytics **YOLO26s** instance-segmentation model trained on
Manga109-derived data, exactly matching the goal's three classes:

| Class ID | Model label | Our schema |
|----------|-------------|------------|
| 0 | `frame` | `panel` |
| 1 | `text` | `text` |
| 2 | `balloon` | `bubble` |

Trained at **`imgsz=1280`**. The provider reads these values as its defaults
(`DEFAULT_SEG_CLASS_NAMES`, `DEFAULT_SEG_INPUT_SIZE` in
`src/lib/providers/onnxProvider.ts`), and **class mapping is model-driven**:
`normalizeClassName()` folds `frame/panel` → panel and
`balloon/ballon/bubble` → bubble, so a model that names the classes
differently still maps correctly.

### The one required conversion step

The HF repo ships **only `best.pt` / `last.pt` (Ultralytics/PyTorch)** — there
is **no `.onnx`**. The web app uses **onnxruntime-web (browser WASM), which
loads `.onnx` only**, so the weights must be exported once. `best.pt` is
already downloaded into `models/` (git-ignored). Convert it with:

```bash
pip install ultralytics onnx
python scripts/export_onnx.py
```

This writes `models/yolo26s-manga-seg.onnx` at `imgsz=1280, opset=17`. Re-fetch
the weights manually if needed:

```bash
curl -L -o models/best.pt \
  https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation/resolve/main/best.pt
```

> Why not run the `.pt` directly in the app? onnxruntime-web cannot load
> PyTorch checkpoints, and the in-browser runtime has no Python/PyTorch. The
> `.pt → .onnx` export is the standard, lossless Ultralytics path and is the
> only place Python is needed; inference itself stays fully local in the
> browser via WASM.

### Where the app looks for weights

- **Dev:** `models/` at the project root, streamed by Vite at `/models`.
- **Packaged:** embedded `/models` assets inside the Tauri frontend bundle.

The tag-release workflow downloads pinned upstream revisions, verifies the
published SHA-256 values, exports the YOLO checkpoint to ONNX, and requires all
model files before packaging. The large weights remain git-ignored.

Expected filenames:

| File | Model | Output |
|------|-------|--------|
| `yolo26s-manga-seg.onnx` | YOLO26s manga instance segmentation | `frame/text/balloon` + mask prototypes |
| `ppocrv6-det.onnx` | PP-OCRv6 small text detector (DBNet) | probability map `[1,1,H,W]` |
| `ppocrv6-rec.onnx` | PP-OCRv6 small text recognizer | CTC logits `[1,T,numClasses]` |
| `ppocrv6_dict.txt` | OCR character dictionary | 18708 chars (50 languages), committed in `models/` |

The v6 dictionary ships in `models/` (committed, so OCR works out of the box;
the old `ppocr_keys_v1.txt` is **wrong** for v6 — its 6623 classes don't match
the small rec head's 18710 classes and would silently drop most characters).
Re-fetch it with:

```bash
curl -L -o models/ppocrv6_dict.txt \
  https://cdn.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@main/ppocr/utils/dict/ppocrv6_dict.txt
```
The det/rec ONNX files are **not** in the HF segmentation repo — fetch them from
the official PaddlePaddle ONNX exports, e.g.:

```bash
# small tier (matches the goal's "PP-OCRv6 small"); swap _small_ for _tiny_/_medium_ as needed
curl -L -o models/ppocrv6-det.onnx \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx
curl -L -o models/ppocrv6-rec.onnx \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx
```

> Filenames above follow the PaddlePaddle ONNX export layout; if a release uses
> a different blob name, point `detModel`/`recModel` in `OnnxProviderOptions` at
> whatever you saved. The dictionary name is configurable via `dictModel`.

### PP-OCRv6 inference details

The full det+rec pipeline is implemented in pure TypeScript (no OpenCV) in
`src/lib/ppocr.ts`, as **pure, unit-tested functions**:

- **Det (DBNet):** `detPreprocess` resizes the image so the longest side ≤ 960
  with both dims rounded to multiples of 32, ImageNet-normalized
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). `detPostprocess`
  binarizes the probability map, runs two-pass connected components, filters by
  min-area, and unclip-expands each box back into original-image coordinates.
  Detector boxes are then constrained to YOLO text/bubble regions so artwork
  and panel-border hits do not reach recognition.
- **Rec (CTC):** `recPreprocess` crops each detected box, resizes to a fixed
  height of 48 preserving aspect ratio (padded to width 320), and BGR-normalizes
  pixels to `[-1,1]`. `ctcDecode` does greedy argmax → collapse repeats → drop
  the blank token (index 0) → map `class i → dictionary[i-1]`. Empty results
  and lines below the calibrated 0.8 mean confidence threshold are discarded.

`resolveProvider()` requires the complete YOLO26s + PP-OCRv6 model set (shown in
the toolbar's "模型" badge after loading). The YOLO post-processing (letterbox
preprocessing, per-class NMS, prototype-mask decode) is likewise pure and
unit-tested (`postprocessYolo`, `decodeMask`, `letterbox`), verified against the
real model parameters (3 classes, `imgsz=1280`, 32 mask coefficients) with
synthetic tensors.

## Export format

`manga-panel-parser/v1` — see `ExportDocument` in `src/types.ts`. Each page
includes `panels` (with `order`, `color`, `bbox`, linked `textIds`) and
`textRegions` (with `kind`, `text`, `bbox`, `panelId`), plus a convenience
`otherTexts` array.
