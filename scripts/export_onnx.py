#!/usr/bin/env python3
"""
Export the ShadowB YOLO26s manga segmentation model from .pt to .onnx.

This is the **only** step that requires Python + PyTorch + Ultralytics.
The web app (onnxruntime-web) cannot load .pt files directly.

Prerequisites (install once):
    pip install ultralytics onnx

Usage (from the project root):
    python scripts/export_onnx.py

Output:
    models/yolo26s-manga-seg.onnx  (ready for the web app)
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run:  pip install ultralytics onnx")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "models" / "best.pt"
ONNX_PATH = ROOT / "models" / "yolo26s-manga-seg.onnx"

if not PT_PATH.exists():
    print(f"ERROR: {PT_PATH} not found.")
    print("Download it first:")
    print(
        f"  curl -L -o {PT_PATH} "
        "https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation/resolve/main/best.pt"
    )
    sys.exit(1)

print(f"Loading {PT_PATH} ...")
model = YOLO(str(PT_PATH))

print("Exporting to ONNX (imgsz=1280, opset=17) ...")
model.export(format="onnx", imgsz=1280, opset=17, simplify=True)

# Ultralytics writes the .onnx next to the .pt by default.
# Move it to the expected location if needed.
default_onnx = PT_PATH.with_suffix(".onnx")
if default_onnx.exists() and not ONNX_PATH.exists():
    default_onnx.rename(ONNX_PATH)

if ONNX_PATH.exists():
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"✓ Exported: {ONNX_PATH} ({size_mb:.1f} MB)")
    print("  The web app will now auto-detect this model on next launch.")
else:
    print(f"WARNING: expected output at {ONNX_PATH} not found.")
    print(f"  Check {default_onnx} or the Ultralytics export log above.")
