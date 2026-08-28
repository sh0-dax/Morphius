#!/usr/bin/env python3
"""
Morphius — YOLO26n model export + int8 quantization
====================================================
Exports Ultralytics YOLO26n (nano) to ONNX and quantizes it to INT8 for
use by js/vision.js (onnxruntime-web, WebGPU first / WASM fallback).

DO NOT auto-run this in CI. Run it ONCE on a machine with ultralytics
installed, controlled GPU/CPU, and network access, then commit the
resulting `models/yolo26n_int8.onnx`.

Setup (run these yourself, this script does NOT install anything):
-------------------------------------------------------------------
    pip install -U ultralytics onnx onnxruntime onnxslim

    # yolo26n.pt is downloaded automatically by ultralytics on first use
    # (from https://docs.ultralytics.com/models/yolo26/)

Usage:
    python scripts/export_model.py
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PT = ROOT / "yolo26n.pt"
ONNX_FP32 = ROOT / "yolo26n.onnx"
ONNX_INT8 = ROOT / "models" / "yolo26n_int8.onnx"

# Export settings:
#   format='onnx'          -> ONNX graph
#   dynamic=False          -> fixed 1x3x640x640 input (simplest for browser)
#   simplify=True          -> onnxslim graph cleanup
#   opset=17               -> well-supported by onnxruntime-web
# IMPORTANT: this uses the DEFAULT export which produces the ONE-TO-ONE head
# (end-to-end, NMS-free) → output shape [1, 300, 6], each row
# [x1, y1, x2, y2, score, class]. js/vision.js's parser expects this shape.
# Do NOT pass end2end=False unless you also re-enable NMS on the JS side.
EXPORT_KWARGS = dict(format="onnx", dynamic=False, simplify=True, opset=17)


def main() -> None:
    print(f"[export] loading {MODEL_PT} ...")
    model = YOLO(MODEL_PT)  # auto-downloads weights on first run

    print(f"[export] exporting to ONNX (one-to-one default) ...")
    model.export(**EXPORT_KWARGS)
    fp32_mb = ONNX_FP32.stat().st_size / (1024 * 1024)
    print(f"[export] fp32 ONNX size: {fp32_mb:.2f} MB")

    # ---- Dynamic int8 quantization (fast; no calibration data required) ----
    # If accuracy drops too much, switch to STATIC quantization: calibrate on
    # ~100 COCO val images and pass a RepresentativeDataset to
    # onnxruntime.quantization.quantize_static instead.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[export] quantizing to int8 -> {ONNX_INT8} ...")
    quantize_dynamic(
        model_input=str(ONNX_FP32),
        model_output=str(ONNX_INT8),
        weight_type=QuantType.QInt8,
    )
    int8_mb = ONNX_INT8.stat().st_size / (1024 * 1024)
    print(f"[export] int8 ONNX size: {int8_mb:.2f} MB")
    print(f"[export] done. commit `{ONNX_INT8.name}` and update the YOLO_MODEL_URL "
          f"path in js/vision.js if you renamed it.")


if __name__ == "__main__":
    main()
