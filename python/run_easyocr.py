"""
run_easyocr.py — CLI runner for the EasyOCR ONNX pipeline.

Usage examples
--------------
# Basic OCR on an image (models in ../models by default):
  python run_easyocr.py image.jpg

# Save annotated image and JSON results:
  python run_easyocr.py image.jpg --output result.jpg --json-output result.json

# Tune detection thresholds:
  python run_easyocr.py image.jpg --text-threshold 0.6 --low-text 0.3

# Specify a custom models directory:
  python run_easyocr.py image.jpg --models-dir D:/models/easyocr
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from easyocr import EasyOcr


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_results(img: np.ndarray, results: list[dict]) -> np.ndarray:
    """Draw bounding-box polygons and recognised text onto a copy of *img*."""
    vis     = img.copy()
    n_boxes = len(results)
    cmap = [
        (
            int(255 * (1 - i / max(n_boxes - 1, 1))),
            int(180 * i / max(n_boxes - 1, 1)),
            int(255 * i / max(n_boxes - 1, 1)),
        )
        for i in range(n_boxes)
    ] if n_boxes else []

    for idx, r in enumerate(results):
        box   = np.array(r["box"], dtype=np.int32)    # [4, 2]
        color = cmap[idx] if cmap else (0, 230, 0)

        # Draw quadrilateral
        cv2.polylines(vis, [box], isClosed=True, color=color, thickness=2)

        # Place text label above the top edge of the bounding box
        label = r["text"] if r["text"] else "<empty>"
        tx    = int(box[:, 0].min())
        ty    = max(int(box[:, 1].min()) - 6, 12)
        cv2.putText(
            vis, label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return vis


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EasyOCR ONNX inference pipeline (CRAFT + CRNN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("image", help="Path to the input image.")

    # Paths
    p.add_argument(
        "--models-dir", default="../models",
        help="Directory containing the ONNX model files.",
    )
    p.add_argument(
        "--output", default=None,
        help="Save the annotated image to this path (e.g. result.jpg).",
    )
    p.add_argument(
        "--json-output", default=None,
        help="Save the OCR results as JSON to this path.",
    )

    # Detection tuning
    p.add_argument(
        "--text-threshold", type=float, default=0.7,
        help="Minimum peak text score to retain a connected component.",
    )
    p.add_argument(
        "--low-text", type=float, default=0.4,
        help="Binary-map threshold applied to the combined score map.",
    )
    p.add_argument(
        "--link-threshold", type=float, default=0.4,
        help="Affinity (link) score threshold used when combining maps.",
    )

    # Hardware
    p.add_argument(
        "--gpu", action="store_true",
        help="Use CUDA execution provider (requires onnxruntime-gpu).",
    )

    # Output verbosity
    p.add_argument(
        "--no-print", action="store_true",
        help="Suppress per-region console output.",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Could not decode image: {img_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Initialise OCR engine
    # ------------------------------------------------------------------
    models_dir = Path(args.models_dir)
    print(f"[INFO] Loading models from : {models_dir.resolve()}")

    ocr = EasyOcr(
        models_dir,
        text_threshold=args.text_threshold,
        low_text=args.low_text,
        link_threshold=args.link_threshold,
        use_gpu=args.gpu,
    )

    # ------------------------------------------------------------------
    # Run OCR
    # ------------------------------------------------------------------
    print(f"[INFO] Running OCR on      : {img_path.resolve()}")
    t0      = time.perf_counter()
    results = ocr.ocr(img)
    elapsed = time.perf_counter() - t0

    print(f"[INFO] Regions detected    : {len(results)}  ({elapsed * 1000:.1f} ms)")

    if not args.no_print:
        for i, r in enumerate(results):
            print(f"  [{i + 1:3d}]  conf={r['conf']:.3f}  text={r['text']!r}")

    # ------------------------------------------------------------------
    # Optional outputs
    # ------------------------------------------------------------------
    if args.output:
        vis = _draw_results(img, results)
        cv2.imwrite(args.output, vis)
        print(f"[INFO] Annotated image     : {args.output}")

    if args.json_output:
        serialisable = [
            {"box": r["box"].tolist(), "text": r["text"], "conf": r["conf"]}
            for r in results
        ]
        Path(args.json_output).write_text(
            json.dumps(serialisable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[INFO] JSON results        : {args.json_output}")


if __name__ == "__main__":
    main()
