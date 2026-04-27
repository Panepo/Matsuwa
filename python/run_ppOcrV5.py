"""
run_ppOcrV5.py — CLI runner for the PP-OCRv5 ONNX pipeline.

Usage examples
--------------
# Basic OCR on an image (models in ../models by default):
  python run_ppOcrV5.py image.jpg

# Save annotated image and JSON results:
  python run_ppOcrV5.py image.jpg --output result.jpg --json-output result.json

# Enable all pre-processing steps:
  python run_ppOcrV5.py image.jpg --dewarp --doc-ori --line-ori

# Specify a custom models directory:
  python run_ppOcrV5.py image.jpg --models-dir D:/models/ppocr
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
from ppOcrV5 import PpOcrV5


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_results(img: np.ndarray, results: list[dict]) -> np.ndarray:
    """Draw bounding boxes and recognised text onto a copy of *img*."""
    vis      = img.copy()
    n_boxes  = len(results)
    cmap     = [
        (int(255 * (1 - i / max(n_boxes - 1, 1))),
         int(180 * i / max(n_boxes - 1, 1)),
         int(255 * i / max(n_boxes - 1, 1)))
        for i in range(n_boxes)
    ] if n_boxes else []

    for idx, r in enumerate(results):
        box   = np.array(r["box"], dtype=np.int32)    # [4, 2]
        color = cmap[idx] if cmap else (0, 230, 0)

        # Draw polygon
        cv2.polylines(vis, [box], isClosed=True, color=color, thickness=2)

        # Place text label above the top edge of the box
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
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PP-OCRv5 ONNX inference pipeline",
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

    # Pipeline switches
    p.add_argument(
        "--dewarp", action="store_true",
        help="Apply UVDoc document dewarping before OCR.",
    )
    p.add_argument(
        "--doc-ori", action="store_true",
        help="Correct whole-document orientation (detects 0 / 90 / 180 / 270°).",
    )
    p.add_argument(
        "--line-ori", action="store_true",
        help="Correct per-line orientation (detects 0 / 180°) after detection.",
    )

    # Detection tuning
    p.add_argument("--det-limit", type=int, default=960,
                   help="Resize the longer edge to this length before detection.")
    p.add_argument("--det-thresh", type=float, default=0.3,
                   help="Probability threshold for the DBNet binary map.")
    p.add_argument("--box-thresh", type=float, default=0.5,
                   help="Minimum average probability score to keep a detected box.")
    p.add_argument("--unclip-ratio", type=float, default=1.6,
                   help="Box expansion ratio applied after detection.")

    # Hardware
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA (requires onnxruntime-gpu).")

    # Output verbosity
    p.add_argument("--no-print", action="store_true",
                   help="Suppress per-region console output.")

    return p


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

    print(f"Image : {img_path}  ({img.shape[1]}×{img.shape[0]} px)")

    # ------------------------------------------------------------------
    # Initialise model
    # ------------------------------------------------------------------
    models_dir = Path(args.models_dir)
    print(f"Models: {models_dir.resolve()}")

    try:
        ocr = PpOcrV5(
            models_dir,
            det_limit_side_len=args.det_limit,
            det_thresh=args.det_thresh,
            det_box_thresh=args.box_thresh,
            det_unclip_ratio=args.unclip_ratio,
            use_gpu=args.gpu,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    print("Running OCR pipeline …")
    t_start = time.perf_counter()
    results = ocr.ocr(
        img,
        dewarp=args.dewarp,
        correct_doc_ori=args.doc_ori,
        correct_line_ori=args.line_ori,
    )
    t_elapsed = time.perf_counter() - t_start
    print(f"Pipeline time: {t_elapsed:.3f} s")

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\nFound {len(results)} text region(s):\n")
    if not args.no_print:
        col_w = 60
        header = f"{'#':>4}  {'Text':<{col_w}}  Conf"
        print(header)
        print("-" * len(header))
        for i, r in enumerate(results, 1):
            text_preview = r["text"][:col_w]
            print(f"{i:>4}  {text_preview:<{col_w}}  {r['confidence']:.4f}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.json_output:
        out_json = Path(args.json_output)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nJSON  → {out_json.resolve()}")

    # ------------------------------------------------------------------
    # Save annotated image
    # ------------------------------------------------------------------
    if args.output:
        out_img = Path(args.output)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        vis = _draw_results(img, results)
        cv2.imwrite(str(out_img), vis)
        print(f"Image → {out_img.resolve()}")


if __name__ == "__main__":
    main()
