"""
run_vehSummary.py — CLI runner: OCR → vehicle information extraction via LLM.

Usage examples
--------------
  # Basic extraction (reads Ollama settings from .env):
    python run_vehSummary.py ../images/car.jpg

  # Save JSON output:
    python run_vehSummary.py ../images/car.jpg --output-json ./outputs/veh.json

  # Enable all OCR pre-processing:
    python run_vehSummary.py ../images/car.jpg --dewarp --doc-ori --line-ori

  # Override Ollama settings at the command line:
    python run_vehSummary.py ../images/car.jpg --ollama-url http://localhost:11434 --ollama-model llama3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from ppOcrV5 import PpOcrV5
from vehSummary import VehSummary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PP-OCRv5 + Ollama LLM vehicle information extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional
    p.add_argument("image", help="Path to the input image.")

    # Paths
    p.add_argument(
        "--models-dir", default="../models",
        help="Directory containing the ONNX model files.",
    )
    p.add_argument(
        "--output-json", default=None,
        help="Save extracted vehicle info as JSON to this path.",
    )
    p.add_argument(
        "--env", default=None,
        help="Path to the .env file (default: .env next to this script).",
    )

    # OCR pipeline switches
    p.add_argument("--dewarp",   action="store_true",
                   help="Apply UVDoc document dewarping before OCR.")
    p.add_argument("--doc-ori",  action="store_true",
                   help="Correct whole-document orientation (0/90/180/270°).")
    p.add_argument("--line-ori", action="store_true",
                   help="Correct per-line orientation (0/180°) after detection.")

    # Detection tuning
    p.add_argument("--det-limit",    type=int,   default=960,
                   help="Resize the longer edge to this length before detection.")
    p.add_argument("--det-thresh",   type=float, default=0.3,
                   help="Probability threshold for the DBNet binary map.")
    p.add_argument("--box-thresh",   type=float, default=0.5,
                   help="Minimum average probability score to keep a detected box.")
    p.add_argument("--unclip-ratio", type=float, default=1.6,
                   help="Box expansion ratio applied after detection.")

    # Hardware
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA for ONNX inference (requires onnxruntime-gpu).")

    # LLM overrides
    p.add_argument("--ollama-url",   default=None,
                   help="Ollama server URL (overrides .env OLLAMA_URL).")
    p.add_argument("--ollama-model", default=None,
                   help="Ollama model tag (overrides .env OLLAMA_MODEL).")
    p.add_argument("--timeout", type=int, default=None, metavar="SECONDS",
                   help="LLM request timeout in seconds (default: no limit).")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

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
    # OCR
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

    print("Running OCR …")
    ocr_results = ocr.ocr(
        img,
        dewarp=args.dewarp,
        correct_doc_ori=args.doc_ori,
        correct_line_ori=args.line_ori,
    )
    print(f"  Found {len(ocr_results)} text region(s).")

    # ------------------------------------------------------------------
    # Vehicle summary extraction
    # ------------------------------------------------------------------
    extractor = VehSummary(
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        env_path=args.env,
        request_timeout=args.timeout,
    )

    print(f"\nQuerying LLM  : {extractor.ollama_model}")
    print(f"Ollama server : {extractor.ollama_url}")

    # Show the assembled text so the user can verify the OCR join
    assembled = extractor.assemble_text(ocr_results)
    print("\n" + "─" * 60)
    print("Assembled OCR text:")
    print(assembled)
    print("─" * 60)

    print("\nLLM response:")
    print("─" * 60)
    try:
        info = extractor.extract(ocr_results)
    except (ConnectionError, RuntimeError, TimeoutError) as exc:
        print(f"\n[ERROR] LLM query failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if "_assembled_text" in info and len(info) == 1:
        print("[WARN] Marker '***VEHICLEINFO***' not found in OCR text — LLM was not queried.")
        print("─" * 60)
        sys.exit(0)
    print("─" * 60)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    display = {k: v for k, v in info.items() if k != "_assembled_text"}
    col = max((len(k) for k in display), default=10)

    print("\nExtracted vehicle information:")
    print("─" * 60)
    for key, val in display.items():
        print(f"  {key:<{col}} : {val}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out.resolve()}")


if __name__ == "__main__":
    main()
