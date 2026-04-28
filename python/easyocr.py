"""
EasyOCR ONNX inference pipeline.

Models used:
  craft_mlt_25k_jpqd.onnx  — CRAFT text detection
  english_g2_jpqd.onnx      — CRNN English text recognition (CTC)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


class EasyOcr:
    """End-to-end EasyOCR pipeline backed by ONNX Runtime.

    Uses CRAFT for multi-lingual text detection and a CRNN model for
    English text recognition via CTC decoding.
    """

    # CRAFT fixed input dimensions (per model spec)
    _DET_H: int = 640
    _DET_W: int = 640

    # Recognition fixed input dimensions (per model spec)
    _REC_H: int = 32
    _REC_W: int = 100

    # Character set for english_g2 — index 0 is CTC blank, indices 1..94
    # match the 94-char set declared in english_g2_jpqd.yaml.
    _CHARSET: str = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    )

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        models_dir: str | Path,
        *,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        min_area: int = 10,
        use_gpu: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        models_dir      : Directory that contains the ONNX model files.
        text_threshold  : Minimum peak text-score a connected component must
                          have to be kept as a text region.
        low_text        : Binary-map threshold applied to the combined
                          (text + affinity) score map.
        link_threshold  : Minimum affinity score used when combining maps.
        min_area        : Minimum pixel area (in score-map space) for a
                          connected component to be considered.
        use_gpu         : Use CUDAExecutionProvider when True.
        """
        models_dir = Path(models_dir)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def _load(name: str) -> ort.InferenceSession:
            path = models_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            return ort.InferenceSession(str(path), sess_opts, providers=providers)

        self.det_session = _load("craft_mlt_25k_jpqd.onnx")
        self.rec_session = _load("english_g2_jpqd.onnx")

        self._det_input_name = self.det_session.get_inputs()[0].name
        self._rec_input_name = self.rec_session.get_inputs()[0].name

        # Character list: index 0 = CTC blank (""), indices 1..N = characters
        self._characters: list[str] = [""] + list(self._CHARSET)

        # Detection hyper-parameters
        self.text_threshold = text_threshold
        self.low_text       = low_text
        self.link_threshold = link_threshold
        self.min_area       = min_area

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def ocr(self, img: np.ndarray) -> list[dict]:
        """Run end-to-end OCR on a BGR uint8 image.

        Returns a list of dicts, each containing:
          - ``"box"``  : ``np.ndarray`` shape [4, 2] int32 — polygon corners
                         in original image coordinates (top-left, top-right,
                         bottom-right, bottom-left order).
          - ``"text"`` : str — recognised text for that region.
          - ``"conf"`` : float — mean per-character recognition confidence.
        """
        orig_h, orig_w = img.shape[:2]

        # 1. Run CRAFT detector
        score_map = self._detect(img)          # [2, H', W']

        # 2. Convert score maps → bounding boxes in original image coords
        boxes = self._craft_postprocess(score_map, orig_h, orig_w)

        # 3. Recognise each detected region
        results: list[dict] = []
        for box in boxes:
            crop = self._crop_region(img, box)
            if crop is None or crop.size == 0:
                continue
            text, conf = self._recognise(crop)
            if text:
                results.append({"box": box, "text": text, "conf": conf})

        return results

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------

    def _detect(self, img: np.ndarray) -> np.ndarray:
        """Preprocess, run CRAFT, return [2, H', W'] score maps."""
        resized = cv2.resize(img, (self._DET_W, self._DET_H))
        # BGR → RGB, scale to [0, 1], HWC → NCHW
        x = resized[:, :, ::-1].astype(np.float32) / 255.0
        x = np.expand_dims(x.transpose(2, 0, 1), 0)  # [1, 3, H, W]

        out = self.det_session.run(None, {self._det_input_name: x})
        # Model output: [1, 2, H', W']  (region score, affinity score)
        return out[0][0]  # [2, H', W']

    def _craft_postprocess(
        self,
        score_map: np.ndarray,
        src_h: int,
        src_w: int,
    ) -> list[np.ndarray]:
        """Convert CRAFT score maps to bounding-box polygons.

        Parameters
        ----------
        score_map : [2, H', W'] — channel 0 = region score,
                                  channel 1 = affinity score.
        src_h, src_w : Dimensions of the original input image.

        Returns
        -------
        List of [4, 2] int32 arrays, one polygon per detected text region.
        """
        text_score = score_map[0]  # [H', W']
        link_score = score_map[1]  # [H', W']
        score_h, score_w = text_score.shape

        # Combine region and affinity scores, clamp to [0, 1]
        combined = np.clip(text_score + link_score, 0.0, 1.0)

        # Binarise with low_text threshold
        _, binary = cv2.threshold(
            (combined * 255).astype(np.uint8),
            int(self.low_text * 255),
            255,
            cv2.THRESH_BINARY,
        )

        # Connected-component analysis
        n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=4
        )

        # Scale factors from score-map space → original image space
        scale_x = src_w / score_w
        scale_y = src_h / score_h

        boxes: list[np.ndarray] = []
        for k in range(1, n_labels):
            if stats[k, cv2.CC_STAT_AREA] < self.min_area:
                continue

            seg_mask = label_map == k
            # Discard regions whose peak text score is too low
            if text_score[seg_mask].max() < self.text_threshold:
                continue

            # Fit minimum-area rotated rectangle to component pixels
            ys, xs = np.where(seg_mask)
            pts    = np.stack([xs, ys], axis=1).astype(np.float32)
            rect   = cv2.minAreaRect(pts)
            box    = cv2.boxPoints(rect)  # [4, 2] float32 in score-map coords

            # Map to original image coordinates
            box[:, 0] = np.clip(box[:, 0] * scale_x, 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1] * scale_y, 0, src_h - 1)

            boxes.append(box.astype(np.int32))

        return boxes

    # -------------------------------------------------------------------------
    # Region crop (perspective transform)
    # -------------------------------------------------------------------------

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as (top-left, top-right, bottom-right, bottom-left)."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s         = pts.sum(axis=1)
        rect[0]   = pts[np.argmin(s)]   # top-left:     smallest x+y
        rect[2]   = pts[np.argmax(s)]   # bottom-right: largest  x+y
        d         = np.diff(pts, axis=1)
        rect[1]   = pts[np.argmin(d)]   # top-right:    smallest y-x
        rect[3]   = pts[np.argmax(d)]   # bottom-left:  largest  y-x
        return rect

    def _crop_region(
        self, img: np.ndarray, box: np.ndarray
    ) -> Optional[np.ndarray]:
        """Perspective-transform the detected quadrilateral to a flat crop."""
        pts  = box.astype(np.float32)
        rect = self._order_points(pts)
        tl, tr, br, bl = rect

        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        max_w = max(int(w_top), int(w_bot), 1)

        h_left  = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        max_h   = max(int(h_left), int(h_right), 1)

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (max_w, max_h))

    # -------------------------------------------------------------------------
    # Recognition
    # -------------------------------------------------------------------------

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop → [1, 1, 32, 100] float32 normalised to [0, 1].

        Converts to grayscale, resizes to the fixed recognition input size
        (height=32, width=100) with aspect-ratio preservation, then pads the
        width with white (1.0) if the crop is narrower than 100 px.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        h, w    = gray.shape
        # Compute target width at fixed height, capped at _REC_W
        tgt_w   = min(int(w * self._REC_H / max(h, 1)), self._REC_W)
        tgt_w   = max(tgt_w, 1)
        resized = cv2.resize(gray, (tgt_w, self._REC_H))

        # Right-pad with white to reach exactly _REC_W columns
        canvas          = np.full((self._REC_H, self._REC_W), 255, dtype=np.uint8)
        canvas[:, :tgt_w] = resized

        x = canvas.astype(np.float32) / 255.0
        return np.expand_dims(np.expand_dims(x, 0), 0)  # [1, 1, 32, 100]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over the last axis."""
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy CTC decode with confidence estimation.

        Parameters
        ----------
        logits : [T, C] float32  (T=25 time-steps, C=95 classes)

        Returns
        -------
        (text, mean_confidence) — blank-collapsed decoded string and the
        mean softmax probability of the non-blank characters.
        """
        probs        = self._softmax(logits)          # [T, C]
        best_indices = probs.argmax(axis=-1)           # [T]
        best_probs   = probs.max(axis=-1)              # [T]

        chars: list[str]  = []
        confs: list[float] = []
        prev = -1
        for idx, prob in zip(best_indices, best_probs):
            if idx != prev:
                if idx != 0:  # 0 is the CTC blank token
                    chars.append(self._characters[int(idx)])
                    confs.append(float(prob))
                prev = idx

        text = "".join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    def _recognise(self, crop: np.ndarray) -> tuple[str, float]:
        """Run the CRNN recogniser on a BGR crop.

        Returns ``(text, confidence)``.
        """
        x   = self._preprocess_crop(crop)
        out = self.rec_session.run(None, {self._rec_input_name: x})
        # Model output: [1, T, C] or [T, C]
        logits = out[0]
        if logits.ndim == 3:
            logits = logits[0]  # [T, C]
        return self._ctc_decode(logits)
