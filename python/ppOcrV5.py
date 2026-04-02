"""
PP-OCRv5 ONNX inference pipeline.

Models used:
  PP-OCRv5_server_det_infer.onnx        — DBNet++ text detection
  PP-OCRv5_server_rec_infer.onnx        — CTC text recognition
  PP-OCRv5_server_rec_infer.yml         — character dictionary + config
  PP-LCNet_x1_0_doc_ori_infer.onnx     — document orientation (0/90/180/270°)
  PP-LCNet_x1_0_textline_ori_infer.onnx — text-line orientation (0/180°)
  UVDoc_infer.onnx                       — document dewarping
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import yaml


class PpOcrV5:
    """End-to-end PP-OCRv5 pipeline backed by ONNX Runtime."""

    # ImageNet normalisation used by detection and classifier models
    _DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _DET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Document / text-line orientation angle tables
    _DOC_ORI_ANGLES      = [0, 90, 180, 270]
    _TEXTLINE_ORI_ANGLES = [0, 180]

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        models_dir: str | Path,
        *,
        det_limit_side_len: int   = 960,
        det_thresh:         float = 0.3,
        det_box_thresh:     float = 0.5,
        det_unclip_ratio:   float = 1.6,
        rec_img_h:          int   = 48,
        rec_img_max_w:      int   = 3200,
        use_gpu:            bool  = False,
    ) -> None:
        """
        Parameters
        ----------
        models_dir          : Directory that contains all ONNX / YAML model files.
        det_limit_side_len  : Rescale the longer edge to this before detection.
        det_thresh          : Probability threshold for the DBNet binary map.
        det_box_thresh      : Minimum average probability score to keep a box.
        det_unclip_ratio    : Box expansion ratio (Vatti-style unclip).
        rec_img_h           : Recognition model fixed input height.
        rec_img_max_w       : Recognition model maximum input width.
        use_gpu             : Use CUDA Execution Provider when True.
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

        self.det_session      = _load("PP-OCRv5_server_det_infer.onnx")
        self.rec_session      = _load("PP-OCRv5_server_rec_infer.onnx")
        self.doc_ori_session  = _load("PP-LCNet_x1_0_doc_ori_infer.onnx")
        self.line_ori_session = _load("PP-LCNet_x1_0_textline_ori_infer.onnx")
        self.uvdoc_session    = _load("UVDoc_infer.onnx")

        # Determine number of orientation classes from model output shapes
        doc_n_cls  = self._static_dim(self.doc_ori_session.get_outputs()[0].shape,  1, default=4)
        line_n_cls = self._static_dim(self.line_ori_session.get_outputs()[0].shape, 1, default=2)
        self._doc_angles  = self._DOC_ORI_ANGLES[:doc_n_cls]
        self._line_angles = self._TEXTLINE_ORI_ANGLES[:line_n_cls]

        # Determine classifier input sizes from model input shapes
        doc_in  = self.doc_ori_session.get_inputs()[0].shape   # [1, 3, H, W]
        line_in = self.line_ori_session.get_inputs()[0].shape
        self._doc_h  = self._static_dim(doc_in,  2, default=224)
        self._doc_w  = self._static_dim(doc_in,  3, default=224)
        self._line_h = self._static_dim(line_in, 2, default=48)
        self._line_w = self._static_dim(line_in, 3, default=192)

        # Determine UVDoc input size
        uvdoc_in = self.uvdoc_session.get_inputs()[0].shape
        self._uvdoc_h = self._static_dim(uvdoc_in, 2, default=488)
        self._uvdoc_w = self._static_dim(uvdoc_in, 3, default=488)

        # Load character dictionary
        cfg_path = models_dir / "PP-OCRv5_server_rec_infer.yml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        raw_chars = cfg["PostProcess"]["character_dict"]
        # Index 0 reserved for CTC blank token
        self.characters: list[str] = [""] + [str(c) for c in raw_chars]

        # Detection hyper-params
        self.det_limit_side_len = det_limit_side_len
        self.det_thresh         = det_thresh
        self.det_box_thresh     = det_box_thresh
        self.det_unclip_ratio   = det_unclip_ratio

        # Recognition hyper-params
        self.rec_img_h     = rec_img_h
        self.rec_img_max_w = rec_img_max_w

    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _static_dim(shape: list, idx: int, default: int) -> int:
        """Return shape[idx] if it is an integer, else default."""
        try:
            v = shape[idx]
            return int(v) if isinstance(v, int) and v > 0 else default
        except (IndexError, TypeError):
            return default

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _hwc_bgr_to_nchw_rgb(img: np.ndarray) -> np.ndarray:
        """uint8 HWC BGR → float32 NCHW RGB, scaled to [0, 1]."""
        x = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, /255
        return np.expand_dims(x.transpose(2, 0, 1), 0)    # HWC→NCHW

    def _normalise_imagenet(self, x: np.ndarray) -> np.ndarray:
        """Apply ImageNet mean/std normalisation to NCHW tensor."""
        m = self._DET_MEAN[:, None, None]
        s = self._DET_STD[:, None, None]
        return (x[0] - m) / s

    @staticmethod
    def _rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img
        if angle in (90, -270):
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle in (180, -180):
            return cv2.rotate(img, cv2.ROTATE_180)
        if angle in (270, -90):
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # -------------------------------------------------------------------------
    # Classifier preprocessing (orientation models)
    # -------------------------------------------------------------------------

    def _preprocess_cls(
        self, img: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        resized = cv2.resize(img, (target_w, target_h))
        x = self._hwc_bgr_to_nchw_rgb(resized)                     # [1, 3, H, W]
        x[0] = self._normalise_imagenet(x)
        return x.astype(np.float32)

    # -------------------------------------------------------------------------
    # Document dewarping (UVDoc)
    # -------------------------------------------------------------------------

    def dewarp(self, img: np.ndarray) -> np.ndarray:
        """Unwarp a document image using the UVDoc model."""
        h_orig, w_orig = img.shape[:2]
        inp = cv2.resize(img, (self._uvdoc_w, self._uvdoc_h))
        x = self._hwc_bgr_to_nchw_rgb(inp)
        x[0] = self._normalise_imagenet(x)

        inp_name = self.uvdoc_session.get_inputs()[0].name
        uv = self.uvdoc_session.run(None, {inp_name: x.astype(np.float32)})[0]

        # Normalise output layout to [H_grid, W_grid, 2]
        if uv.ndim == 4 and uv.shape[1] == 2:
            uv = uv[0].transpose(1, 2, 0)          # [1, 2, H, W] → [H, W, 2]
        elif uv.ndim == 4 and uv.shape[-1] == 2:
            uv = uv[0]                              # [1, H, W, 2] → [H, W, 2]

        # Map UV values in [-1, 1] → pixel coordinates in original image
        map_x = ((uv[..., 0] + 1.0) / 2.0 * (w_orig - 1)).astype(np.float32)
        map_y = ((uv[..., 1] + 1.0) / 2.0 * (h_orig - 1)).astype(np.float32)
        map_x = cv2.resize(map_x, (w_orig, h_orig))
        map_y = cv2.resize(map_y, (w_orig, h_orig))
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # -------------------------------------------------------------------------
    # Orientation correction
    # -------------------------------------------------------------------------

    def correct_doc_orientation(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        Classify document orientation and rotate to upright.

        Returns (corrected_image, detected_angle_degrees).
        """
        x = self._preprocess_cls(img, self._doc_h, self._doc_w)
        inp_name = self.doc_ori_session.get_inputs()[0].name
        logits = self.doc_ori_session.run(None, {inp_name: x})[0]   # [1, C]
        idx    = int(np.argmax(logits, axis=1)[0])
        angle  = self._doc_angles[idx]
        return self._rotate_image(img, -angle), angle

    def correct_textline_orientation(
        self, crop: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        Classify text-line orientation (0 ° / 180 °) and rotate if upside-down.

        Returns (corrected_crop, detected_angle_degrees).
        """
        x = self._preprocess_cls(crop, self._line_h, self._line_w)
        inp_name = self.line_ori_session.get_inputs()[0].name
        logits = self.line_ori_session.run(None, {inp_name: x})[0]  # [1, C]
        idx    = int(np.argmax(logits, axis=1)[0])
        angle  = self._line_angles[idx]
        corrected = self._rotate_image(crop, -angle) if angle != 0 else crop
        return corrected, angle

    # -------------------------------------------------------------------------
    # Text detection (DBNet++)
    # -------------------------------------------------------------------------

    def _preprocess_det(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize, pad and normalise an image for the detection model.

        Returns (tensor [1,3,H,W], scale_factor, (resized_h, resized_w)).
        """
        h, w = img.shape[:2]
        scale = min(1.0, self.det_limit_side_len / max(h, w))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(img, (nw, nh))

        # Pad to next multiple of 32
        pad_h = (32 - nh % 32) % 32
        pad_w = (32 - nw % 32) % 32
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

        x = self._hwc_bgr_to_nchw_rgb(padded)
        x[0] = self._normalise_imagenet(x)
        return x.astype(np.float32), scale, (nh, nw)

    def _box_score(self, prob_map: np.ndarray, box: np.ndarray) -> float:
        """Mean probability of the probability map inside the given quadrilateral."""
        h, w   = prob_map.shape
        xmin   = int(np.clip(box[:, 0].min(), 0, w - 1))
        xmax   = int(np.clip(box[:, 0].max(), 0, w - 1)) + 1
        ymin   = int(np.clip(box[:, 1].min(), 0, h - 1))
        ymax   = int(np.clip(box[:, 1].max(), 0, h - 1)) + 1
        mask   = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
        local  = (box - np.array([xmin, ymin], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [local], 1)
        result = cv2.mean(prob_map[ymin:ymax, xmin:xmax], mask)
        return float(result[0])

    def _expand_box(self, box: np.ndarray, ratio: float) -> np.ndarray:
        """Scale a 4-point box outward from its centroid."""
        centroid = box.mean(axis=0)
        return (box - centroid) * ratio + centroid

    def _postprocess_det(
        self,
        pred:       np.ndarray,
        scale:      float,
        orig_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        """
        Convert the DBNet probability map to a list of 4-point polygon arrays
        in original-image coordinates.
        """
        prob_map = pred[0, 0]                                        # [H, W]
        binary   = (prob_map > self.det_thresh).astype(np.uint8)

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated  = cv2.dilate(binary, kernel)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        orig_h, orig_w = orig_shape
        boxes: list[np.ndarray] = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 16:
                continue

            rect = cv2.minAreaRect(cnt)
            box  = cv2.boxPoints(rect)          # float32 [4, 2]

            if self._box_score(prob_map, box) < self.det_box_thresh:
                continue

            expanded = self._expand_box(box, self.det_unclip_ratio)
            expanded[:, 0] = np.clip(expanded[:, 0] / scale, 0, orig_w - 1)
            expanded[:, 1] = np.clip(expanded[:, 1] / scale, 0, orig_h - 1)
            boxes.append(expanded.astype(np.int32))

        # Sort top-to-bottom, then left-to-right
        boxes.sort(key=lambda b: (b[:, 1].min(), b[:, 0].min()))
        return boxes

    def detect(self, img: np.ndarray) -> list[np.ndarray]:
        """
        Detect text regions in *img*.

        Returns a list of 4-point polygon arrays [[x,y]×4] in original
        image coordinates, sorted top-to-bottom / left-to-right.
        """
        orig_h, orig_w = img.shape[:2]
        x, scale, _    = self._preprocess_det(img)
        inp_name        = self.det_session.get_inputs()[0].name
        pred            = self.det_session.run(None, {inp_name: x})[0]  # [1,1,H,W]
        return self._postprocess_det(pred, scale, (orig_h, orig_w))

    # -------------------------------------------------------------------------
    # Text recognition (CTC)
    # -------------------------------------------------------------------------

    def _preprocess_rec(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize a text-region crop to a fixed height while preserving aspect
        ratio (capped at rec_img_max_w), then apply (x − 0.5) / 0.5 normalisation.
        """
        h, w     = crop.shape[:2]
        target_h = self.rec_img_h
        target_w = max(1, int(w * target_h / max(h, 1)))
        target_w = min(target_w, self.rec_img_max_w)
        target_w = max(target_w, (target_w // 4) * 4 or 4)   # align to 4

        resized  = cv2.resize(crop, (target_w, target_h))
        x        = self._hwc_bgr_to_nchw_rgb(resized)         # [1, 3, H, W], [0,1]
        x        = (x - 0.5) / 0.5                             # normalise to [-1, 1]
        return x.astype(np.float32)

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        """
        Greedy CTC decode for a single sequence.

        Parameters
        ----------
        logits : float32 array of shape [seq_len, num_classes].

        Returns
        -------
        (text, mean_confidence)
        """
        probs   = self._softmax(logits)          # [seq_len, num_classes]
        indices = probs.argmax(axis=-1)          # [seq_len]
        max_p   = probs.max(axis=-1)             # [seq_len]

        n_chars = len(self.characters)
        chars:  list[str]   = []
        scores: list[float] = []
        prev = -1
        for idx, p in zip(indices.tolist(), max_p.tolist()):
            if idx != prev:
                if idx != 0 and idx < n_chars:           # 0 = CTC blank; skip OOV tokens
                    chars.append(self.characters[idx])
                    scores.append(p)
                prev = idx

        text       = "".join(chars)
        confidence = float(np.mean(scores)) if scores else 0.0
        return text, confidence

    def recognize(
        self, crops: list[np.ndarray]
    ) -> list[tuple[str, float]]:
        """
        Recognise text in a list of BGR crop images.

        Returns a list of (text, confidence) tuples in the same order.
        """
        inp_name = self.rec_session.get_inputs()[0].name
        results: list[tuple[str, float]] = []
        for crop in crops:
            x    = self._preprocess_rec(crop)
            out  = self.rec_session.run(None, {inp_name: x})[0]  # [1, T, C]
            text, conf = self._ctc_decode(out[0])
            results.append((text, conf))
        return results

    # -------------------------------------------------------------------------
    # Text-region cropping helper
    # -------------------------------------------------------------------------

    @staticmethod
    def crop_text_region(img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Perspective-warp a rotated bounding box from *img* into a straight
        horizontal rectangle.

        Parameters
        ----------
        img : BGR image.
        box : int32 array of shape [4, 2] — the four corner points in any order.

        Returns
        -------
        Warped crop as a BGR image.
        """
        pts = box.astype(np.float32)
        sums  = pts.sum(axis=1)           # x + y
        diffs = pts[:, 0] - pts[:, 1]    # x − y

        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts[np.argmin(sums)]   # top-left     (min x+y)
        rect[1] = pts[np.argmax(diffs)]  # top-right    (max x−y)
        rect[2] = pts[np.argmax(sums)]   # bottom-right (max x+y)
        rect[3] = pts[np.argmin(diffs)]  # bottom-left  (min x−y)

        w = max(
            int(np.linalg.norm(rect[1] - rect[0])),
            int(np.linalg.norm(rect[2] - rect[3])),
            1,
        )
        h = max(
            int(np.linalg.norm(rect[3] - rect[0])),
            int(np.linalg.norm(rect[2] - rect[1])),
            1,
        )

        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (w, h))

    # -------------------------------------------------------------------------
    # Full OCR pipeline
    # -------------------------------------------------------------------------

    def ocr(
        self,
        img: np.ndarray,
        *,
        dewarp:          bool = False,
        correct_doc_ori: bool = False,
        correct_line_ori: bool = False,
    ) -> list[dict]:
        """
        Run the full OCR pipeline on *img*.

        Optional pre-processing (applied in order):
          1. dewarp          — document dewarping with UVDoc.
          2. correct_doc_ori — document orientation correction (0/90/180/270°).

        Then:
          3. Text region detection.
          4. Perspective-crop each region.
          5. (optional) correct_line_ori — text-line orientation (0/180°).
          6. Text recognition.

        Returns
        -------
        List of dicts, one per detected text region::

            {
                "box":        [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],
                "text":       "recognised string",
                "confidence": 0.0–1.0,
            }
        """
        if dewarp:
            img = self.dewarp(img)

        if correct_doc_ori:
            img, _ = self.correct_doc_orientation(img)

        boxes = self.detect(img)
        crops = [self.crop_text_region(img, b) for b in boxes]

        if correct_line_ori:
            crops = [self.correct_textline_orientation(c)[0] for c in crops]

        texts = self.recognize(crops)

        return [
            {
                "box":        box.tolist(),
                "text":       text,
                "confidence": round(conf, 4),
            }
            for box, (text, conf) in zip(boxes, texts)
        ]
