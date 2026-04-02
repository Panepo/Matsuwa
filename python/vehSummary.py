"""
vehSummary.py — Vehicle information extractor via Ollama LLM.

Takes OCR results from PpOcrV5 (list of dicts with 'text', 'box',
'confidence' keys), reassembles the possibly line-broken text into
reading order, then queries an Ollama-hosted LLM to extract structured
vehicle information as a JSON dict.

Configuration is read from environment variables (or a .env file):
  OLLAMA_URL   — e.g. http://10.68.129.51:8088
  OLLAMA_MODEL — e.g. qwen3.5:9b
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv


class VehSummary:
    """Extract structured vehicle information from PP-OCRv5 results using an LLM."""

    _SYSTEM_PROMPT = (
        "You are a vehicle document analyst. "
        "Given raw OCR text extracted from a vehicle-related document or image, "
        "extract all vehicle information you can find and return it as a valid JSON object. "
        "Use the following keys (omit any key not present in the text): "
        "make, model, year, license_plate, color, vin"
        "Return ONLY the JSON object — no explanation, no markdown fences."
    )

    def __init__(
        self,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        *,
        env_path: str | Path | None = None,
        line_merge_ratio: float = 0.5,
        request_timeout: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        ollama_url        : Base URL of the Ollama server.  Overrides .env / env var.
        ollama_model      : Model tag to use (e.g. ``qwen3.5:9b``).  Overrides .env.
        env_path          : Path to a .env file to load.  Defaults to the .env file
                            that lives next to this source file.
        line_merge_ratio  : Fraction of average line-height used as the vertical
                            grouping tolerance when reassembling broken OCR tokens.
        request_timeout   : HTTP request timeout in seconds for the Ollama call.
                            Defaults to ``None`` (wait indefinitely).  Set to a
                            positive integer if you want a hard upper bound.
        """
        _env = Path(env_path) if env_path else Path(__file__).parent / ".env"
        load_dotenv(dotenv_path=_env, override=False)

        self.ollama_url = (
            ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL", "llama3")
        self.line_merge_ratio = line_merge_ratio
        self.request_timeout = request_timeout
        self.stream_output = False  # set True only if the server supports SSE streaming

    # ------------------------------------------------------------------
    # Text reassembly
    # ------------------------------------------------------------------

    @staticmethod
    def _box_metrics(box: list) -> tuple[float, float, float]:
        """Return (x_center, y_center, height) for a 4-point bounding box."""
        pts = np.array(box, dtype=np.float32)
        x_center = pts[:, 0].mean()
        y_center = pts[:, 1].mean()
        height = pts[:, 1].max() - pts[:, 1].min()
        return float(x_center), float(y_center), float(height)

    def assemble_text(self, ocr_results: list[dict]) -> str:
        """
        Reassemble OCR tokens into a reading-order block of text.

        Tokens whose vertical centres are within ``line_merge_ratio × avg_height``
        of each other are placed on the same line (sorted left-to-right).
        Distinct rows are separated by newlines.

        Parameters
        ----------
        ocr_results : List of dicts as returned by ``PpOcrV5.ocr()``.

        Returns
        -------
        str
            Reassembled text, newline-separated by visual line.
        """
        if not ocr_results:
            return ""

        # Annotate each token with (x_center, y_center, height, text)
        annotated: list[tuple[float, float, float, str]] = []
        for r in ocr_results:
            text = r.get("text", "").strip()
            if not text:
                continue
            xc, yc, h = self._box_metrics(r["box"])
            annotated.append((xc, yc, h, text))

        if not annotated:
            return ""

        # Primary sort: top-to-bottom, secondary: left-to-right
        annotated.sort(key=lambda t: (t[1], t[0]))

        avg_h = float(np.mean([t[2] for t in annotated]))
        tolerance = avg_h * self.line_merge_ratio

        # Group tokens into visual lines
        lines: list[list[tuple[float, float, float, str]]] = []
        current_line: list[tuple[float, float, float, str]] = [annotated[0]]
        current_y = annotated[0][1]

        for token in annotated[1:]:
            if abs(token[1] - current_y) <= tolerance:
                current_line.append(token)
            else:
                lines.append(sorted(current_line, key=lambda t: t[0]))
                current_line = [token]
                current_y = token[1]
        lines.append(sorted(current_line, key=lambda t: t[0]))

        return "\n".join(" ".join(t[3] for t in line) for line in lines)

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _query_llm(self, text: str) -> str:
        """
        Send *text* to the Ollama ``/api/chat`` endpoint and return the
        assistant's raw response string.

        When ``self.stream_output`` is True (the default), tokens are printed
        to stdout as they arrive so the caller can see progress in real time.
        """
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "stream": self.stream_output,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR text:\n\n{text}"},
            ],
        }
        try:
            resp = requests.post(
                url, json=payload,
                timeout=self.request_timeout,
                stream=self.stream_output,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.ollama_url}: {exc}"
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            limit = f"{self.request_timeout}s" if self.request_timeout else "unlimited"
            raise TimeoutError(
                f"Ollama read timed out (timeout={limit}). "
                "The model may need more time — pass a larger --timeout value or set request_timeout=None."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama HTTP error: {exc}\n{resp.text}"
            ) from exc

        if not self.stream_output:
            data = resp.json()
            return data["message"]["content"]

        # Streaming: accumulate tokens and echo them to stdout
        import sys
        chunks: list[str] = []
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                chunks.append(token)
                print(token, end="", flush=True)
            if chunk.get("done"):
                break
        print()  # newline after streaming completes
        return "".join(chunks)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """
        Extract and parse the first JSON object found in *raw*.

        Handles cases where the model wraps its output in markdown fences
        or prepends/appends extra prose around the JSON.
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # Fast path: direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Extract the first {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text under a single key
        return {"raw": raw}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _VEHICLE_MARKER = "***VEHICLEINFO***"

    def extract(self, ocr_results: list[dict]) -> dict[str, Any]:
        """
        Full pipeline: reassemble OCR text → query LLM → parse structured JSON.

        If the assembled text contains ``***VEHICLEINFO***``, only the text
        that follows the marker is sent to the LLM, reducing context length.
        If the marker is absent, nothing is sent and an empty dict is returned.

        Parameters
        ----------
        ocr_results : List of dicts as returned by ``PpOcrV5.ocr()``.

        Returns
        -------
        dict
            Extracted vehicle fields.  Always includes the key
            ``"_assembled_text"`` containing the full pre-LLM text block for
            debugging purposes.
        """
        assembled = self.assemble_text(ocr_results)

        marker_idx = assembled.find(self._VEHICLE_MARKER)
        if marker_idx == -1:
            return {"_assembled_text": assembled}

        text_to_send = assembled[marker_idx + len(self._VEHICLE_MARKER):].strip()
        raw_llm = self._query_llm(text_to_send)
        info = self._parse_json(raw_llm)
        info["_assembled_text"] = assembled
        return info
