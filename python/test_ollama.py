"""
test_ollama.py — Simple connectivity test for the Ollama server.

Checks:
  1. Server reachable  (/api/tags)
  2. Target model is available
  3. Chat endpoint responds  (/api/chat)

Reads OLLAMA_URL and OLLAMA_MODEL from the .env file or environment variables.

Usage:
  python test_ollama.py
  python test_ollama.py --url http://10.68.129.51:8088 --model qwen3:9b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_settings(url: str | None, model: str | None) -> tuple[str, str]:
    _env = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env, override=False)
    resolved_url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
    resolved_model = model or os.environ.get("OLLAMA_MODEL", "llama3")
    return resolved_url, resolved_model


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_server_reachable(base_url: str, timeout: int = 10) -> bool:
    print(f"\n[1] Checking server reachability: {base_url}/api/tags")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        _ok(f"Server responded with HTTP {resp.status_code}")
        return True
    except requests.exceptions.ConnectionError as exc:
        _fail(f"Connection refused — {exc}")
    except requests.exceptions.Timeout:
        _fail(f"Request timed out after {timeout}s")
    except requests.exceptions.HTTPError as exc:
        _fail(f"HTTP error — {exc}")
    return False


def test_model_available(base_url: str, model: str, timeout: int = 10) -> bool:
    print(f"\n[2] Checking model availability: {model!r}")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        if any(m == model or m.startswith(model.split(":")[0]) for m in available):
            _ok(f"Model found. Available models: {available}")
            return True
        else:
            _fail(f"Model {model!r} not found. Available: {available}")
    except Exception as exc:
        _fail(f"Could not retrieve model list — {exc}")
    return False


def test_chat_endpoint(base_url: str, model: str, timeout: int = 30) -> bool:
    print(f"\n[3] Testing chat endpoint with model {model!r}")
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
        _ok(f"Chat response received: {content!r}")
        return True
    except requests.exceptions.Timeout:
        _fail(f"Chat request timed out after {timeout}s")
    except requests.exceptions.HTTPError as exc:
        _fail(f"HTTP error — {exc}\n       {resp.text[:300]}")
    except Exception as exc:
        _fail(f"Unexpected error — {exc}")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test Ollama server connectivity.")
    parser.add_argument("--url", default=None, help="Ollama base URL (overrides .env / OLLAMA_URL).")
    parser.add_argument("--model", default=None, help="Model tag (overrides .env / OLLAMA_MODEL).")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds.")
    args = parser.parse_args()

    base_url, model = _load_settings(args.url, args.model)

    print("=" * 55)
    print("  Ollama Connection Test")
    print("=" * 55)
    print(f"  URL   : {base_url}")
    print(f"  Model : {model}")
    print("=" * 55)

    results = [
        test_server_reachable(base_url, timeout=args.timeout),
        test_model_available(base_url, model, timeout=args.timeout),
        test_chat_endpoint(base_url, model, timeout=args.timeout),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 55}")
    print(f"  Result: {passed}/{total} tests passed")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
