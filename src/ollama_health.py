"""Check whether Ollama is reachable before analysis."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """
    Returns (ok, message). base_url is OpenAI-compatible, e.g. http://localhost:11434/v1
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    tags_url = f"{root}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return False, (
            "Not running or not installed. Install from https://ollama.com, then run "
            "`ollama serve` and `./scripts/setup_ollama.sh`."
            f" ({e.reason})"
        )
    except Exception as e:
        return False, str(e)

    names = [m.get("name", "") for m in data.get("models", [])]
    # match "qwen2.5:7b-instruct" or "qwen2.5:7b-instruct-latest"
    if not any(model.split(":")[0] in n for n in names):
        return False, (
            f"Running, but model `{model}` not pulled. Run: `ollama pull {model}`"
        )
    return True, "Connected"
