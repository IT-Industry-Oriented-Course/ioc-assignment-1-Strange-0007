from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class GeminiTextGen:
    """Minimal Gemini API client using the Generative Language API.

    Uses generateContent and returns the concatenated text parts.
    """

    api_key: str
    model: str
    timeout_s: float = 60.0

    def generate(self, prompt: str, *, max_output_tokens: int = 768) -> str:
        # v1beta endpoint is still widely supported for API-key access.
        # Accept both "gemini-..." and "models/gemini-..." forms.
        model = self.model.strip()
        if model.startswith("models/"):
            model = model[len("models/") :]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_output_tokens,
            },
        }

        resp = requests.post(url, params=params, headers=headers, json=payload, timeout=self.timeout_s)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Never leak the API key (which is passed via query param).
            snippet = (resp.text or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            raise RuntimeError(
                f"Gemini API error (HTTP {resp.status_code}). "
                f"Check GEMINI_MODEL and API access. Response: {snippet}"
            ) from e

        data = resp.json()

        # Typical shape:
        # {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list) and parts:
                texts = []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        texts.append(p["text"])
                if texts:
                    return "".join(texts)

        return json.dumps(data)


def extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of the first JSON object from model output."""
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None
