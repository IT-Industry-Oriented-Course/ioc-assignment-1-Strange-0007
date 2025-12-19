from __future__ import annotations

from dotenv import load_dotenv
import os
import requests


def main() -> int:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    key = os.getenv("GEMINI_API_KEY", "").strip()

    print("gemini_key_set:", bool(key))
    if not key:
        print("Missing GEMINI_API_KEY in .env")
        return 2

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = requests.get(url, params={"key": key}, timeout=60)
    if resp.status_code != 200:
        body = (resp.text or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + "…"
        print(f"HTTP {resp.status_code}: {body}")
        return 1

    data = resp.json()
    models = data.get("models", []) if isinstance(data, dict) else []

    # Print a compact list: name + supportedGenerationMethods
    for m in models:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        methods = m.get("supportedGenerationMethods")
        if isinstance(methods, list):
            methods_str = ",".join(str(x) for x in methods)
        else:
            methods_str = ""
        if name:
            print(f"{name}  methods=[{methods_str}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
