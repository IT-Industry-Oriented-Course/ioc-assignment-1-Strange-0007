from __future__ import annotations

import argparse
import json
import sys
import re

from .agent import ClinicalWorkflowAgent
from .tools import build_tool_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinical_agent", description="Clinical workflow function-calling agent (POC)")
    parser.add_argument("request", nargs="?", help="Natural language request (operational only)")
    parser.add_argument("--dry-run", action="store_true", help="Plan and simulate actions without persisting bookings")
    parser.add_argument("--print-schemas", action="store_true", help="Print tool JSON schemas and exit")

    args = parser.parse_args(argv)

    if args.print_schemas:
        tools = build_tool_registry()
        print(json.dumps([t.json_schema() for t in tools.values()], ensure_ascii=False, indent=2))
        return 0

    if not args.request:
        parser.error("request is required unless --print-schemas is used")

    try:
        agent = ClinicalWorkflowAgent.from_env()
        result = agent.run(args.request, dry_run=args.dry_run)
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2, default=str))
        return 0 if result.status in {"ok", "refused"} else 2
    except Exception as e:
        msg = str(e)
        # Best-effort redaction of query-param keys in messages.
        msg = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1***", msg)
        print(json.dumps({"status": "error", "error": msg}, ensure_ascii=False, indent=2), file=sys.stdout)
        return 1
