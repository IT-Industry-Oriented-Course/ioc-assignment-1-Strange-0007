from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from .audit import AuditLogger
from .gemini_client import GeminiTextGen
from .gemini_client import extract_first_json_object as gemini_extract_json
from .sandbox.api import SandboxAPI, SandboxDataStore
from .schemas import AgentPlan, AgentResponse, Appointment, InsuranceEligibility, Patient, ToolCall
from .tools import build_tool_registry


_MEDICAL_ADVICE_RE = re.compile(
    r"\b("
    r"diagnos(?:is|e|ing)?|"
    r"treat(?:ment|ing)?|"
    r"prescrib(?:e|ing|ed)|prescription|rx|"
    r"medicine|medication|drug(?:s)?|pill(?:s)?|antibiotic(?:s)?|"
    r"dose|dosage|side\s*effects?|contraindication(?:s)?|"
    r"what\s+should\s+i\s+do|should\s+i\s+take"
    r")\b",
    re.IGNORECASE,
)


def _is_medical_advice_request(text: str) -> bool:
    return bool(_MEDICAL_ADVICE_RE.search(text or ""))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today_utc() -> date:
    return _utc_now().date()


def _build_system_prompt(tool_schemas: list[dict[str, Any]], *, now_utc: datetime) -> str:
    schema_json = json.dumps(tool_schemas, ensure_ascii=False)
    today = now_utc.date().isoformat()
    now_iso = now_utc.isoformat()
    return (
        "You are a clinical workflow automation agent. You must NOT provide diagnosis or medical advice. "
        "You are ONLY allowed to choose tool calls from the provided tools to coordinate operations. "
        "Never invent patient_id, slot_id, provider_id, or dates; only use IDs from tool results. "
        "Because you are planning before tool execution, you may use placeholders wrapped in angle brackets for IDs that will be obtained from earlier tool results. "
        "Use placeholders that clearly indicate intent, e.g., '<PATIENT_ID_FROM_SEARCH_PATIENT>' and '<SLOT_ID_FROM_FIND_AVAILABLE_SLOTS>'. "
        "If multiple slots are returned, choose the earliest slot (by start time). "
        "If you cannot proceed safely, output a refusal.\n\n"
        f"CURRENT TIME CONTEXT (UTC): today={today}, now={now_iso}. "
        "Interpret relative date phrases like 'today', 'tomorrow', and 'next week' using this UTC context.\n\n"
        "PLANNING RULES:\n"
        "- For booking/scheduling requests, your plan MUST include (in order): search_patient -> find_available_slots -> book_appointment.\n"
        "- If the user also asks to check insurance eligibility, include check_insurance_eligibility after search_patient.\n"
        "- When the user uses relative date phrases, you MUST convert them to explicit ISO dates for tool arguments.\n"
        "- If required details are missing (e.g., no patient name), return a refusal instead of a partial plan.\n\n"
        "OUTPUT FORMAT (STRICT): Return ONLY a single JSON object (no prose).\n"
        "Return either:\n"
        "1) Plan (list of tool calls, in order):\n"
        '{"type":"plan","tool_calls":[{"name":"<tool_name>","arguments":{...}}],"reason":"<short>"}\n'
        "2) Refusal (if medical advice or unsafe/ambiguous):\n"
        '{"type":"refusal","reason":"<why you must refuse or what info is missing>"}\n\n'
        f"AVAILABLE TOOLS (with JSON Schemas):\n{schema_json}\n"
    )


def _build_plan_prompt(system_prompt: str, user_request: str) -> str:
    return (
        system_prompt
        + "\nCURRENT USER REQUEST:\n"
        + user_request
        + "\n\nCREATE A PLAN NOW." 
    )


@dataclass
class AgentConfig:
    llm_provider: str  # "gemini"
    gemini_api_key: str
    gemini_model: str
    audit_log_path: str
    sandbox_data_dir: str


class ClinicalWorkflowAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = build_tool_registry()
        self.tool_schemas = [t.json_schema() for t in self.tools.values()]
        self.system_prompt = _build_system_prompt(self.tool_schemas, now_utc=_utc_now())

        self.api = SandboxAPI(SandboxDataStore(config.sandbox_data_dir))
        if config.llm_provider != "gemini":
            raise RuntimeError("Only LLM_PROVIDER=gemini is supported.")

        self.gemini_llm: Optional[GeminiTextGen] = GeminiTextGen(api_key=config.gemini_api_key, model=config.gemini_model)

    @staticmethod
    def from_env() -> "ClinicalWorkflowAgent":
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        cfg = AgentConfig(
            llm_provider=provider,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "").strip(),
            audit_log_path=os.getenv("AUDIT_LOG_PATH", "audit_logs\\audit.jsonl").strip(),
            sandbox_data_dir=os.getenv("SANDBOX_DATA_DIR", "clinical_agent\\sandbox\\data").strip(),
        )

        if cfg.llm_provider not in {"gemini"}:
            raise RuntimeError("Only LLM_PROVIDER=gemini is supported.")

        if cfg.llm_provider == "gemini" and not cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        if not cfg.gemini_model:
            raise RuntimeError("GEMINI_MODEL is required (use a value from `python -m scripts.gemini_list_models`).")

        return ClinicalWorkflowAgent(cfg)

    def run(self, user_request: str, *, dry_run: bool) -> AgentResponse:
        session_id = uuid.uuid4().hex
        audit = AuditLogger(path=self.config.audit_log_path, session_id=session_id)

        audit.log("request_received", {"request": user_request, "dry_run": dry_run})

        if _is_medical_advice_request(user_request):
            audit.log("refusal", {"reason": "medical_advice"})
            return AgentResponse(
                status="refused",
                session_id=session_id,
                dry_run=dry_run,
                request=user_request,
                refusal_reason="I can’t provide medical advice/diagnosis. Please rephrase as an operational task (scheduling, eligibility checks, follow-ups).",
            )

        trace: list[dict[str, Any]] = []
        resolved_patient: Optional[Patient] = None
        eligibility: Optional[InsuranceEligibility] = None
        appointment: Optional[Appointment] = None

        plan = self._plan_once(user_request)
        audit.log("llm_plan", plan.model_dump())
        if plan.type == "refusal":
            return AgentResponse(
                status="refused",
                session_id=session_id,
                dry_run=dry_run,
                request=user_request,
                refusal_reason=plan.reason,
                tool_trace=trace,
            )

        for tool_call in plan.tool_calls:
            try:
                resolved_patient, eligibility, appointment = self._execute_tool_call(
                    audit,
                    tool_call,
                    dry_run=dry_run,
                    trace=trace,
                    resolved_patient=resolved_patient,
                    eligibility=eligibility,
                    appointment=appointment,
                )
            except (ValueError, ValidationError) as e:
                audit.log(
                    "tool_error",
                    {
                        "error": "tool_execution_failed",
                        "tool": getattr(tool_call, "name", None),
                        "message": str(e),
                    },
                )
                response = AgentResponse(
                    status="refused",
                    session_id=session_id,
                    dry_run=dry_run,
                    request=user_request,
                    refusal_reason=(
                        "Planner produced invalid tool arguments (often placeholders instead of real IDs). "
                        f"Details: {str(e)}"
                    ),
                    patient=resolved_patient,
                    insurance_eligibility=eligibility,
                    appointment=appointment,
                    tool_trace=trace,
                )
                audit.log("final_response", response.model_dump())
                return response
            if appointment is not None:
                break

       
        if appointment is None and eligibility is None:
            slot_searches = [t for t in trace if t.get("tool") == "find_available_slots"]
            if slot_searches:
                last_result = slot_searches[-1].get("result")
                if isinstance(last_result, list) and len(last_result) == 0:
                    response = AgentResponse(
                        status="refused",
                        session_id=session_id,
                        dry_run=dry_run,
                        request=user_request,
                        refusal_reason="No available slots found for the requested criteria; unable to book an appointment.",
                        patient=resolved_patient,
                        insurance_eligibility=None,
                        appointment=None,
                        tool_trace=trace,
                    )
                    audit.log("final_response", response.model_dump())
                    return response

        # If the planner produced only a partial plan (e.g., found the patient but didn't proceed to slots/booking),
        # return a refusal rather than a generic error.
        if appointment is None and eligibility is None:
            refusal_reason = "Unable to complete the workflow safely."

            search_steps = [t for t in trace if t.get("tool") == "search_patient"]
            if search_steps:
                last = search_steps[-1].get("result")
                if isinstance(last, list):
                    if len(last) == 0:
                        refusal_reason = "No matching patient found; please provide the full patient name."
                    elif len(last) > 1:
                        refusal_reason = "Multiple patients matched; please provide the full patient name."
                    else:
                        refusal_reason = "Patient found, but the planner did not complete scheduling. Please retry with a specific date/time (or date range) for the appointment."

            response = AgentResponse(
                status="refused",
                session_id=session_id,
                dry_run=dry_run,
                request=user_request,
                refusal_reason=refusal_reason,
                patient=resolved_patient,
                insurance_eligibility=None,
                appointment=None,
                tool_trace=trace,
            )
            audit.log("final_response", response.model_dump())
            return response

        response = AgentResponse(
            status="ok" if appointment or eligibility else "error",
            session_id=session_id,
            dry_run=dry_run,
            request=user_request,
            refusal_reason=None if (appointment or eligibility or resolved_patient) else "No actionable workflow could be completed.",
            patient=resolved_patient,
            insurance_eligibility=eligibility,
            appointment=appointment,
            tool_trace=trace,
        )
        audit.log("final_response", response.model_dump())
        return response

    def _execute_tool_call(
        self,
        audit: AuditLogger,
        tool_call: ToolCall,
        *,
        dry_run: bool,
        trace: list[dict[str, Any]],
        resolved_patient: Optional[Patient],
        eligibility: Optional[InsuranceEligibility],
        appointment: Optional[Appointment],
    ) -> tuple[Optional[Patient], Optional[InsuranceEligibility], Optional[Appointment]]:
        if tool_call.name not in self.tools:
            audit.log("tool_error", {"error": "unknown_tool", "tool": tool_call.name})
            raise RuntimeError(f"Unknown tool requested: {tool_call.name}")

        spec = self.tools[tool_call.name]
        arguments = dict(tool_call.arguments)
        if tool_call.name == "book_appointment":
            arguments.setdefault("dry_run", dry_run)

        arguments = _resolve_argument_placeholders(
            tool_name=tool_call.name,
            arguments=arguments,
            resolved_patient=resolved_patient,
            trace=trace,
        )

        try:
            args = spec.input_model.model_validate(arguments)
        except ValidationError as e:
            audit.log("tool_error", {"error": "invalid_arguments", "tool": tool_call.name, "details": e.errors()})
            raise

        audit.log("tool_call", {"tool": tool_call.name, "arguments": arguments})
        result = spec.handler(self.api, args)
        audit.log("tool_result", {"tool": tool_call.name, "result": _safe_dump(result)})

        if tool_call.name == "search_patient" and isinstance(result, list) and len(result) == 1:
            resolved_patient = result[0]
        if tool_call.name == "check_insurance_eligibility":
            eligibility = result
        if tool_call.name == "book_appointment":
            appointment = result

        trace.append({"tool": tool_call.name, "arguments": arguments, "result": _safe_dump(result)})
        return resolved_patient, eligibility, appointment

    def _plan_once(self, user_request: str) -> AgentPlan:
        prompt = _build_plan_prompt(self.system_prompt, user_request)

        assert self.gemini_llm is not None
        raw = self.gemini_llm.generate(prompt)
        data = gemini_extract_json(raw)

        if not isinstance(data, dict):
            return AgentPlan(type="refusal", reason="Planner returned non-JSON output; cannot proceed safely.")

        try:
            return AgentPlan.model_validate(data)
        except ValidationError:
            return AgentPlan(type="refusal", reason="Planner output failed schema validation; cannot proceed safely.")

def _safe_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_safe_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    return obj


_PLACEHOLDER_RE = re.compile(r"^<[^>]+>$")


def _looks_like_placeholder(value: Any, *, contains: str) -> bool:
    if not isinstance(value, str):
        return False
    if not _PLACEHOLDER_RE.match(value.strip()):
        return False
    return contains.lower() in value.lower()


def _last_slots_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for step in reversed(trace):
        if step.get("tool") == "find_available_slots":
            result = step.get("result")
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict)]
            return []
    return []


def _pick_earliest_slot_id(slots: list[dict[str, Any]]) -> str:
    if not slots:
        raise ValueError("No available slots to choose from.")

    def _slot_start(s: dict[str, Any]) -> datetime:
        start = s.get("start")
        if isinstance(start, datetime):
            return start
        if isinstance(start, str):
            return datetime.fromisoformat(start)
        raise ValueError("Slot is missing a parseable 'start' field.")

    best = min(slots, key=_slot_start)
    slot_id = best.get("id")
    if not isinstance(slot_id, str) or not slot_id:
        raise ValueError("Slot is missing an 'id' field.")
    return slot_id


def _resolve_argument_placeholders(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    resolved_patient: Optional[Patient],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(arguments)

    if tool_name in {"check_insurance_eligibility", "book_appointment"}:
        patient_id = out.get("patient_id")
        if _looks_like_placeholder(patient_id, contains="patient"):
            if resolved_patient is None:
                raise ValueError("patient_id is a placeholder but no patient was resolved from search_patient.")
            out["patient_id"] = resolved_patient.id

    if tool_name == "book_appointment":
        slot_id = out.get("slot_id")
        if _looks_like_placeholder(slot_id, contains="slot"):
            slots = _last_slots_from_trace(trace)
            out["slot_id"] = _pick_earliest_slot_id(slots)

    return out
