from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel

from .schemas import (
    BookAppointmentInput,
    CheckInsuranceEligibilityInput,
    FindAvailableSlotsInput,
    SearchPatientInput,
)
from .sandbox.api import SandboxAPI


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class ToolSpec(Generic[InputT, OutputT]):
    name: str
    description: str
    input_model: type[InputT]
    handler: Callable[[SandboxAPI, InputT], OutputT]

    def json_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


def build_tool_registry() -> dict[str, ToolSpec[Any, Any]]:
    def _search(api: SandboxAPI, args: SearchPatientInput):
        return api.search_patient(args.name_query)

    def _elig(api: SandboxAPI, args: CheckInsuranceEligibilityInput):
        return api.check_insurance_eligibility(args.patient_id, args.as_of)

    def _slots(api: SandboxAPI, args: FindAvailableSlotsInput):
        return api.find_available_slots(args.specialty, args.start_date, args.end_date)

    def _book(api: SandboxAPI, args: BookAppointmentInput):
        patient = api.get_patient_by_id(args.patient_id)
        if patient is None:
            raise ValueError(f"Unknown patient_id: {args.patient_id}")
        return api.book_appointment(patient, args.slot_id, args.reason, dry_run=args.dry_run)

    tools: list[ToolSpec[Any, Any]] = [
        ToolSpec(
            name="search_patient",
            description="Lookup patients by name substring; returns candidate Patient resources.",
            input_model=SearchPatientInput,
            handler=_search,
        ),
        ToolSpec(
            name="check_insurance_eligibility",
            description="Check a patient's insurance eligibility as-of a date; returns CoverageEligibilityResponse.",
            input_model=CheckInsuranceEligibilityInput,
            handler=_elig,
        ),
        ToolSpec(
            name="find_available_slots",
            description="Find available appointment slots for a specialty within a date range.",
            input_model=FindAvailableSlotsInput,
            handler=_slots,
        ),
        ToolSpec(
            name="book_appointment",
            description="Book an appointment for a patient and slot. Supports dry_run.",
            input_model=BookAppointmentInput,
            handler=_book,
        ),
    ]

    return {t.name: t for t in tools}
