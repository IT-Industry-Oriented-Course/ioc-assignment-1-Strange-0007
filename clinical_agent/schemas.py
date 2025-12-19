from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class FHIRResource(BaseModel):
    model_config = {"extra": "forbid"}

    resourceType: str


class Patient(FHIRResource):
    resourceType: Literal["Patient"] = "Patient"
    id: str
    name: str
    dob: Optional[date] = None
    phone: Optional[str] = None


class InsuranceEligibility(FHIRResource):
    resourceType: Literal["CoverageEligibilityResponse"] = "CoverageEligibilityResponse"
    id: str
    patient_id: str
    as_of: date
    payer: str
    member_id: str
    status: Literal["active", "inactive", "unknown"]


class Slot(FHIRResource):
    resourceType: Literal["Slot"] = "Slot"
    id: str
    specialty: str
    provider_id: str
    provider_name: str
    location: str
    start: datetime
    end: datetime
    available: bool


class Appointment(FHIRResource):
    resourceType: Literal["Appointment"] = "Appointment"
    id: str
    status: Literal["booked", "cancelled"]
    patient_id: str
    patient_name: str
    provider_id: str
    provider_name: str
    slot_id: str
    specialty: str
    start: datetime
    end: datetime
    reason: str
    created_at: datetime


# Tool inputs
class SearchPatientInput(BaseModel):
    model_config = {"extra": "forbid"}
    name_query: str = Field(..., min_length=2)


class CheckInsuranceEligibilityInput(BaseModel):
    model_config = {"extra": "forbid"}
    patient_id: str = Field(..., min_length=1)
    as_of: date


class FindAvailableSlotsInput(BaseModel):
    model_config = {"extra": "forbid"}
    specialty: str = Field(..., min_length=2)
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _validate_range(self) -> "FindAvailableSlotsInput":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self


class BookAppointmentInput(BaseModel):
    model_config = {"extra": "forbid"}
    patient_id: str = Field(..., min_length=1)
    slot_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=2)
    dry_run: bool = False


class ToolCall(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    arguments: dict[str, Any]


class AgentAction(BaseModel):
    """Single-step decision from the LLM."""

    model_config = {"extra": "forbid"}

    type: Literal["tool_call", "stop", "refusal"]
    tool_call: Optional[ToolCall] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "AgentAction":
        if self.type == "tool_call" and self.tool_call is None:
            raise ValueError("tool_call required when type=tool_call")
        if self.type in {"stop", "refusal"} and not self.reason:
            raise ValueError("reason required when type=stop/refusal")
        return self


class AgentResponse(BaseModel):
    model_config = {"extra": "forbid"}

    status: Literal["ok", "refused", "error"]
    session_id: str
    dry_run: bool
    request: str
    refusal_reason: Optional[str] = None

    patient: Optional[Patient] = None
    insurance_eligibility: Optional[InsuranceEligibility] = None
    appointment: Optional[Appointment] = None

    tool_trace: list[dict[str, Any]] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """Single-shot plan produced by the LLM.

    The agent executes tool_calls deterministically (after validation).
    """

    model_config = {"extra": "forbid"}

    type: Literal["plan", "refusal"]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate_plan(self) -> "AgentPlan":
        if self.type == "plan" and not self.tool_calls:
            raise ValueError("tool_calls required when type=plan")
        if self.type == "refusal" and not self.reason:
            raise ValueError("reason required when type=refusal")
        return self
