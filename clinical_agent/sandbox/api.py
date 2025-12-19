from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Optional

from dateutil.parser import isoparse

from ..schemas import Appointment, InsuranceEligibility, Patient, Slot


def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class SandboxDataStore:
    data_dir: str

    @property
    def patients_path(self) -> str:
        return os.path.join(self.data_dir, "patients.csv")

    @property
    def insurance_path(self) -> str:
        return os.path.join(self.data_dir, "insurance.csv")

    @property
    def providers_path(self) -> str:
        return os.path.join(self.data_dir, "providers.csv")

    @property
    def slots_path(self) -> str:
        return os.path.join(self.data_dir, "slots.csv")

    @property
    def appointments_path(self) -> str:
        return os.path.join(self.data_dir, "appointments.csv")


class SandboxAPI:
    def __init__(self, store: SandboxDataStore):
        self.store = store

    def get_patient_by_id(self, patient_id: str) -> Optional[Patient]:
        for row in _read_csv(self.store.patients_path):
            if (row.get("patient_id") or "") == patient_id:
                name = (row.get("name") or "").strip()
                dob = row.get("dob") or None
                return Patient(
                    id=row["patient_id"],
                    name=name,
                    dob=date.fromisoformat(dob) if dob else None,
                    phone=(row.get("phone") or None),
                )
        return None

    def search_patient(self, name_query: str) -> list[Patient]:
        q = name_query.strip().lower()
        matches: list[Patient] = []
        for row in _read_csv(self.store.patients_path):
            name = (row.get("name") or "").strip()
            if q in name.lower():
                dob = row.get("dob") or None
                matches.append(
                    Patient(
                        id=row["patient_id"],
                        name=name,
                        dob=date.fromisoformat(dob) if dob else None,
                        phone=(row.get("phone") or None),
                    )
                )
        return matches

    def check_insurance_eligibility(self, patient_id: str, as_of: date) -> InsuranceEligibility:
        rows = _read_csv(self.store.insurance_path)
        row = next((r for r in rows if r.get("patient_id") == patient_id), None)
        if not row:
            return InsuranceEligibility(
                id=f"elig-{patient_id}-{as_of.isoformat()}",
                patient_id=patient_id,
                as_of=as_of,
                payer="unknown",
                member_id="unknown",
                status="unknown",
            )

        status = (row.get("status") or "unknown").strip().lower()
        status = status if status in {"active", "inactive", "unknown"} else "unknown"
        return InsuranceEligibility(
            id=f"elig-{patient_id}-{as_of.isoformat()}",
            patient_id=patient_id,
            as_of=as_of,
            payer=(row.get("payer") or "unknown"),
            member_id=(row.get("member_id") or "unknown"),
            status=status,  # type: ignore[arg-type]
        )

    def find_available_slots(self, specialty: str, start_date: date, end_date: date) -> list[Slot]:
        specialty_l = specialty.strip().lower()
        providers = {r["provider_id"]: r for r in _read_csv(self.store.providers_path)}

        slots: list[Slot] = []
        for row in _read_csv(self.store.slots_path):
            if (row.get("specialty") or "").strip().lower() != specialty_l:
                continue
            if (row.get("available") or "").strip().lower() != "true":
                continue

            start = isoparse(row["start_time"]).astimezone(timezone.utc)
            end = isoparse(row["end_time"]).astimezone(timezone.utc)

            if start.date() < start_date or start.date() > end_date:
                continue

            prov = providers.get(row["provider_id"], {})
            slots.append(
                Slot(
                    id=row["slot_id"],
                    specialty=row["specialty"],
                    provider_id=row["provider_id"],
                    provider_name=(prov.get("name") or "Unknown"),
                    location=(row.get("location") or "Main"),
                    start=start,
                    end=end,
                    available=True,
                )
            )
        return sorted(slots, key=lambda s: s.start)

    def book_appointment(self, patient: Patient, slot_id: str, reason: str, *, dry_run: bool) -> Appointment:
        slots = _read_csv(self.store.slots_path)
        slot_row = next((r for r in slots if r.get("slot_id") == slot_id), None)
        if not slot_row:
            raise ValueError(f"Unknown slot_id: {slot_id}")
        if (slot_row.get("available") or "").strip().lower() != "true":
            raise ValueError(f"Slot not available: {slot_id}")

        providers = {r["provider_id"]: r for r in _read_csv(self.store.providers_path)}
        prov = providers.get(slot_row["provider_id"], {})

        start = isoparse(slot_row["start_time"]).astimezone(timezone.utc)
        end = isoparse(slot_row["end_time"]).astimezone(timezone.utc)
        created_at = datetime.now(timezone.utc)

        appt = Appointment(
            id=f"appt-{uuid.uuid4().hex[:12]}",
            status="booked",
            patient_id=patient.id,
            patient_name=patient.name,
            provider_id=slot_row["provider_id"],
            provider_name=(prov.get("name") or "Unknown"),
            slot_id=slot_row["slot_id"],
            specialty=slot_row["specialty"],
            start=start,
            end=end,
            reason=reason,
            created_at=created_at,
        )

        if dry_run:
            return appt

        # persist: mark slot unavailable and append appointment
        for r in slots:
            if r.get("slot_id") == slot_id:
                r["available"] = "false"
        _write_csv(
            self.store.slots_path,
            slots,
            fieldnames=["slot_id", "provider_id", "specialty", "start_time", "end_time", "location", "available"],
        )

        appts = _read_csv(self.store.appointments_path)
        appts.append(
            {
                "appointment_id": appt.id,
                "patient_id": appt.patient_id,
                "provider_id": appt.provider_id,
                "slot_id": appt.slot_id,
                "specialty": appt.specialty,
                "start_time": appt.start.isoformat(),
                "end_time": appt.end.isoformat(),
                "reason": appt.reason,
                "status": appt.status,
                "created_at": appt.created_at.isoformat(),
            }
        )
        _write_csv(
            self.store.appointments_path,
            appts,
            fieldnames=[
                "appointment_id",
                "patient_id",
                "provider_id",
                "slot_id",
                "specialty",
                "start_time",
                "end_time",
                "reason",
                "status",
                "created_at",
            ],
        )

        return appt
