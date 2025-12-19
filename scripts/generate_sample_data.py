from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone


def _write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    data_dir = os.getenv("SANDBOX_DATA_DIR", os.path.join("clinical_agent", "sandbox", "data"))
    os.makedirs(data_dir, exist_ok=True)

    patients = [
        {"patient_id": "pat-001", "name": "Ravi Kumar", "dob": "1987-06-12", "phone": "+91-99999-00001"},
        {"patient_id": "pat-002", "name": "Ananya Sharma", "dob": "1992-03-08", "phone": "+91-99999-00002"},
        {"patient_id": "pat-003", "name": "John Doe", "dob": "1979-11-20", "phone": "+1-555-0100"},
    ]
    _write_csv(os.path.join(data_dir, "patients.csv"), patients, ["patient_id", "name", "dob", "phone"])

    insurance = [
        {"patient_id": "pat-001", "payer": "ACME Health", "member_id": "ACME-RA-1001", "status": "active"},
        {"patient_id": "pat-002", "payer": "ACME Health", "member_id": "ACME-AN-1002", "status": "active"},
        {"patient_id": "pat-003", "payer": "BestCare", "member_id": "BC-JD-2201", "status": "active"},
    ]
    _write_csv(os.path.join(data_dir, "insurance.csv"), insurance, ["patient_id", "payer", "member_id", "status"])

    providers = [
        {"provider_id": "prov-100", "name": "Dr. Meera Iyer", "specialty": "cardiology"},
        {"provider_id": "prov-200", "name": "Dr. Daniel Kim", "specialty": "general"},
    ]
    _write_csv(os.path.join(data_dir, "providers.csv"), providers, ["provider_id", "name", "specialty"])

    now = datetime.now(timezone.utc)
    # Slots are generated relative to current UTC time so prompts like "tomorrow" and "next week" work.
    slots: list[dict[str, str]] = []
    slot_id = 1
    # Cardiology: next 14 days, multiple morning options.
    for d in range(1, 15):
        day = (now + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        for hour in (9, 10, 11):
            start = day.replace(hour=hour)
            end = start + timedelta(minutes=30)
            slots.append(
                {
                    "slot_id": f"slot-{slot_id:03d}",
                    "provider_id": "prov-100",
                    "specialty": "cardiology",
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "location": "Clinic A",
                    "available": "true",
                }
            )
            slot_id += 1

    # General: next 7 days.
    for d in range(1, 8):
        day = (now + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        for hour in (10, 14):
            start = day.replace(hour=hour)
            end = start + timedelta(minutes=20)
            slots.append(
                {
                    "slot_id": f"slot-{slot_id:03d}",
                    "provider_id": "prov-200",
                    "specialty": "general",
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "location": "Clinic B",
                    "available": "true",
                }
            )
            slot_id += 1

    # Seed one historical slot + appointment with all columns populated.
    past_day = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    past_start = past_day.replace(hour=10)
    past_end = past_start + timedelta(minutes=20)
    seed_slot_id = "slot-900"
    slots.append(
        {
            "slot_id": seed_slot_id,
            "provider_id": "prov-200",
            "specialty": "general",
            "start_time": past_start.isoformat(),
            "end_time": past_end.isoformat(),
            "location": "Clinic B",
            "available": "false",
        }
    )

    _write_csv(
        os.path.join(data_dir, "slots.csv"),
        slots,
        ["slot_id", "provider_id", "specialty", "start_time", "end_time", "location", "available"],
    )

    created_at = datetime.now(timezone.utc).isoformat()
    _write_csv(
        os.path.join(data_dir, "appointments.csv"),
        [
            {
                "appointment_id": "appt-seed-001",
                "patient_id": "pat-003",
                "provider_id": "prov-200",
                "slot_id": seed_slot_id,
                "specialty": "general",
                "start_time": past_start.isoformat(),
                "end_time": past_end.isoformat(),
                "reason": "annual checkup",
                "status": "booked",
                "created_at": created_at,
            }
        ],
        [
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

    print(f"Wrote sandbox CSVs to: {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
