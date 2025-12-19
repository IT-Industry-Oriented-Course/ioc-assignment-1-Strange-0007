[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HkHTIwfX)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=22090394&assignment_repo_type=AssignmentRepo)


# Clinical Workflow Function-Calling Agent (POC)

Implements the assignment POC: a function-calling LLM agent that **orchestrates** appointment/care coordination workflows using **validated tool calls**, **CSV-backed sandbox APIs**, **dry-run mode**, and **audit logging**.

## What it does

Given a request like:

> "Schedule a cardiology follow-up for patient Ravi Kumar next week and check insurance eligibility"

The agent:
- Plans tool calls via **one Gemini text-generation API**
- Validates tool call inputs (Pydantic / JSON Schema)
- Executes sandbox APIs backed by generated CSV data
- Returns a **structured appointment object** (not prose)
- Logs every action to a JSONL audit log

## Quick start (Windows cmd)

1) Create venv + install:

```bat
cd /d "<root>\IOC"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Configure env:

```bat
copy .env.example .env
```
Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`.

3) Generate sandbox CSVs:

```bat
python -m scripts.generate_sample_data  
```

Sample data includes patients: `Ravi Kumar`, `Ananya Sharma`, `John Doe`. It also generates appointment slots for the next few days.

4) Run agent (dry-run or whatever):

```bat
python -m clinical_agent "Schedule a cardiology follow-up for patient Ravi Kumar next week and check insurance eligibility" --dry-run
```

5) Run agent (real booking):

```bat
python -m clinical_agent "Book a cardiology appointment for patient Ravi Kumar tomorrow morning"
```

More prompts to try:

```bat
python -m clinical_agent "Book a general appointment for patient John Doe in 2 days"
python -m clinical_agent "Check insurance eligibility for patient Ananya Sharma as of today" --dry-run

python -m clinical_agent "Book cardiology for Ravi Kumar tomorrow morning"
```

## Notes
- Audit log is written to `audit_logs\audit.jsonl`.
