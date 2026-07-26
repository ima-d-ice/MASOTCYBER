# MAS — Multi-Agent Anomaly Detection for Industrial Control Systems

A multi-agent system for SWaT anomaly detection that combines TranAD-based detectors, a persistence-based escalation gate, and LLM-based investigators/orchestration.

<p align="center">
<img src="https://img.shields.io/badge/PA--F1-0.8541-brightgreen" alt="PA-F1">
<img src="https://img.shields.io/badge/Strict%20F1-0.3750-blue" alt="Strict F1">
<img src="https://img.shields.io/badge/False%20Positives-23-orange" alt="FP">
<img src="https://img.shields.io/badge/LLM%20Calls-4912-purple" alt="LLM Calls">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
</p>

---

## What this project does

This system processes SWaT windows in three stages:

1. Detector stage
   - Uses TranAD-style anomaly detectors for six stages.
   - Produces per-stage scores for each 30-second window.

2. Escalation gate
   - Filters low-confidence or redundant triggers before invoking LLMs.
   - Uses persistence and recheck logic to reduce unnecessary API usage.

3. LLM investigation + orchestration
   - Investigators inspect suspicious windows.
   - The orchestrator aggregates the evidence and produces a final ATTACK/NORMAL decision.

---

## Current run summary

### Strict point-wise metrics
- Precision: 0.9488
- Recall: 0.2337
- F1: 0.3750
- TP: 426
- FP: 23
- FN: 1,397
- TN: 13,151

### Point-adjusted SWaT-style metrics
- Precision: 0.9836
- Recall: 0.7548
- F1: 0.8541

### Operational metrics
- Total windows processed: 14,997
- Actual triggered windows: 807
- Total LLM API calls: 4,912
- Total prompt tokens: 5,881,645
- Total completion tokens: 2,484,765

---

## Architecture

The pipeline is organized as:

- [main.py](main.py) — entry point for running the full MAS pipeline
- [graph.py](graph.py) — LangGraph orchestration flow
- [state.py](state.py) — typed state and schemas
- [detector.py](detector.py) — detector wrappers
- [escalation_gate.py](escalation_gate.py) — triggering and recheck logic
- [agents/investigator.py](agents/investigator.py) — per-stage LLM investigators
- [agents/orchestrator.py](agents/orchestrator.py) — final decision aggregation




## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

---


## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| MAX_WINDOWS | 5000 | Number of windows to process |
| START_WINDOW | 0 | Window index to start from |
| GATE_PERSISTENCE | 2 | Consecutive breaches required to trigger |
| GATE_SEVERITY_MULT | 1.0 | Score multiplier filter (1.0 = disabled) |
| GATE_RECHECK | none | Fixed recheck interval (or none for adaptive) |

```bash
# Process all 15K windows
MAX_WINDOWS=15000 python main.py

# Resume from window 5000
START_WINDOW=5000 MAX_WINDOWS=10000 python main.py

# Fixed recheck for lower LLM cost
GATE_RECHECK=5 python main.py
```

---

## Key Design Decisions

Why P3, P4, P5 Only?

The 6 TranAD detectors were evaluated on ROC-AUC. Stages P1, P2, and P6 showed low discriminative power and were excluded from the LLM pipeline to reduce API costs without sacrificing detection quality. grade_system.py produces the full per-stage comparison to validate this decision.

Trade-off: This exclusion is the primary reason for the low attack coverage (26%). Many attacks target P1/P2/P6 exclusively and are invisible to the MAS pipeline.

Why an Escalation Gate?

Without the gate, the raw detector generates ~1,989 threshold breaches — each would trigger an LLM call. The gate requires N consecutive breaches (persistence) and suppresses re-investigation during active episodes, reducing threshold breaches by 55.6% while maintaining detection quality.

Why LLM Agents?

The detector answers "is something wrong?" The LLM answers "what is wrong and should we act?" By giving the LLM access to tools (sensor queries, physical constraint checks, baseline comparisons), it can distinguish a genuine attack from sensor noise with 91.8% precision — eliminating 97.2% of false positives.

Why Tool-Calling Instead of Raw Prompting?

The investigator uses a structured tool-calling loop (max 5 iterations) rather than a single prompt. This ensures the LLM follows a consistent investigation protocol: query → check rules → compare baseline → submit verdict. The structured StageVerdict schema (Pydantic) guarantees parseable output.

## Dataset

This system is evaluated on the SWaT (Secure Water Treatment) dataset from iTrust, Singapore University of Technology and Design:

- Normal data: 7 days of normal operation (~495K rows at 1Hz)
- Attack data: 5 days with 36 labeled cyber-physical attacks (~450K rows)
- 51 sensors: Flow meters, level sensors, pressure sensors, actuator states
- 6 process stages: P1 (Raw Water), P2 (Pre-treatment), P3 (UF), P4 (De-chlorination), P5 (RO), P6 (Backwash)
---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Anomaly Detection | PyTorch (TranAD) |
| Agent Framework | LangGraph (StateGraph, Send API) |
| LLM Inference | Groq (Qwen 3.6 27B, GPT-OSS 120B, Llama 3.3 70B) |
| Data Schemas | Pydantic v2 |
| Data Processing | Pandas, NumPy, scikit-learn |
| Threshold Calibration | Peaks-Over-Threshold (EVT) |

---

## Known Limitations & Future Work

1. Recall remains the main weakness: the strict point-wise F1 is 0.3750, which indicates that many true attacks are still missed even though precision is high.
2. Coverage is still limited by the detector stage selection: the current MAS path relies mainly on P3, P4, and P5, so attacks that are mainly visible in P1, P2, or P6 are under-detected.
3. LLM cost is substantial: the current run used 4,912 LLM calls and 5.88M prompt tokens, which makes the pipeline expensive to operate at scale.
4. Detection latency is non-trivial: several attacks are only detected after several windows, which may be too slow for time-sensitive industrial response.
5. Results are single-run evidence: stronger claims should be based on repeated runs with different start windows and a statistical summary such as mean and standard deviation.
6. Future work should focus on improving recall and coverage while preserving the low false-positive profile, for example through better detector calibration, stage expansion, or more selective LLM triggering.

---

## License

This project is for academic and research purposes. The SWaT dataset is provided by iTrust, SUTD under their own licensing terms.