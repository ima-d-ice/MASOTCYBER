# Sentinel MAS — Multi-Agent Anomaly Detection for Industrial Control Systems

A **3-layer Multi-Agent System** that combines PyTorch anomaly detectors with LLM-powered investigative reasoning to detect cyber-physical attacks on the **SWaT (Secure Water Treatment)** testbed — while reducing false alarms by **99.2%**.

<p align="center">
<img src="https://img.shields.io/badge/PA--F1-0.8654-brightgreen" alt="PA-F1">
<img src="https://img.shields.io/badge/False%20Positives-27-blue" alt="FP">
<img src="https://img.shields.io/badge/FP%20Reduction-99.2%25-orange" alt="FP Reduction">
<img src="https://img.shields.io/badge/LLM%20Calls-354-purple" alt="LLM Calls">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
<img src="https://img.shields.io/badge/LangGraph-0.2%2B-green" alt="LangGraph">
</p>

---

## The Problem

Traditional anomaly detectors on industrial control systems generate **thousands of false alarms** — roughly one every 2 minutes. Operators quickly suffer **alarm fatigue** and begin ignoring the system entirely, defeating the purpose of monitoring.

| | Raw AI Detector | This System (Sentinel MAS) |
|---|---:|---:|
| **False Positives (5 days)** | 3,506 | **27** |
| **False Alarms / Hour** | 29.2 | **0.2** |
| **Point-Adjusted F1** | 0.8349 | **0.8654** |
| **Precision** | ~0.05 | **0.8866** |

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │           LangGraph DAG Pipeline             │
                          │                                              │
 Sensor Data              │  ┌──────────┐                                │
 (30s windows)  ─────────►│  │ DETECTOR │ TranAD models (P3, P4, P5)    │
                          │  │  NODE    │ Score each stage               │
                          │  └────┬─────┘                                │
                          │       │                                      │
                          │       ▼                                      │
                          │  ┌──────────┐    No triggers                 │
                          │  │ESCALATION├───────────────► PASSTHROUGH    │
                          │  │  GATE    │                 (pred=0,       │
                          │  └────┬─────┘                  zero LLM)    │
                          │       │ Triggered                            │
                          │       ▼                                      │
                          │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
                          │  │INVEST. P3│  │INVEST. P4│  │INVEST. P5│   │
                          │  │(LLM +    │  │(LLM +    │  │(LLM +    │   │
                          │  │ tools)   │  │ tools)   │  │ tools)   │   │
                          │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
                          │       │             │             │          │
                          │       └──────┬──────┘─────────────┘          │
                          │              ▼                               │
                          │       ┌──────────────┐                       │
                          │       │ ORCHESTRATOR │ Consensus builder     │
                          │       │ (LLM)        │ → ATTACK / NORMAL    │
                          │       └──────────────┘                       │
                          └──────────────────────────────────────────────┘
```

### Layer 1 — PyTorch TranAD Detectors

Six independent **TranAD** (Transformer-based Adversarial Anomaly Detection) models, one per SWaT process stage. Each computes a reconstruction MSE score against a calibrated threshold.

- **Model**: Two-phase adversarial transformer ([Tuli et al., VLDB 2022](https://arxiv.org/abs/2201.07284))
- **Active stages**: P3, P4, P5 (selected by ROC-AUC; P1, P2, P6 excluded due to low discriminative power)
- **Threshold calibration**: Peaks-Over-Threshold (POT) via Extreme Value Theory

### Layer 2 — LLM Investigator Agents

Per-stage **tool-calling LLM agents** (Qwen 3.6 27B for investigation, with fallback models) that investigate flagged anomalies using three tools:

| Tool | Purpose |
|------|---------|
| `query_sensors` | Read current sensor values from the data window |
| `check_physical_rules` | Validate against physical constraints (flow ranges, tank limits) |
| `compare_to_baseline` | Compute z-scores against historical normal baselines |

Each investigator produces a structured `StageVerdict` with verdict, confidence, explanation, and implicated sensors.

### Layer 3 — Orchestrator Agent

A consensus-building LLM agent (GPT-OSS 120B) that aggregates all investigator verdicts, cross-correlates across stages, and produces:
- A final `ATTACK` / `NORMAL` verdict
- A detailed Markdown incident report with evidence, impact assessment, and recommended actions

### Escalation Gate

A **persistence-based filter** between the detector and LLM layers that prevents redundant LLM calls:
- Requires `N` consecutive threshold breaches before triggering (persistence = 2)
- Suppresses repeated calls during active episodes (recheck interval = 5)
- **Result**: 97% reduction in LLM invocations vs. raw threshold breaches

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Point-Adjusted F1 | **0.8654** |
| Point-Wise Precision | **0.8866** |
| Point-Wise Recall | 0.1157 |
| Point-Wise F1 | 0.2048 |
| False Positive Rate | 0.205% |
| False Positives | 27 (down from 3,506) |
| LLM Invocations | 354 / 14,997 windows (2.4% duty cycle) |
| Rate Limit Errors | 0 |

### Comparison with Published Baselines

| System | Year | PA-F1 | FP Count | LLM Calls |
|--------|------|-------|----------|-----------|
| DAGMM | 2018 | ~0.80 | High | 0 |
| USAD | 2020 | ~0.85 | High | 0 |
| GDN | 2021 | ~0.86 | Medium | 0 |
| TranAD | 2022 | ~0.90 | 3,506 | 0 |
| **Sentinel MAS** | **2025** | **0.8654** | **27** | **354** |

### Sample Incident Report

When the system detects an attack, the orchestrator generates a detailed incident report:

> **Timestamp:** 2015-12-31 04:20:30
>
> A coordinated malicious manipulation detected in **Stage P3**. Verdict: **True Positive** (confidence: 0.98).
>
> - **Physical Violation:** Tank level LIT301 reads 1015.7, exceeding maximum 1000
> - **Statistical Anomalies:** DPIT301 z-score: 9.6, FIT301: 8.33, LIT301: 10.31
> - **Model Evidence:** TranAD MSE is 20.8× the detection threshold
>
> **Recommended Actions:** Activate emergency shutdown, isolate sensors, conduct forensic analysis...

354 such reports are generated and saved in `reports/`.

---

## Project Structure

```
finalmasot/
├── main.py                  # Entry point — runs the full MAS pipeline
├── graph.py                 # LangGraph DAG (fan-out/fan-in with conditional routing)
├── state.py                 # Pydantic schemas (MASState, StageVerdict, OrchestratorReport)
├── model.py                 # TranAD architecture (two-phase adversarial transformer)
├── engine.py                # Training loop + POT threshold calibration
├── detector.py              # Stage detector bundles (model + scaler + threshold)
├── escalation_gate.py       # Persistence-based gate to suppress redundant LLM calls
├── groq_client.py           # Production Groq client (round-robin rotation, exponential backoff, TPD handling)
├── tools.py                 # Tool definitions (query_sensors, check_rules, compare_baseline)
├── data_utils.py            # Data loading, windowing, SWaTWindowDataset
│
├── agents/
│   ├── investigator.py      # Per-stage LLM investigator (tool-calling loop, max 5 iterations)
│   └── orchestrator.py      # Consensus builder (cross-stage verdict aggregation)
│
├── configs/
│   ├── stage_p1.yaml        # Feature lists per stage
│   ├── stage_p2.yaml
│   ├── stage_p3.yaml
│   ├── stage_p4.yaml
│   ├── stage_p5.yaml
│   └── stage_p6.yaml
│
├── weights/
│   ├── agent_p1.pth → p6.pth   # Trained TranAD weights (6 stages)
│   └── monolithic_tranad.pth   # Single monolithic model (baseline)
│
├── data/
│   ├── normal_v1.csv        # SWaT normal operation (~150MB, 495K rows)
│   └── attack.csv           # SWaT attack dataset (~135MB, 450K rows)
│
├── reports/                 # 354 generated incident reports (Markdown)
├── audit_logs.jsonl         # Full audit trail (per-window scores, triggers, verdicts)
├── mas_langgraph_results.csv # Final predictions vs ground truth
│
├── grade_system.py          # Comprehensive evaluation (all metrics, generates grade_report.md)
├── test_gate.py             # Gate evaluation with per-stage ROC-AUC
├── sweep_gate_params.py     # Grid search over gate parameters
├── analyze_logs.py          # Quick audit log analysis
├── compile_reports.py       # Compile all incident reports into one document
├── extract_reports.py       # Extract reports from audit logs to Markdown files
│
├── requirements.txt
├── .env                     # Groq API keys (10 keys for rotation)
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10+
- ~300MB disk space (data + weights)
- Groq API key(s) for LLM inference (free tier works)

### Installation

```bash
git clone <repo-url>
cd finalmasot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your Groq API keys:

```env
GROQ_API_KEY_1=gsk_your_key_here
GROQ_API_KEY_2=gsk_your_second_key    # Optional: add up to 10 for rotation
```

The system uses **Qwen 3.6 27B** for investigators and **GPT-OSS 120B** for the orchestrator by default. Override with:

```env
ORCHESTRATOR_MODEL=openai/gpt-oss-120b
INVESTIGATOR_MODEL=qwen/qwen3.6-27b
```

---

## Usage

### Run the Full MAS Pipeline

```bash
python main.py
```

This processes the SWaT attack dataset through the full pipeline: detector → gate → investigator → orchestrator. Outputs:
- `mas_langgraph_results.csv` — predictions vs ground truth
- `audit_logs.jsonl` — full audit trail
- `reports/` — incident reports for every triggered window

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WINDOWS` | 5000 | Number of windows to process |
| `START_WINDOW` | 0 | Window index to start from |
| `GATE_PERSISTENCE` | 2 | Consecutive breaches required to trigger |
| `GATE_RECHECK` | 5 | Windows between re-investigations (or `none`) |

```bash
# Process all 15K windows
MAX_WINDOWS=15000 python main.py

# Resume from window 5000
START_WINDOW=5000 MAX_WINDOWS=10000 python main.py
```

### Evaluate the System (No API Calls)

```bash
python grade_system.py
```

Scores all 6 detectors (P1–P6) across 15K windows and generates `grade_report.md` with:
- ROC-AUC, PR-AUC per stage
- Point-Wise and Point-Adjusted F1
- Detection latency per attack segment
- Attack scenario coverage (X/35 detected)
- LLM verdict accuracy analysis
- Three-paradigm comparison table
- Final letter grade card

### Test the Escalation Gate

```bash
python test_gate.py                        # Default: persistence=2, recheck=10
GATE_PERSISTENCE=3 python test_gate.py     # Test stricter gate
```

### Sweep Gate Parameters

```bash
python sweep_gate_params.py    # Grid search: persistence × recheck combinations
```

### Compile Incident Reports

```bash
python compile_reports.py      # Merge all 354 reports → compiled_reports.md
```

---

## Key Design Decisions

### Why P3, P4, P5 Only?

The 6 TranAD detectors were evaluated on ROC-AUC. Stages P1, P2, and P6 showed low discriminative power and were excluded from the LLM pipeline to reduce API costs without sacrificing detection quality. `grade_system.py` produces the full per-stage comparison to validate this decision.

### Why an Escalation Gate?

Without the gate, the raw detector generates ~3,500 threshold breaches — each would trigger an LLM call. The gate requires `N` consecutive breaches (persistence) and suppresses re-investigation during active episodes, reducing LLM calls by **97%** while maintaining detection quality.

### Why LLM Agents?

The detector answers *"is something wrong?"* The LLM answers *"what is wrong and should we act?"* By giving the LLM access to tools (sensor queries, physical constraint checks, baseline comparisons), it can distinguish a genuine attack from sensor noise with **88.7% precision** — eliminating 99.2% of false positives.

### Why Tool-Calling Instead of Raw Prompting?

The investigator uses a structured tool-calling loop (max 5 iterations) rather than a single prompt. This ensures the LLM follows a consistent investigation protocol: query → check rules → compare baseline → submit verdict. The structured `StageVerdict` schema (Pydantic) guarantees parseable output.

---

## Dataset

This system is evaluated on the **SWaT (Secure Water Treatment)** dataset from iTrust, Singapore University of Technology and Design:

- **Normal data**: 7 days of normal operation (~495K rows at 1Hz)
- **Attack data**: 5 days with 36 labeled cyber-physical attacks (~450K rows)
- **51 sensors**: Flow meters, level sensors, pressure sensors, actuator states
- **6 process stages**: P1 (Raw Water), P2 (Pre-treatment), P3 (UF), P4 (De-chlorination), P5 (RO), P6 (Backwash)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Anomaly Detection | PyTorch (TranAD) |
| Agent Framework | LangGraph (StateGraph, Send API) |
| LLM Inference | Groq (Qwen 3.6 27B, GPT-OSS 120B) |
| Data Schemas | Pydantic v2 |
| Data Processing | Pandas, NumPy, scikit-learn |
| Threshold Calibration | Peaks-Over-Threshold (EVT) |

---

## License

This project is for academic and research purposes. The SWaT dataset is provided by iTrust, SUTD under their own licensing terms.
