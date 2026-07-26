MAS — Multi-Agent Anomaly Detection for Industrial Control Systems

A 3-layer Multi-Agent System that combines PyTorch anomaly detectors with LLM-powered investigative reasoning to detect cyber-physical attacks on the SWaT (Secure Water Treatment) testbed.

<p align="center">
<img src="https://img.shields.io/badge/PA--F1-0.8472-brightgreen" alt="PA-F1">
<img src="https://img.shields.io/badge/False%20Positives-18-blue" alt="FP">
<img src="https://img.shields.io/badge/FP%20Reduction-97.2%25-orange" alt="FP Reduction">
<img src="https://img.shields.io/badge/LLM%20Calls-14997-purple" alt="LLM Calls">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
<img src="https://img.shields.io/badge/LangGraph-0.2%2B-green" alt="LangGraph">
</p>

⸻
The Problem

Traditional anomaly detectors on industrial control systems generate thousands of false alarms — roughly one every 2 minutes. Operators quickly suffer alarm fatigue and begin ignoring the system entirely, defeating the purpose of monitoring.
	Raw AI Detector (P3+P4+P5)	This System (MAS)
False Positives (5 days)	639	18
False Alarms / Hour	~5.1	0.14
Point-Adjusted F1	0.7394	0.8472
Precision	~0.74	0.9178



⸻
Architecture

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


Layer 1 — PyTorch TranAD Detectors

Six independent TranAD (Transformer-based Adversarial Anomaly Detection) models, one per SWaT process stage. Each computes a reconstruction MSE score against a calibrated threshold.

- Model: Two-phase adversarial transformer (Tuli et al., VLDB 2022)
- Active stages: P3, P4, P5 (selected by ROC-AUC; P1, P2, P6 excluded due to low discriminative power)
Threshold calibration: Peaks-Over-Threshold (POT) via Extreme Value Theory
Stage	Threshold	ROC-AUC	PR-AUC
P3	0.00091	0.8521	0.7673
P4	0.02852	0.8145	0.7002
P5	0.00405	0.8154	0.6795

Layer 2 — LLM Investigator Agents

Per-stage tool-calling LLM agents (Qwen 3.6 27B primary, GPT-OSS 20B fallback, GPT-OSS Safeguard 20B tertiary) that investigate flagged anomalies using three tools:
Tool	Purpose
query_sensors	Read current sensor values from the data window
check_physical_rules	Validate against physical constraints (flow ranges, tank limits)
compare_to_baseline	Compute z-scores against historical normal baselines

Each investigator produces a structured StageVerdict with verdict, confidence, explanation, and implicated sensors.

Layer 3 — Orchestrator Agent

A consensus-building LLM agent (GPT-OSS 120B primary, Llama 3.3 70B fallback) that aggregates all investigator verdicts, cross-correlates across stages, and produces:
- A final ATTACK / NORMAL verdict
- A detailed Markdown incident report with evidence, impact assessment, and recommended actions

Escalation Gate

A persistence-based filter between the detector and LLM layers that prevents redundant LLM calls:
- Requires N consecutive threshold breaches before triggering (persistence = 2)
- Suppresses repeated calls during active episodes via adaptive recheck intervals
Result: 55.6% reduction in threshold breaches vs. raw detector; 97.2% reduction in false positives vs. raw detectorAdaptive Recheck Intervals (worst-case assumes TP verdict = most frequent re-checking):
Last Verdict	Recheck Every N Windows
Unknown	8
False Positive	13
True Positive	6
⸻
Results

MAS Pipeline Performance (Comprehensive Run)
Metric	Point-Wise	Point-Adjusted
Precision	0.9178	0.9869
Recall	0.1103	0.7422
F1 Score	0.1969	0.8472
False Positive Rate	0.00137 (0.137%)	—
False Positives	18	—
True Positives	201	—

Operational Efficiency
Metric	Value
Total Windows Processed	14,997
Windows Triggered (Gate)	746
Total LLM Invocations	14,997
Investigator Triggers	746 stage-level events
LLM Calls per Detection (TP)	74.61
Rate Limit Errors	35
Unique API Keys Used	12
Total Key Rotations	1,728
Runtime (Full MAS)	~3.5 hours

Efficiency Note: With a fixed recheck_interval=5, LLM calls drop to ~354 (2.4% duty cycle) while maintaining PA-F1 ≈ 0.84. The 14,997 figure reflects adaptive rechecking with frequent True Positive verdicts (6-window interval).

Attack Scenario Coverage
Metric	Value
Segments Detected	9 / 35
Coverage Rate	25.7%
Avg Detection Latency	87s
Median Latency	0s

Detected Attacks: A6, A7, A8, A13, A15, A17, A18, A22, A34

Critical Limitation: The system misses 74.3% of attack scenarios, primarily because many attacks target P1/P2/P6 (excluded stages) or produce subtle sensor deviations below the TranAD threshold. This is the primary area for future improvement.

Three-Paradigm Comparison
Metric	Raw Detector (P3+P4+P5)	Gate Only	Full MAS
Threshold Breaches	1,989	884	—
False Positives	639	164	18
PW-F1	0.7083	0.5320	0.1969
PA-F1	0.7394	0.8346	0.8472
LLM Calls	0	0	14,997
FP Reduction vs Raw	—	—	97.2%

Comparison with Published SWaT Baselines
System	Year	PA-F1	FP Count	LLM Calls	Key Feature
DAGMM	2018	~0.80	High	0	GMM + autoencoder
LSTM-VAE	2019	~0.82	High	0	Variational autoencoder
USAD	2020	~0.85	High	0	Adversarial autoencoder
GDN	2021	~0.86	Medium	0	Graph deviation network
TranAD	2022	~0.90	3,506	0	Adversarial transformer
Sentinel MAS	2025	0.8472	18	14,997	TranAD + LLM verification

Key Insight: This system achieves sub-20 FPs while maintaining PA-F1 > 0.84 — a precision-focused profile distinct from pure ML baselines. The LLM verification layer eliminates 97.2% of false positives at the cost of significant API usage.
⸻
Grade Card
Category	Score	Grade
Detection (PA-F1)	0.8472	B+
Precision	0.9178	A
FP Reduction	97.2%	A
Attack Coverage	26%	F
LLM Efficiency	100% duty cycle (adaptive)	F
Robustness	35 rate limits, fallback success	B
⸻
Project Structure

finalmasot/
├── main.py                  # Entry point — runs the full MAS pipeline
├── graph.py                 # LangGraph DAG (fan-out/fan-in with conditional routing)
├── state.py                 # Pydantic schemas (MASState, StageVerdict, OrchestratorReport)
├── model.py                 # TranAD architecture (two-phase adversarial transformer)
├── engine.py                # Training loop + POT threshold calibration
├── detector.py              # Stage detector bundles (model + scaler + threshold)
├── escalation_gate.py       # Persistence-based gate with adaptive recheck
├── groq_client.py           # Production Groq client (round-robin, exponential backoff, per-model TPD)
├── tools.py                 # Tool definitions (query_sensors, check_rules, compare_baseline)
├── data_utils.py            # Data loading, windowing, SWaTWindowDataset
│
├── agents/
│   ├── investigator.py      # Per-stage LLM investigator (tool-calling loop, max 5 iterations, 3-model fallback)
│   └── orchestrator.py      # Consensus builder (2-model fallback, fixed TPD bug)
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
├── reports/                 # Generated incident reports (Markdown)
├── audit_logs.jsonl         # Full audit trail (per-window scores, triggers, verdicts)
├── mas_langgraph_results.csv # Final predictions vs ground truth
│
├── grade_system.py          # Comprehensive evaluation (all metrics, generates grade_report.md)
├── test_gate.py             # Gate evaluation with per-stage ROC-AUC
├── sweep_gate_params.py     # Grid search over gate parameters
├── gate_sweep.py            # LLM cost estimator + parameter sweep
├── analyze_logs.py          # Quick audit log analysis
├── compile_reports.py       # Compile all incident reports into one document
├── extract_reports.py       # Extract reports from audit logs to Markdown files
│
├── requirements.txt
├── .env                     # Groq API keys (up to 20 for rotation)
└── .gitignore

⸻
Setup

Prerequisites

- Python 3.10+
- ~300MB disk space (data + weights)
- Groq API key(s) for LLM inference (free tier works; multiple keys recommended for rotation)

Installation

git clone <repo-url>
cd finalmasot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt


Configuration

Create a .env file with your Groq API keys:

GROQ_API_KEY_1=gsk_your_key_here
GROQ_API_KEY_2=gsk_your_second_key
# ... up to GROQ_API_KEY_20


The system uses Qwen 3.6 27B for investigators and GPT-OSS 120B for the orchestrator by default. Override with:

ORCHESTRATOR_MODEL=openai/gpt-oss-120b
ORCHESTRATOR_FALLBACK_MODEL=llama-3.3-70b-versatile
INVESTIGATOR_MODEL=qwen/qwen3.6-27b
INVESTIGATOR_FALLBACK_MODEL=openai/gpt-oss-20b
INVESTIGATOR_TERTIARY_MODEL=openai/gpt-oss-safeguard-20b

⸻
Usage

Run the Full MAS Pipeline

python main.py


This processes the SWaT attack dataset through the full pipeline: detector → gate → investigator → orchestrator. Outputs:
- mas_langgraph_results.csv — predictions vs ground truth
- audit_logs.jsonl — full audit trail
- reports/ — incident reports for every triggered window

Environment variables:
Variable	Default	Description
MAX_WINDOWS	5000	Number of windows to process
START_WINDOW	0	Window index to start from
GATE_PERSISTENCE	2	Consecutive breaches required to trigger
GATE_SEVERITY_MULT	1.0	Score multiplier filter (1.0 = disabled)
GATE_RECHECK	none	Fixed recheck interval (or none for adaptive)

# Process all 15K windows
MAX_WINDOWS=15000 python main.py

# Resume from window 5000
START_WINDOW=5000 MAX_WINDOWS=10000 python main.py

# Fixed recheck for lower LLM cost
GATE_RECHECK=5 python main.py


Evaluate the System (No API Calls)

python grade_system.py


Scores all 6 detectors (P1–P6) across 15K windows and generates grade_report.md with:
- ROC-AUC, PR-AUC per stage
- Point-Wise and Point-Adjusted F1
- Detection latency per attack segment
- Attack scenario coverage (X/35 detected)
- LLM verdict accuracy analysis
- Three-paradigm comparison table
- Final letter grade card

Sweep Gate Parameters & Estimate LLM Costs

python gate_sweep.py --max-windows 1000 --output sweep.csv


Finds the optimal gate configuration by sweeping persistence, severity multiplier, and recheck intervals. Ranks by PA-F1 and LLM call count under worst-case assumptions.

Test the Escalation Gate

python test_gate.py                        # Default: persistence=2, recheck=10
GATE_PERSISTENCE=3 python test_gate.py     # Test stricter gate


Compile Incident Reports

python compile_reports.py      # Merge all reports → compiled_reports.md

⸻
Key Design Decisions

Why P3, P4, P5 Only?

The 6 TranAD detectors were evaluated on ROC-AUC. Stages P1, P2, and P6 showed low discriminative power and were excluded from the LLM pipeline to reduce API costs without sacrificing detection quality. grade_system.py produces the full per-stage comparison to validate this decision.

Trade-off: This exclusion is the primary reason for the low attack coverage (26%). Many attacks target P1/P2/P6 exclusively and are invisible to the MAS pipeline.

Why an Escalation Gate?

Without the gate, the raw detector generates ~1,989 threshold breaches — each would trigger an LLM call. The gate requires N consecutive breaches (persistence) and suppresses re-investigation during active episodes, reducing threshold breaches by 55.6% while maintaining detection quality.

Why LLM Agents?

The detector answers "is something wrong?" The LLM answers "what is wrong and should we act?" By giving the LLM access to tools (sensor queries, physical constraint checks, baseline comparisons), it can distinguish a genuine attack from sensor noise with 91.8% precision — eliminating 97.2% of false positives.

Why Tool-Calling Instead of Raw Prompting?

The investigator uses a structured tool-calling loop (max 5 iterations) rather than a single prompt. This ensures the LLM follows a consistent investigation protocol: query → check rules → compare baseline → submit verdict. The structured StageVerdict schema (Pydantic) guarantees parseable output.

Production Groq Client Features

- Round-robin key rotation across up to 20 API keys
- Exponential backoff with jitter for 429/5xx errors
- Per-model TPD tracking — a key exhausted for gpt-oss-120b can still be used for llama-3.3-70b
- Automatic fallback across 3 investigator models and 2 orchestrator models
- Graceful degradation to pure-detector mode if all keys exhaust TPD
⸻
Dataset

This system is evaluated on the SWaT (Secure Water Treatment) dataset from iTrust, Singapore University of Technology and Design:

- Normal data: 7 days of normal operation (~495K rows at 1Hz)
- Attack data: 5 days with 36 labeled cyber-physical attacks (~450K rows)
- 51 sensors: Flow meters, level sensors, pressure sensors, actuator states
- 6 process stages: P1 (Raw Water), P2 (Pre-treatment), P3 (UF), P4 (De-chlorination), P5 (RO), P6 (Backwash)
⸻
Tech Stack
Component	Technology
Anomaly Detection	PyTorch (TranAD)
Agent Framework	LangGraph (StateGraph, Send API)
LLM Inference	Groq (Qwen 3.6 27B, GPT-OSS 120B, Llama 3.3 70B)
Data Schemas	Pydantic v2
Data Processing	Pandas, NumPy, scikit-learn
Threshold Calibration	Peaks-Over-Threshold (EVT)
⸻
Known Limitations & Future Work

1. Attack Coverage (26%): The system misses most attacks targeting P1, P2, and P6. Including these stages would improve coverage but increase LLM costs and false positives.
2. LLM Efficiency: Adaptive rechecking with TP verdicts produces a 100% duty cycle (14,997 calls). Fixed recheck intervals reduce this but may miss evolving attacks.
3. Detection Latency: Average latency is 87s; some attacks are detected only after several minutes.
4. Single-Run Results: Metrics are from a single execution. For publication, run 5+ times with different start_window offsets and report mean ± std.
⸻
License

This project is for academic and research purposes. The SWaT dataset is provided by iTrust, SUTD under their own licensing terms.