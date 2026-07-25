from __future__ import annotations
import os
import json
import logging
import torch
import pandas as pd
import numpy as np
import signal

class GraphTimeout(Exception):
    pass

def alarm_handler(signum, frame):
    raise GraphTimeout("Window processing exceeded 60s")

# Only set up signal handler on Unix
if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, alarm_handler)

from state import MASState
from data_utils import load_data
from detector import load_detectors, build_gate
from groq_client import ProductionGroqClient
from graph import build_mas_graph
from tools import ACTIVE_STAGES

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mas")


def calc_point_adjust(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Point-adjusted metric: if any prediction in a contiguous attack segment is 1,
    mark the entire segment as 1.
    """
    adj_preds = np.copy(preds)
    in_attack = False
    start_idx = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_attack:
            in_attack = True
            start_idx = i
        elif labels[i] == 0 and in_attack:
            in_attack = False
            if np.any(preds[start_idx:i]):
                adj_preds[start_idx:i] = 1
    if in_attack and np.any(preds[start_idx:]):
        adj_preds[start_idx:] = 1
    return adj_preds


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_normal, df_attack = load_data()
    max_windows = int(os.getenv("MAX_WINDOWS", "5000"))
    start_window = int(os.getenv("START_WINDOW", "0")) # NEW: Skip this many windows
        # --- GATE TUNING PARAMETERS ---
    gate_persistence = int(os.getenv("GATE_PERSISTENCE", "2"))
    gate_severity_mult = float(os.getenv("GATE_SEVERITY_MULT", "1.0"))
    recheck_str = os.getenv("GATE_RECHECK", "none")
    gate_recheck = int(recheck_str) if recheck_str.lower() != "none" else None

    bundles = load_detectors(device, df_normal)

    gate = build_gate(
        bundles,
        persistence=gate_persistence,
        recheck_interval=gate_recheck,
        severity_mult=gate_severity_mult
    )
    gate.reset()  # Clear any historical state from previous runs
    router = ProductionGroqClient()

    # Validate all configured models at startup
    print("Validating configured LLM models...")
    test_msg = [{"role": "user", "content": "Hi"}]
    from groq_client import MODEL_REGISTRY
    for role, model in MODEL_REGISTRY.items():
        try:
            router.chat.completions.create(model=model, messages=test_msg, max_tokens=1)
            print(f"  ✓ {role}: {model}")
        except Exception as e:
            print(f"  ✗ {role}: {model} — {type(e).__name__}: {e}")

    app = build_mas_graph(bundles, gate, router)

    window_size = 30
    stride = 30
   

    predictions, ground_truths, timestamps = [], [], []
    audit_path = "audit_logs.jsonl"
    # Clear audit file and setup reports dir
    try:
        with open(audit_path, "w", encoding="utf-8"):
            pass
    except OSError as e:
        log.error("Cannot create audit log file %s: %s", audit_path, e)
    from datetime import datetime
    # Capture timestamp once at startup to avoid midnight directory splits
    run_timestamp = datetime.now().strftime('%Y-%m-%d')
    report_dir = f"reports/{run_timestamp}"
    os.makedirs(report_dir, exist_ok=True)

    # Validate ground truth column exists
    if "Normal/Attack" not in df_attack.columns:
        log.warning("Column 'Normal/Attack' not found in attack data — ground truth will default to 0 (NORMAL)")

    # Track if we've entered pure-detector fallback mode (TPD exhaustion)
    pure_detector_mode = False

    # Efficiency tracking
    total_investigations_triggered = 0
    total_api_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_rate_limits = 0
    total_keys_rotated = 0
    start_row = start_window * stride
    end_window = start_window + max_windows

    log.info("Running LangGraph MAS on stages %s (Windows %d to %d)", ACTIVE_STAGES, start_window, end_window)
    

    # Changed the '0' in range() to 'start_row'
    window_idx = start_window - 1  # Initialize before loop
    for i in range(start_row, len(df_attack) - window_size, stride):
        window_idx = i // stride
        if window_idx >= end_window: # Changed to end_window
            break
        window = df_attack.iloc[i:i + window_size].copy()
        ts = str(window["Timestamp"].iloc[-1]) if "Timestamp" in window else ""

        router.reset_window_stats()

        state_input: MASState = {
            "window_idx": window_idx,
            "window_df": window,
            "timestamp": ts,
        }

        if pure_detector_mode:
            # Pure-detector fallback: run detector scores + gate logic only, no LLM
            try:
                scores_fb: dict[str, float] = {}
                triggers_fb: dict[str, bool] = {}
                any_critical_fb = False
                for stage in ACTIVE_STAGES:
                    if stage not in bundles:
                        scores_fb[stage] = 0.0
                        triggers_fb[stage] = False
                        continue
                    feats = bundles[stage].features
                    w_local = window.copy()
                    for feat in feats:
                        if feat not in w_local.columns:
                            w_local[feat] = 0.0
                    wnp = w_local[feats].values[-30:]
                    sc = bundles[stage].score(wnp)
                    triggered = gate.update(stage, sc)
                    scores_fb[stage] = sc
                    triggers_fb[stage] = triggered
                    ratio = sc / bundles[stage].threshold if bundles[stage].threshold > 0 else 0
                    if ratio > 50:
                        any_critical_fb = True
                pred_fb = 1 if (any_critical_fb or any(triggers_fb.values())) else 0
                result = {
                    "final_prediction": pred_fb,
                    "active_alarms": [s for s, t in triggers_fb.items() if t],
                    "orchestrator_report": None,
                    "stage_scores": scores_fb,
                    "gate_triggers": triggers_fb,
                }
            except Exception as e:
                log.error("Pure-detector fallback failed for window %d: %s", window_idx, e)
                result = {
                    "final_prediction": 0,
                    "active_alarms": [],
                    "orchestrator_report": None,
                    "stage_scores": {},
                    "gate_triggers": {},
                }
        else:
            try:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(60)  # 60 second timeout per window
                try:
                    result = app.invoke(state_input)
                finally:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
            except GraphTimeout:
                log.error("Window %d timed out after 60s", window_idx)
                result = {
                    "final_prediction": 0,
                    "active_alarms": [],
                    "orchestrator_report": None,
                    "stage_scores": {},
                    "gate_triggers": {},
                }
            except Exception as e:
                if isinstance(e, RuntimeError) and "TPD" in str(e):
                    log.critical("FATAL: All API keys exhausted (TPD). Switching to pure-detector mode.")
                    pure_detector_mode = True
                    # Re-process this window in pure-detector mode instead of pred=0
                    continue
                    
                log.error("Window %d failed: %s", window_idx, e)
                result = {
                    "final_prediction": 0,
                    "active_alarms": [],
                    "orchestrator_report": None,
                    "stage_scores": {},
                    "gate_triggers": {},
                }

        pred = int(result.get("final_prediction", 0))
        predictions.append(pred)

        gt_val = 0
        if "Normal/Attack" in window.columns:
            gt_val = 1 if str(window["Normal/Attack"].iloc[-1]).strip().lower() == "attack" else 0
        ground_truths.append(gt_val)
        timestamps.append(ts)

        stats = router.get_window_stats()
        triggers = result.get("gate_triggers", {})
        any_triggered = any(triggers.get(s, False) for s in ACTIVE_STAGES)
        if any_triggered:
            total_investigations_triggered += 1
        total_api_calls += stats.successful_requests
        total_prompt_tokens += stats.prompt_tokens
        total_completion_tokens += stats.completion_tokens
        total_rate_limits += stats.rate_limit_hits
        total_keys_rotated += len(set(stats.keys_used))

        report_obj = result.get("orchestrator_report")
        report_dict = report_obj.model_dump() if report_obj else None

        log_entry = {
            "window_idx": window_idx,
            "timestamp": ts,
            "scores": result.get("stage_scores", {}),
            "triggers": triggers,
            "prediction": pred,
            "ground_truth": gt_val,
            "rate_limit_hits": stats.rate_limit_hits,
            "api_keys_used": stats.keys_used,
            "active_model": stats.active_model,
            "orchestrator_report": report_dict,
        }
        try:
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
                f.flush()
        except Exception as e:
            log.error("Failed to write audit log for window %d: %s", window_idx, e)
            
        # Save markdown report as a separate file if it exists
        if report_obj and report_obj.report_markdown:
            report_filename = f"{report_dir}/incident_window_{window_idx}.md"
            try:
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(report_obj.report_markdown)
            except OSError as e:
                log.error("Failed to write report for window %d: %s", window_idx, e)

        if window_idx % 50 == 0:
            log.info("[%d] pred=%d gt=%d triggers=%s", window_idx, pred, gt_val, triggers)

    # Convert to numpy
    preds_arr = np.array(predictions)
    gts_arr = np.array(ground_truths)

    # Point-wise metrics
    tp = int(((preds_arr == 1) & (gts_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (gts_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (gts_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (gts_arr == 0)).sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)

    # Point-adjusted metrics
    adj_preds = calc_point_adjust(gts_arr, preds_arr)
    adj_tp = int(((adj_preds == 1) & (gts_arr == 1)).sum())
    adj_fp = int(((adj_preds == 1) & (gts_arr == 0)).sum())
    adj_fn = int(((adj_preds == 0) & (gts_arr == 1)).sum())
    adj_p = adj_tp / (adj_tp + adj_fp + 1e-9)
    adj_r = adj_tp / (adj_tp + adj_fn + 1e-9)
    adj_f1 = 2 * adj_p * adj_r / (adj_p + adj_r + 1e-9)

    print(f"\n{'='*60}")
    print("=== FINAL METRICS ===")
    print(f"{'='*60}")
    print(f"\n--- Point-Wise (Strict) ---")
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    print(f"\n--- Point-Adjusted (SWaT Standard) ---")
    print(f"Adj Precision: {adj_p:.4f} | Adj Recall: {adj_r:.4f} | Adj F1: {adj_f1:.4f}")

    log.info("--- System Efficiency ---")
    log.info(f"Total Windows Processed: {window_idx + 1}")
    log.info(f"Actual Time Windows Triggered: {total_investigations_triggered}")
    log.info(f"Total LLM API Calls (Groq): {total_api_calls}")
    log.info(f"Total Prompt Tokens: {total_prompt_tokens}")
    log.info(f"Total Completion Tokens: {total_completion_tokens}")
    log.info(f"Total Groq Rate Limits Hit: {total_rate_limits}")
    log.info(f"Total API Keys Rotated: {total_keys_rotated}")
    log.info("============================================================")

    out_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Ground_Truth": ground_truths,
        "Prediction": predictions,
    })
    out_df.to_csv("mas_langgraph_results.csv", index=False)
    
    # JSONL is already the canonical format. Skip expensive JSON conversion.
    # If JSON is needed, convert offline: python -c "import json; [json.loads(l) for l in open('audit_logs.jsonl')]"
    print("Audit logs kept in JSONL format: audit_logs.jsonl")
    print("\nSaved mas_langgraph_results.csv")


if __name__ == "__main__":
    main()