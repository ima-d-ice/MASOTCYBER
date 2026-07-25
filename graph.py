from __future__ import annotations
from typing import Any
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from state import MASState, StageVerdict, OrchestratorReport
from agents.investigator import InvestigatorAgent
from agents.orchestrator import OrchestratorAgent
from tools import ACTIVE_STAGES, TOOL_REGISTRY
from detector import StageDetectorBundle
from escalation_gate import EscalationGate


def build_mas_graph(
    bundles: dict[str, StageDetectorBundle],
    gate: EscalationGate,
    router: Any,
):
    """Construct the LangGraph MAS pipeline (P3/P4/P5 only)."""

    investigators = {
        stage: InvestigatorAgent(stage, router, TOOL_REGISTRY)
        for stage in ACTIVE_STAGES
    }
    orchestrator = OrchestratorAgent(router)

    def detector_node(state: MASState) -> dict:
        """Run all stage detectors, update gate, find triggered stages."""
        window_df = state["window_df"]
        scores: dict[str, float] = {}
        triggers: dict[str, bool] = {}
        severities: dict[str, str] = {}

        for stage in ACTIVE_STAGES:
            if stage not in bundles:
                print(f"[detector_node] WARNING: {stage} not in bundles, skipping")
                scores[stage] = 0.0
                triggers[stage] = False
                severities[stage] = "low"
                continue
            feats = bundles[stage].features
            window_local = window_df.copy()
            for f in feats:
                if f not in window_local.columns:
                    print(f"[detector_node] WARNING: Sensor {f} missing from data, falling back to 0.0")
                    window_local[f] = 0.0

            try:
                window_np = window_local[feats].values[-30:]
                score = bundles[stage].score(window_np)
                triggered = gate.update(stage, score)
                
                threshold = bundles[stage].threshold
                ratio = score / threshold if threshold > 0 else 0
                
                severity = "low"
                if ratio > 50:
                    severity = "critical"
                elif ratio > 10:
                    severity = "high"
                elif ratio > 3:
                    severity = "medium"
                    
            except Exception as e:
                print(f"[detector_node] {stage} scoring failed: {e}")
                score = 0.0
                triggered = False
                severity = "low"

            scores[stage] = score
            triggers[stage] = triggered
            severities[stage] = severity

        return {
            "stage_scores": scores,
            "gate_triggers": triggers,
            "stage_severities": severities,
        }

    def investigator_node_factory(stage: str):
        def node(state: MASState) -> dict:
            triggers = state.get("gate_triggers", {})
            if not triggers.get(stage, False):
                return {"investigator_verdicts": [StageVerdict(
                    stage=stage,
                    verdict="False Positive",
                    confidence=1.0,
                    explanation=f"{stage} gate did not trigger; no investigation needed.",
                    implicated_sensors=[],
                )]}

            score = state["stage_scores"][stage]
            threshold = bundles[stage].threshold
            ts = state.get("timestamp", "")
            window_df = state.get("window_df")
            try:
                verdict = investigators[stage].investigate(score, threshold, ts, window_df)
                gate.set_last_verdict(stage, verdict.verdict)
            except Exception as e:
                verdict = StageVerdict(
                    stage=stage,
                    verdict="Unknown",
                    confidence=0.0,
                    explanation=f"Investigation failed: {e}",
                    implicated_sensors=[],
                )
                gate.set_last_verdict(stage, verdict.verdict)
            return {"investigator_verdicts": [verdict]}
        return node

    def orchestrator_node(state: MASState) -> dict:
        severities = state.get("stage_severities", {})
        critical_stages = [s for s, sev in severities.items() if sev == "critical"]
        
        if critical_stages:

            report = OrchestratorReport(
                consensus_verdict="ATTACK",
                involved_stages=critical_stages,
                report_markdown=f"CRITICAL OVERRIDE: {critical_stages} exceeded 50x threshold. Hardware-level isolation engaged.",
            )
            # Only set True Positive on stages that actually exceeded the critical
            # threshold (50x), not all triggered stages — a medium-severity stage
            # shouldn't get fast recheck intervals from a different stage's criticality
            for s in critical_stages:
                gate.set_last_verdict(s, "True Positive")
            return {
                "orchestrator_report": report,
                "final_prediction": 1,
                "active_alarms": critical_stages,
            }

        verdicts = state.get("investigator_verdicts", [])


        # Filter out synthetic "no investigation needed" verdicts —
        # only send REAL investigation results to the orchestrator LLM
        real_verdicts = [
            v for v in verdicts
            if not (v.verdict == "False Positive" and v.confidence == 1.0
                    and "no investigation needed" in v.explanation)
        ]
        if not real_verdicts:
            return {
                "orchestrator_report": None,
                "final_prediction": 0,
                "active_alarms": [],
            }
        ts = state.get("timestamp", "")
        try:
            scores = state.get("stage_scores", {})
            report = orchestrator.build_consensus(real_verdicts, ts, stage_scores=scores)
        except Exception as e:
            if isinstance(e, RuntimeError) and "TPD" in str(e):
                raise e
            report = OrchestratorReport(
                consensus_verdict="NORMAL",
                involved_stages=[],
                report_markdown=f"Orchestrator failed: {e}",
            )
        prediction = 1 if report.consensus_verdict == "ATTACK" else 0
        return {
            "orchestrator_report": report,
            "final_prediction": prediction,
            "active_alarms": report.involved_stages if prediction else [],
        }


        # ---------- Passthrough Node ----------
    def passthrough_node(state: MASState) -> dict:
        """No triggers -> NORMAL. NO LLM CALLS!"""
        return {
            "final_prediction": 0,
            "active_alarms": [],
            "orchestrator_report": OrchestratorReport(
                consensus_verdict="NORMAL",
                involved_stages=[],
                report_markdown="No triggers.",
            ),
        }

        # ---------- Routing ----------
    def route_after_detection(state: MASState) -> list[Send] | str:
        severities = state.get("stage_severities", {})
        if any(sev == "critical" for sev in severities.values()):
            return "orchestrator"
            
        triggers = state.get("gate_triggers", {})
        # Only fan-out to investigators whose gate actually triggered
        triggered = [s for s in ACTIVE_STAGES if triggers.get(s, False)]
        if triggered:
            return [Send(f"investigator_{s}", state) for s in triggered]
        # No triggers at all → passthrough (zero LLM calls)
        return "passthrough"

    # ---------- Build graph ----------
    g = StateGraph(MASState)

    g.add_node("detector", detector_node)
    g.add_node("passthrough", passthrough_node)

    # One node per stage investigator
    for stage in ACTIVE_STAGES:
        g.add_node(f"investigator_{stage}", investigator_node_factory(stage))

    g.add_node("orchestrator", orchestrator_node)

    g.add_edge(START, "detector")
    
    # Dynamic routing via Send API: only triggered investigators are dispatched
    g.add_conditional_edges(
        "detector",
        route_after_detection
    )

    # Fan-in: all investigators -> orchestrator
    for stage in ACTIVE_STAGES:
        g.add_edge(f"investigator_{stage}", "orchestrator")

    g.add_edge("orchestrator", END)
    g.add_edge("passthrough", END)

    return g.compile()