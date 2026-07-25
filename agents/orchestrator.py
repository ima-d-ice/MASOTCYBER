from __future__ import annotations
import json
from typing import Any
from pydantic import ValidationError

from state import OrchestratorReport, StageVerdict
from groq_client import DEFAULT_MODEL

SYSTEM_PROMPT = """You are the Supreme Orchestrator for the SWaT water plant.
Multiple Stage Investigators have provided verdicts. Establish a GLOBAL consensus.

Call `submit_incident_report` with:
1. consensus_verdict: "ATTACK" only if genuine attack is occurring somewhere.
2. involved_stages: stages whose verdict was "True Positive" (or strongly
   physically correlated with a confirmed stage — co-occurrence alone is NOT evidence).
3. report_markdown: formatted Markdown report.

If unsure, default to NORMAL. False alarms are costly.
"""


class OrchestratorAgent:
    def __init__(self, router: Any):
        self.router = router
        self.tool_schema = [{
            "type": "function",
            "function": {
                "name": "submit_incident_report",
                "description": "Submit final incident report with consensus verdict.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "consensus_verdict": {"type": "string", "enum": ["ATTACK", "NORMAL"]},
                        "involved_stages": {"type": "array", "items": {"type": "string"}},
                        "report_markdown": {"type": "string"},
                    },
                    "required": ["consensus_verdict", "involved_stages", "report_markdown"],
                },
            },
        }]

    def build_consensus(self, verdicts: list[StageVerdict], timestamp: str, stage_scores: dict[str, float] | None = None) -> OrchestratorReport:
        verdicts_data = [v.model_dump() for v in verdicts]
        # Include raw anomaly scores for cross-stage correlation context
        payload = {"verdicts": verdicts_data}
        if stage_scores:
            payload["anomaly_scores"] = stage_scores
        verdicts_text = json.dumps(payload, indent=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"At {timestamp}, stage investigation results:\n{verdicts_text}\n\nCall submit_incident_report."},
        ]
        from groq_client import MODEL_REGISTRY
        
        models_to_try = [
            MODEL_REGISTRY["orchestrator"],
            MODEL_REGISTRY["orchestrator_fallback"]
        ]
        
        last_error = None
        for model in models_to_try:
            try:
                response = self.router.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self.tool_schema,
                    tool_choice={"type": "function", "function": {"name": "submit_incident_report"}},
                    temperature=0.0,
                )
                
                message = response.choices[0].message
                if message.tool_calls:
                    tc = message.tool_calls[0]
                    try:
                        fargs = json.loads(tc.function.arguments)
                        return OrchestratorReport(**fargs)
                    except (json.JSONDecodeError, ValueError) as e:
                        err_msg = f"Invalid JSON arguments: {e}"
                        last_error = f"{last_error} | {err_msg}" if last_error else err_msg
                        continue
                else:
                    last_error = ValueError("No tool calls returned by model")
                    continue
            except Exception as e:
                last_error = e
                continue
                
        # If all models fail, return a fallback report
        return OrchestratorReport(
            consensus_verdict="NORMAL",
            involved_stages=[],
            report_markdown=f"**DEGRADED FALLBACK**: Orchestrator failed across all models. Last error: {last_error}. Treat as NORMAL due to lack of evidence."
        )