from __future__ import annotations
import json
import time
import groq
from typing import Any, Optional
from collections import defaultdict
import pandas as pd
from pydantic import ValidationError

from state import StageVerdict
from tools import build_tool_schemas, execute_tool_with_context
from groq_client import DEFAULT_MODEL

SYSTEM_PROMPT = """You are the  Investigator Agent for Stage {stage} of the SWaT water treatment plant.

Your task is to investigate an anomaly flagged by the PyTorch detector and determine if it is a genuine cyber-physical attack (True Positive) or just sensor noise (False Positive).

CRITICAL WORKFLOW RULES:
1. You MUST investigate before concluding. Do NOT guess. Your first action must be to call `query_sensors` to get the current readings.
2. After getting readings, you MUST call `check_physical_rules` to see if any physical constraints are violated (e.g., pump is ON but flow is 0, tank levels are impossible).
3. You MUST call `compare_to_baseline` to check if sensors are statistically anomalous (z-score > 3).
4. You are strictly limited to the following tools: `query_sensors`, `check_physical_rules`, `compare_to_baseline`, and `submit_verdict`. Do NOT invent or call any other tools.

DECISION CRITERIA — Mark as "True Positive" IF ANY of the following:
- Physical constraints are violated (e.g., flow > 5, tank level < 0).
- Sensors show impossible or extremely improbable states (z-score > 5).
- There is clear evidence of coordinated malicious manipulation (e.g., multiple sensors deviate from baseline simultaneously in a physically correlated way).
- The anomaly score (MSE) is EXTREMELY high relative to the threshold, suggesting the TranAD model is highly confident this is not just noise. A score more than 10x the threshold is strong evidence of an attack even if individual sensors look normal.

DECISION CRITERIA — Mark as "False Positive" IF ALL of the following:
- No physical constraints are violated.
- All z-scores are < 3.
- The anomaly score is only marginally above threshold (< 2x).
- Sensors appear to be within normal operational variance with no coordinated patterns.

MAINTENANCE AWARENESS (Domain Context):
The following operational procedures produce sensor patterns that may appear anomalous but are planned and expected:
- P5 Backwash Cycle: FIT502, FIT503, FIT504 will spike simultaneously for several minutes with extremely high z-scores. This is ROUTINE MAINTENANCE, not an attack. ONLY flag as True Positive if OTHER sensors (e.g. AIT501-AIT504, PIT501-503) also show anomalies during this period, or if physical constraints are violated.
"""



class InvestigatorAgent:
    """Per-stage investigator with bounded tool-calling loop."""

    MAX_ITERATIONS = 5

    def __init__(self, stage: str, router: Any, tools_registry: set[str]):
        self.stage = stage
        self.router = router
        self.tools_registry = tools_registry
        self.system_prompt = SYSTEM_PROMPT.format(stage=stage)
        self.tool_schemas = build_tool_schemas(stage)

    def investigate(
        self,
        score: float,
        threshold: float,
        timestamp: str,
        window_df: Optional[pd.DataFrame] = None,
    ) -> StageVerdict:
        """Run investigation with access to current window data."""
        user_msg = (
            f"PyTorch TranAD flagged {self.stage} at {timestamp}. "
            f"MSE={score:.6f} (threshold={threshold:.6f}). Investigate."
        )
        # Context for tools (current window data)
        context = {"window_df": window_df, "stage": self.stage}
        from groq_client import MODEL_REGISTRY
        
        # Try primary investigator model, then fallback if all attempts fail
        models_to_try = [
            MODEL_REGISTRY["investigator"],
            MODEL_REGISTRY["investigator_fallback"],
            MODEL_REGISTRY.get("investigator_tertiary"),
        ]
        models_to_try = [m for m in models_to_try if m]  # Filter out None
        
        for model in models_to_try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]
            consecutive_failures = 0
            for attempt in range(self.MAX_ITERATIONS):
                try:
                    response = self.router.chat.completions.create(
                                model=model,
                                messages=messages,
                                tools=self.tool_schemas,
                                tool_choice="auto",
                                temperature=0.0,
                            )
                except Exception as e:
                    if isinstance(e, groq.APIStatusError):
                        if e.status_code in (401, 403, 404):
                            raise e
                        elif e.status_code == 413:
                            # Context Too Long: remove the oldest COMPLETE assistant+tool interaction
                            # Must remove the assistant message AND all its corresponding tool responses
                            # to avoid orphaned tool responses corrupting the conversation
                            oldest_asst_idx = None
                            for mi in range(len(messages)):
                                if messages[mi].get("role") == "assistant" and messages[mi].get("tool_calls"):
                                    oldest_asst_idx = mi
                                    break
                            if oldest_asst_idx is not None:
                                # Collect tool_call IDs from this assistant message
                                asst_msg = messages[oldest_asst_idx]
                                tc_ids = {tc["id"] for tc in asst_msg.get("tool_calls", [])}
                                # Remove the assistant message first
                                messages.pop(oldest_asst_idx)
                                # Remove all tool response messages for those IDs (iterate in reverse)
                                for mi in range(len(messages) - 1, -1, -1):
                                    if messages[mi].get("role") == "tool" and messages[mi].get("tool_call_id") in tc_ids:
                                        messages.pop(mi)
                                continue
                            else:
                                break
                    break  # Fall back to next model for 5xx, 429, etc.

                message = response.choices[0].message
                if not message.tool_calls:
                    # LLM responded with text but no tool calls.
                    text = (message.content or "").lower()
                    if "true positive" in text:
                        return StageVerdict(
                            stage=self.stage,
                            verdict="True Positive",
                            confidence=0.5,
                            explanation=f"Inferred from text: {message.content}",
                            implicated_sensors=[],
                        )
                    elif "false positive" in text:
                        return StageVerdict(
                            stage=self.stage,
                            verdict="False Positive",
                            confidence=0.5,
                            explanation=f"Inferred from text: {message.content}",
                            implicated_sensors=[],
                        )
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                    })
                      
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        # Model is degraded (returns text but no tools). Try next model.
                        break

                    # Feed error back to LLM to retry instead of giving up
                    messages.append({
                        "role": "user",
                        "content": "Your response was not valid. Please call a tool (query_sensors, check_physical_rules, compare_to_baseline, or submit_verdict)."
                    })
                    continue

                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        } for tc in message.tool_calls
                    ],
                })

                has_invalid_tool = False
                for tc in message.tool_calls:
                    fname = tc.function.name
                    try:
                        fargs = json.loads(tc.function.arguments)
                        if not isinstance(fargs, dict):
                            raise ValueError("args must be object")
                    except (json.JSONDecodeError, ValueError) as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": fname,
                            "content": f"Error: invalid JSON arguments. Details: {e}. Please retry with valid JSON.",
                        })
                        has_invalid_tool = True
                        continue

                    if fname == "submit_verdict":
                        try:
                            return StageVerdict(stage=self.stage, **fargs)
                        except ValidationError as e:
                            # Feed the validation error back instead of returning Unknown
                            error_str = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": fname,
                                "content": f"Schema validation error: {error_str}. Please correct your arguments and retry.",
                            })
                            # Do NOT set has_invalid_tool = True; this is a fixable schema error, not a degradation
                            continue

                    if fname in self.tools_registry:
                        try:
                            result = execute_tool_with_context(fname, context, **fargs)
                        except Exception as e:
                            result = f"Tool error: {e}"
                    else:
                        result = f"Unknown tool: {fname}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fname,
                        "content": json.dumps(result, default=str) if isinstance(result, (dict, list)) else str(result),
                    })
                    
                if has_invalid_tool:
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        break
                else:
                    consecutive_failures = 0

        return StageVerdict(
            stage=self.stage,
            verdict="Unknown",
            confidence=0.0,
            explanation="Exceeded max tool-calling iterations or models failed.",
            implicated_sensors=[],
        )