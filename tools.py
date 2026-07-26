"""Tool definitions for OT MAS investigators."""
from __future__ import annotations
import json
import numpy as np
from typing import Any

ACTIVE_STAGES = ["P3", "P4", "P5"]

TOOL_REGISTRY = {
    "query_sensors",
    "check_physical_rules", 
    "compare_to_baseline",
    "submit_verdict",
}


def build_tool_schemas(stage: str) -> list[dict]:
    """Return OpenAI-compatible function schemas for Groq tool-calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_sensors",
                "description": f"Read current sensor values for {stage}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sensor names to query.",
                        }
                    },
                    "required": ["sensors"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_physical_rules",
                "description": f"Check if sensor readings violate physical constraints for {stage}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "readings": {
                            "type": "object",
                            "description": "Dict of sensor_name -> value.",
                        }
                    },
                    "required": ["readings"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_to_baseline",
                "description": f"Compare sensors to historical normal baseline for {stage}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Sensors to compare.",
                        }
                    },
                    "required": ["sensors"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_verdict",
                "description": "Submit final investigation verdict.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["True Positive", "False Positive", "Unknown"],
                        },
                        "confidence": {"type": "number"},
                        "explanation": {"type": "string"},
                        "implicated_sensors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["verdict", "confidence", "explanation"],
                },
            },
        },
    ]


def execute_tool_with_context(fname: str, context: dict, **kwargs) -> Any:
    """Execute a tool with access to the current data window."""
    window_df = context.get("window_df")
    stage = context.get("stage", "UNKNOWN")
    
    if window_df is None:
        return {"error": "No window data available"}
    
    if fname == "query_sensors":
        sensors = kwargs.get("sensors", [])
        result = {}
        for s in sensors:
            if s in window_df.columns:
                result[s] = {
                    "current": float(window_df[s].iloc[-1]),
                    "mean_window": float(window_df[s].mean()),
                    "max_window": float(window_df[s].max()),
                }
            else:
                result[s] = {"error": "Sensor not found"}
        return result
    
    elif fname == "check_physical_rules":
        readings = kwargs.get("readings", {})
        violations = []
        if stage == "P3":
            if readings.get("LIT301", 0) > 1000:
                violations.append("LIT301 exceeds tank maximum (1000mm)")
            if readings.get("LIT301", 500) < 0:
                violations.append("LIT301 negative level impossible")
            if readings.get("DPIT301", 0) < 0:
                violations.append("DPIT301 negative differential pressure impossible")
            if readings.get("P301", 0) > 0 and readings.get("FIT301", 1) == 0:
                violations.append("P301 is ON but FIT301 (flow) is 0 — pipe blockage or sensor tamper")
            if readings.get("P302", 0) > 0 and readings.get("FIT301", 1) == 0:
                violations.append("P302 is ON but FIT301 (flow) is 0 — pipe blockage or sensor tamper")
            mv_closed = all(readings.get(v, 0) == 0 for v in ["MV301", "MV302", "MV303", "MV304"])
            if mv_closed and readings.get("FIT301", 0) > 1:
                violations.append("All MV301-304 valves closed but FIT301 shows flow — valve or sensor tamper")
            if readings.get("FIT301", 0) > 5:
                violations.append(f"FIT301={readings.get('FIT301'):.2f} exceeds max expected flow (5 m³/h)")
        elif stage == "P4":
            if readings.get("LIT401", 0) > 1200:
                violations.append("LIT401 exceeds tank maximum (1200mm)")
            if readings.get("LIT401", 500) < 0:
                violations.append("LIT401 negative level impossible")
            if readings.get("P401", 0) > 0 and readings.get("FIT401", 1) == 0:
                violations.append("P401 is ON but FIT401 (flow) is 0 — pipe blockage or sensor tamper")
            if readings.get("P402", 0) > 0 and readings.get("FIT401", 1) == 0:
                violations.append("P402 is ON but FIT401 (flow) is 0 — pipe blockage or sensor tamper")
            if readings.get("AIT401", 0) < 0:
                violations.append("AIT401 negative conductivity impossible")
            if readings.get("AIT402", 0) < 0:
                violations.append("AIT402 negative ORP reading impossible")
            if readings.get("UV401", 0) > 0 and readings.get("FIT401", 1) == 0:
                violations.append("UV401 is ON but no flow — energy waste or attack")
        elif stage == "P5":
            if readings.get("AIT501", 0) < 0:
                violations.append("AIT501 negative reading impossible")
            if readings.get("AIT502", 0) < 0:
                violations.append("AIT502 negative reading impossible")
            if readings.get("AIT503", 0) < 0:
                violations.append("AIT503 negative reading impossible")
            if readings.get("AIT504", 0) < 0:
                violations.append("AIT504 negative reading impossible")
            if readings.get("PIT501", 0) < 0:
                violations.append("PIT501 negative pressure impossible")
            if readings.get("PIT502", 0) < 0:
                violations.append("PIT502 negative pressure impossible")
            if readings.get("PIT503", 0) < 0:
                violations.append("PIT503 negative pressure impossible")
            if readings.get("P501", 0) > 0 and readings.get("FIT501", 1) == 0:
                violations.append("P501 is ON but FIT501 (flow) is 0 — pipe blockage or sensor tamper")
            if readings.get("P502", 0) > 0 and readings.get("FIT501", 1) == 0:
                violations.append("P502 is ON but FIT501 (flow) is 0 — pipe blockage or sensor tamper")
            pit501 = readings.get("PIT501", 0)
            pit502 = readings.get("PIT502", 0)
            if pit501 > 0 and pit502 > 0 and pit502 > pit501:
                violations.append(f"PIT502 ({pit502:.1f}) > PIT501 ({pit501:.1f}) — permeate pressure exceeds feed, impossible in normal RO operation")
        return {"violations": violations, "violation_count": len(violations)}
    
    elif fname == "compare_to_baseline":
        sensors = kwargs.get("sensors", [])
        result = {}
        for s in sensors:
            if s in window_df.columns:
                vals = window_df[s].values
                z = float((vals[-1] - vals.mean()) / (vals.std() + 1e-6))
                result[s] = {"z_score": round(z, 2), "anomalous": abs(z) > 3}
            else:
                result[s] = {"error": "Sensor not found"}
        return result
    
    return {"error": f"Unknown tool: {fname}"}