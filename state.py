from __future__ import annotations
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field


class StageVerdict(BaseModel):
    """Schema returned by each investigator."""
    stage: str
    verdict: Literal["True Positive", "False Positive", "Unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    implicated_sensors: list[str] = Field(default_factory=list)


class OrchestratorReport(BaseModel):
    """Schema returned by orchestrator."""
    consensus_verdict: Literal["ATTACK", "NORMAL"]
    involved_stages: list[str]
    report_markdown: str


def merge_verdicts(left: list[StageVerdict], right: list[StageVerdict]) -> list[StageVerdict]:
    """Reducer: dedupe by stage, last write wins."""
    merged = {v.stage: v for v in left}
    for v in right:
        merged[v.stage] = v
    return list(merged.values())


class MASState(TypedDict, total=False):
    # Input
    window_idx: int
    window_df: object               # pandas.DataFrame
    window_tensor: object           # torch.Tensor (scaled)
    timestamp: str

    # Detector outputs
    stage_scores: dict[str, float]
    gate_triggers: dict[str, bool]
    stage_severities: dict[str, str]

    # Investigator outputs (parallel fan-out)
    investigator_verdicts: Annotated[list[StageVerdict], merge_verdicts]

    # Orchestrator output
    orchestrator_report: Optional[OrchestratorReport]

    # Final
    final_prediction: int                           # 0 or 1
    active_alarms: list[str]
    rate_limit_hits: int
    api_keys_used: list[int]