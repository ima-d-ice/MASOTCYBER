from __future__ import annotations
import os
from typing import Any
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import TranADCore
from data_utils import load_config
from escalation_gate import EscalationGate, StageConfig
from tools import ACTIVE_STAGES


# Fallback feature lists — updated to match your real stage configs
STAGE_FEATURES: dict[str, list[str]] = {
    "P3": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302"],
    "P4": ["AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403", "P404", "UV401"],
    "P5": ["AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502", "PIT501", "PIT502", "PIT503"],
}


class StageDetectorBundle:
    """Holds model, scaler, threshold, features, and device for one stage."""

    def __init__(
        self,
        stage: str,
        model: TranADCore,
        scaler: MinMaxScaler,
        threshold: float,
        features: list[str],
        device: torch.device,
    ):
        self.stage = stage
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.features = features
        self.device = device

    def score(self, window_np: np.ndarray) -> float:
        """Returns mean MSE of last timestep (Phase 2 reconstruction)."""
        scaled = self.scaler.transform(window_np)
        t = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, x2 = self.model(t)
        residual = ((x2 - t) ** 2).squeeze(0)[-1].cpu().numpy()
        score = float(np.mean(residual))
        if not np.isfinite(score):
            print(f"[detector] WARNING: Non-finite score ({score}) for {self.stage}, treating as normal")
            return 0.0
        return score


def load_detectors(
    device: torch.device,
    df_normal: pd.DataFrame,
    configs_dir: str = "configs",
) -> dict[str, StageDetectorBundle]:
    """Load P3/P4/P5 detectors with scalers fit on normal data."""
    bundles: dict[str, StageDetectorBundle] = {}
    df_normal = df_normal.copy()
    
    # Training-calibrated thresholds (from POT calibration during training)
    thresholds = {"P3": 0.00091, "P4": 0.02852, "P5": 0.00405, "P1": 0.00471, "P2": 0.00048, "P6": 0.00011}

    for stage in ACTIVE_STAGES:
        config_path = os.path.join(configs_dir, f"stage_{stage.lower()}.yaml")
        if os.path.exists(config_path):
            config = load_config(config_path)
            feats = config["features"]
            print(f"[detector] {stage}: loaded {len(feats)} features from {config_path}")
        else:
            feats = STAGE_FEATURES[stage]
            print(f"[detector] {stage}: using hardcoded fallback ({len(feats)} features)")

        for f in feats:
            if f not in df_normal.columns:
                print(f"[detector] WARNING: Sensor {f} missing from data, falling back to 0.0")
                df_normal[f] = 0.0

        data = df_normal[feats].values
        scaler = MinMaxScaler()
        scaler.fit(data[:int(len(data) * 0.8)])

        model = TranADCore(feature_dim=len(feats), window_size=30, hidden_dim=64)
        weights_path = f"weights/agent_{stage.lower()}.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"[detector] {stage}: loaded weights from {weights_path}")
        else:
            print(f"[detector] WARNING: {weights_path} not found — using random init!")

        model.to(device).eval()

        bundles[stage] = StageDetectorBundle(stage, model, scaler, thresholds.get(stage, 0.0), feats, device)

    return bundles


# In detector.py

def build_gate(
    bundles: dict[str, StageDetectorBundle],
    persistence: int = 2,
    recheck_interval: int | None = None,
    severity_mult: float = 1.0
) -> EscalationGate:
    """Build the escalation gate with configurable persistence and recheck_interval."""
    return EscalationGate({
        stage: StageConfig(
            threshold=b.threshold,
            persistence=persistence,
            severity_mult=severity_mult,
            recheck_interval=recheck_interval
        )
        for stage, b in bundles.items()
    })