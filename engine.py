from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import math

from model import TranADCore


class POT:
    """
    Peaks-Over-Threshold (POT) using Extreme Value Theory (EVT).
    Dynamically finds the best threshold for anomaly scores based on the validation data.
    """
    def __init__(self, init_pct: int = 99, risk: float = 1e-4):
        self.init_pct = init_pct
        self.risk = risk
        self.threshold = None

    def calc_point2point(self, x: np.ndarray) -> float:
        """
        Fits a Generalized Pareto Distribution (GPD) to the tail of the scores.
        Simplified implementation for robustness.
        """
        # Find the empirical percentile as initial threshold (t)
        t = np.percentile(x, self.init_pct)
        peaks = x[x > t] - t
        
        if len(peaks) == 0:
            self.threshold = t
            return self.threshold
            
        # Simplified POT threshold logic (finding extreme upper bound)
        mean_peaks = np.mean(peaks)
        std_peaks = np.std(peaks) + 1e-6
        gamma = mean_peaks / std_peaks
        sigma = mean_peaks * (1 + gamma)
        
        # Calculate final threshold using GPD quantile
        epsilon = 1e-8
        try:
            self.threshold = t + (sigma / gamma) * (((self.risk * len(x) / len(peaks)) ** -gamma) - 1)
        except Exception:
            self.threshold = t + sigma * 5  # fallback
            
        if np.isnan(self.threshold) or np.isinf(self.threshold):
            self.threshold = t
            
        return self.threshold


def train_tranad(
    model: TranADCore,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TranADCore:
    """
    Orchestrates the two-phase training loop shifting weight over epochs.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    mse_loss = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Dynamic Loss Weights
        # Epoch 1: w1 = 1.0, w2 = 0.0 (Phase 1 Baseline focus)
        # Epoch 5: w1 = 0.2, w2 = 0.8 (Phase 2 Adversarial focus)
        w1 = 1.0 / epoch
        w2 = 1.0 - w1
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            x1, x2 = model(batch)
            
            # Loss formulation
            loss1 = mse_loss(x1, batch)
            loss2 = mse_loss(x2, batch)
            
            total_loss = (w1 * loss1) + (w2 * loss2)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
        train_loss /= len(train_loader)
        
        # Simple Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x1, x2 = model(batch)
                total_loss = (w1 * mse_loss(x1, batch)) + (w2 * mse_loss(x2, batch))
                val_loss += total_loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch}/{epochs}] | W1: {w1:.2f}, W2: {w2:.2f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        scheduler.step()
        
    return model


def calibrate_threshold(
    model: TranADCore,
    val_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """
    Passes validation data through trained model to get anomaly scores,
    then uses EVT POT to determine the final threshold.
    """
    model.eval()
    val_scores = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            x1, x2 = model(batch)
            # TranAD uses ||x2 - batch||^2 as the final anomaly score per timestamp
            # We take the mean across features to get the score per window timestamp
            scores = torch.mean((x2 - batch) ** 2, dim=-1)
            # Typically, we just care about the last timestamp of the window
            last_step_scores = scores[:, -1]
            val_scores.extend(last_step_scores.cpu().numpy())
            
    val_scores = np.array(val_scores)
    pot = POT(init_pct=99, risk=1e-4)
    threshold = pot.calc_point2point(val_scores)
    print(f"Calibrated Dynamic Threshold: {threshold:.5f}")
    return threshold