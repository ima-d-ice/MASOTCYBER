from __future__ import annotations
import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_p6_into_p5(configs_dir: str = "configs") -> tuple[dict | None, bool]:
    """
    Checks if P6 config has fold_into_p5: true.
    If so, returns a merged P5 config (features combined) and returns a flag indicating P6 is folded.
    """
    p5_path = os.path.join(configs_dir, "stage_p5.yaml")
    p6_path = os.path.join(configs_dir, "stage_p6.yaml")
    
    if not os.path.exists(p5_path) or not os.path.exists(p6_path):
        return None, False
        
    p5_cfg = load_config(p5_path)
    p6_cfg = load_config(p6_path)
    
    if p6_cfg.get("fold_into_p5", False):
        p5_cfg["features"].extend(p6_cfg["features"])
        # Deduplicate while preserving order just in case
        p5_cfg["features"] = list(dict.fromkeys(p5_cfg["features"]))
        return p5_cfg, True
        
    return p5_cfg, False


def load_data(normal_path: str = "data/normal_v1.csv", attack_path: str = "data/attack.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_path)
    df_attack = pd.read_csv(attack_path)
    
    # Standardize column names
    df_normal.columns = df_normal.columns.str.strip()
    df_attack.columns = df_attack.columns.str.strip()
    
    # Convert Timestamp if needed
    if 'Timestamp' in df_normal.columns:
        df_normal['Timestamp'] = pd.to_datetime(df_normal['Timestamp'], errors='coerce', dayfirst=True)
        df_normal.sort_values('Timestamp', inplace=True)
        df_normal.reset_index(drop=True, inplace=True)
    if 'Timestamp' in df_attack.columns:
        df_attack['Timestamp'] = pd.to_datetime(df_attack['Timestamp'], errors='coerce', dayfirst=True)
        df_attack.sort_values('Timestamp', inplace=True)
        df_attack.reset_index(drop=True, inplace=True)
        
    return df_normal, df_attack


class SWaTWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 10,
        is_attack: bool = False,
        labels: np.ndarray | None = None,
    ):
        """
        Args:
            data: np.ndarray of shape (N, D)
            window_size: int
            is_attack: bool
            labels: np.ndarray of shape (N,) for attack labels, only used if is_attack=True
        """
        self.data = data
        self.window_size = window_size
        self.is_attack = is_attack
        self.labels = labels
        self.n_samples = len(data) - window_size + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx : idx + self.window_size]
        window = torch.FloatTensor(window)
        if self.is_attack and self.labels is not None:
            label = self.labels[idx + self.window_size - 1]
            return window, torch.tensor(label, dtype=torch.float32)
        return window


def build_stage_pipeline(
    stage_config: dict,
    df_normal: pd.DataFrame,
    df_attack: pd.DataFrame,
    test_split: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]:
    """
    Builds the localized data pipeline for a single stage config.
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    features = stage_config["features"]
    
    # Fallback to zero for missing features to prevent crashes on bad configs
    df_normal_local = df_normal.copy()
    df_attack_local = df_attack.copy()
    for f in features:
        if f not in df_normal_local.columns:
            df_normal_local[f] = 0.0
        if f not in df_attack_local.columns:
            df_attack_local[f] = 0.0

    normal_data = df_normal_local[features].values
    attack_data = df_attack_local[features].values
    
    attack_labels = None
    if "Normal/Attack" in df_attack_local.columns:
        attack_labels = df_attack_local["Normal/Attack"].astype(str).str.strip().str.lower().eq("attack").astype(int).values

    # Train/Val split on Normal data
    split_idx = int(len(normal_data) * (1 - test_split))
    train_raw = normal_data[:split_idx]
    val_raw = normal_data[split_idx:]
    
    # Fit scaler ONLY on train normal data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(attack_data)
    
    window_size = stage_config.get("window_size", 10)
    
    train_ds = SWaTWindowDataset(train_scaled, window_size=window_size)
    val_ds = SWaTWindowDataset(val_scaled, window_size=window_size)
    test_ds = SWaTWindowDataset(test_scaled, window_size=window_size, is_attack=True, labels=attack_labels)
    
    batch_size = stage_config.get("batch_size", 128)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, scaler