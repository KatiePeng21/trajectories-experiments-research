# src/utils.py
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Standardizer":
        # x: [..., D]
        m = x.mean(dim=0)
        s = x.std(dim=0).clamp(min=1e-8)
        return cls(mean=m, std=s)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b, dim=-1)


def ade_fde(pred: torch.Tensor, truth: torch.Tensor) -> Tuple[float, float]:
    """ADE/FDE in the same units as pred/truth (not geo-aware)."""
    with torch.no_grad():
        ade = _l2(pred, truth).mean().item()
        fde = _l2(pred[:, -1], truth[:, -1]).mean().item()
    return ade, fde


# ---- Geo metrics in meters (lat/lon in degrees, alt in meters) ----

def _haversine_m(lat1, lon1, lat2, lon2) -> torch.Tensor:
    """Great-circle distance on Earth surface in meters."""
    R = 6371000.0
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    lat1r = torch.deg2rad(lat1)
    lat2r = torch.deg2rad(lat2)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
    return R * c  # [..]

def ade_fde_geo_m(pred: torch.Tensor, truth: torch.Tensor, order=("lat","lon","alt")) -> Tuple[float, float]:
    """
    pred, truth: [B, H, T] with T containing lat, lon, alt (any order specified by `order`).
    Returns ADE/FDE in meters (3D: surface distance + altitude).
    """
    idx = {k:i for i,k in enumerate(order)}
    lat_p, lon_p = pred[..., idx["lat"]], pred[..., idx["lon"]]
    lat_t, lon_t = truth[..., idx["lat"]], truth[..., idx["lon"]]
    surf = _haversine_m(lat_p, lon_p, lat_t, lon_t)  # [B,H]
    if "alt" in idx:
        dz = (pred[..., idx["alt"]] - truth[..., idx["alt"]]).abs()  # [B,H]
        dist = torch.sqrt(surf**2 + dz**2)
    else:
        dist = surf
    with torch.no_grad():
        ade = dist.mean().item()
        fde = dist[:, -1].mean().item()
    return ade, fde