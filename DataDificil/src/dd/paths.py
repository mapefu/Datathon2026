"""Rutas y constantes compartidas del proyecto."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train"
PROC = ROOT / "data" / "processed"
FOLDS_DIR = PROC / "folds"
REPORTS = ROOT / "reports"
STARTER_KIT = ROOT / "starter_kit"
SUBMISSION = ROOT / "submission"

# Índice horario absoluto: idx = horas desde BASE (hora local sin zona, como `Fecha`).
BASE = pd.Timestamp("2025-09-01 00:00:00")
END = pd.Timestamp("2026-09-01 23:00:00")
N_IDX = int((END - BASE) / pd.Timedelta(hours=1)) + 1          # 8.808
N_CAL = N_IDX + 48                                              # calendario con margen de horizonte

ETA = math.sqrt(0.90)
MODES = {
    # S = escala de la solar; K = horas de mercado en obs; lam = (0,3/0,7)·CO2_base/Coste_base
    "std": {"cap": 2000.0, "pmax": 500.0, "S": 1.0, "K": 24, "lam": 0.2517},
    "hard": {"cap": 6000.0, "pmax": 1500.0, "S": 12.0, "K": 3, "lam": 0.2617},
}


def idx_of(ts) -> int:
    return int(round((pd.Timestamp(ts) - BASE) / pd.Timedelta(hours=1)))
