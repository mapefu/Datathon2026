"""Memoria, calendario del pliegue y filas de entrenamiento construidas con `agent.build_features`.

Las filas se generan decisión a decisión con la misma función que usa el agente, así que entrenamiento e
inferencia ven exactamente las mismas variables y los mismos límites de causalidad.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .calendar_upc import clear_sky, day_max, level_load, static_calendar, tolls
from .env import load_agent_module
from .paths import N_CAL, N_IDX, SUBMISSION


def agent_module():
    # DD_AGENT: agente alternativo cuyas variables se usan para entrenar (p. ej. submission_v3/agent.py)
    return load_agent_module(os.environ.get("DD_AGENT") or SUBMISSION / "agent.py")


def memory(full: pd.DataFrame) -> dict[str, np.ndarray]:
    """Memoria del agente: la solar es `pv_mem` (con la imputación desde la irradiancia)."""
    src = {"load": "load", "pv": "pv_mem", "irr": "irr", "tamb": "tamb", "price": "price", "co2": "co2"}
    return {k: full[v].to_numpy(dtype=np.float64).copy() for k, v in src.items()}


def calendar(full: pd.DataFrame) -> dict[str, np.ndarray]:
    """Calendario fijo + climatologías calculadas solo con los datos visibles de `full`."""
    C = static_calendar()
    C["cs"] = clear_sky(full["pv_mem"].to_numpy(), C)
    C["cs_dmax"] = day_max(C["cs"], C)
    C["lvl"] = level_load(full["load"].to_numpy(), C)
    C["toll"] = tolls(full["price"].to_numpy(), C)
    return C


def targets(full: pd.DataFrame) -> dict[str, np.ndarray]:
    """Verdad por índice (N_CAL, NaN fuera de datos). La solar objetivo es la medida, no la imputada."""
    out = {}
    for k in ("load", "pv", "price", "co2"):
        a = np.full(N_CAL, np.nan)
        a[:N_IDX] = full[k].to_numpy(dtype=np.float64)
        out[k] = a
    return out


def build_rows(agent, M, C, ts, k_obs: int = 3, hs=None, floors=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, t, h) apilando build_features para cada decisión t (floors: arranque en frío por fila)."""
    hs = np.arange(agent.H_FEAT) if hs is None else np.asarray(hs)
    blocks = []
    for i, t in enumerate(ts):
        fl = None if floors is None else int(floors[i])
        blocks.append(agent.build_features(M, C, int(t), k_obs, hs, fl))
    X = np.vstack(blocks)
    T = np.repeat(np.asarray(ts, dtype=np.int64), len(hs))
    Hh = np.tile(hs.astype(np.int64), len(ts))
    return X, T, Hh
