"""Evaluación local de agentes con el simulador oficial + cota superior (oráculo LP)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .data import ROOT

STARTER_KIT = ROOT / "starter_kit"
if str(STARTER_KIT) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT))

from environment import CampusBatteryEnv  # noqa: E402  (simulador oficial)

SPEC = json.loads((STARTER_KIT / "battery_spec.json").read_text(encoding="utf-8"))

# docs/ENUNCIAT.md: cada infracción resta 1.00 punto del score oficial.
# environment.py usa 0.2 puntos; reportamos ambos y nos fiamos del más estricto.
OFFICIAL_PENALTY = 1.0


def load_agent_class(agent_path: str | Path):
    """Importa BatteryAgent desde un fichero agent.py arbitrario."""
    agent_path = Path(agent_path).resolve()
    sys.path.insert(0, str(agent_path.parent))  # módulos auxiliares del agente
    try:
        spec = importlib.util.spec_from_file_location(f"agent_{abs(hash(agent_path))}", agent_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(agent_path.parent))
    return mod.BatteryAgent


def simulate(agent, df: pd.DataFrame) -> CampusBatteryEnv:
    """Ejecuta el agente hora a hora sobre `df` (índice Fecha) y devuelve el entorno."""
    env = CampusBatteryEnv(df, spec=SPEC)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env


def _official(res: dict) -> float:
    raw = 0.7 * res["co2_estalviat_pct"] + 0.3 * res["cost_estalviat_pct"]
    return round(max(0.0, raw - OFFICIAL_PENALTY * res["infraccions"]), 2)


def evaluate(agent_path: str | Path, df: pd.DataFrame, by_month: bool = True) -> dict:
    """Score global (una única simulación continua) y desglose por mes."""
    env = simulate(load_agent_class(agent_path)(), df)
    res = env.evaluate_results()
    res["score_oficial"] = _official(res)
    if by_month:
        months = df.index.to_period("M")
        res["por_mes"] = {
            str(m): _official(env.evaluate_results(mask=(months == m)))
            for m in months.unique()
        }
    res["acciones"] = pd.Series(env.actions, index=df.index[: len(env.actions)])
    res["soc"] = pd.Series(env.soc_series, index=df.index[: len(env.soc_series)])
    return res


def oracle(df: pd.DataFrame) -> dict:
    """Óptimo con información perfecta (LP). Cota superior del score alcanzable.

    min  sum_t w_t * g_t
    s.a. g_t >= load_t - pv_t + c_t - d_t,  g_t >= 0
         soc_{t+1} = soc_t + eta_c * c_t - d_t / eta_d,  0 <= soc <= cap
         0 <= c_t, d_t <= P
    con w_t = 70 * co2_t / CO2_base + 30 * precio_t / Coste_base  (=> score en puntos).
    """
    n = len(df)
    cap, p_max = float(SPEC["capacity_kwh"]), float(SPEC["power_kw"])
    eta = float(SPEC["efficiency"]) ** 0.5
    soc0 = float(SPEC["soc_init"]) * cap

    net = (df["load_kw"] - df["pv_kw"]).to_numpy()
    price = df["precio_eur_per_kwh"].to_numpy()
    co2 = df["co2_kg_per_kwh"].to_numpy()
    base = np.maximum(net, 0.0)
    co2_base, cost_base = float(base @ co2), float(base @ price)
    w = 70 * co2 / co2_base + 30 * price / cost_base

    # Variables: [c (n), d (n), g (n), s (n)], s_t = SOC al final de la hora t
    I = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((n, n))
    # c - d - g <= -net
    A_ub = sparse.hstack([I, -I, -I, Z])
    b_ub = -net
    # s_t - s_{t-1} - eta c_t + d_t/eta = 0  (s_{-1} = soc0)
    D = I - sparse.eye(n, k=-1, format="csr")
    A_eq = sparse.hstack([-eta * I, I / eta, Z, D])
    b_eq = np.zeros(n)
    b_eq[0] = soc0

    cost = np.concatenate([np.zeros(2 * n), w, np.zeros(n)])
    bounds = [(0, p_max)] * (2 * n) + [(0, None)] * n + [(0, cap)] * n
    sol = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(sol.message)

    c, d, g = sol.x[:n], sol.x[n : 2 * n], sol.x[2 * n : 3 * n]
    d_co2 = 100 * (co2_base - g @ co2) / co2_base
    d_cost = 100 * (cost_base - g @ price) / cost_base
    return {
        "score_oficial": round(0.7 * d_co2 + 0.3 * d_cost, 2),
        "co2_estalviat_pct": round(d_co2, 2),
        "cost_estalviat_pct": round(d_cost, 2),
        "acciones": pd.Series(c - d, index=df.index),
    }
