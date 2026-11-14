"""¿Cuánto podría ganar un modelo de predicción mejor? Cota superior con agentes "tramposos".

Cada agente tramposo es el mismo MPC que submission/agent.py, pero recibe el futuro REAL
(precio, CO2, carga neta) para un horizonte L. Si ni con predicciones perfectas mejora el
score, ningún modelo predictivo puede hacerlo. Cada semana se evalúa por separado, con el
SOC reiniciado al 50 % (como parece hacer el servidor).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .evaluate import SPEC, load_agent_class, oracle
from environment import CampusBatteryEnv  # importado vía sys.path en evaluate.py

CAP = float(SPEC["capacity_kwh"])
P_MAX = float(SPEC["power_kw"])
ETA = math.sqrt(float(SPEC["efficiency"]))


def ratio(df: pd.DataFrame) -> float:
    """CO2_base / Coste_base (kg/€) del periodo: fija el λ óptimo = (0,3/0,7)·ratio."""
    base = np.maximum(df["load_kw"] - df["pv_kw"], 0.0)
    return float(base @ df["co2_kg_per_kwh"]) / float(base @ df["precio_eur_per_kwh"])


class CheatAgent:
    """MPC con el futuro real: precio y CO2 de las próximas L horas y, si use_net, la carga neta.

    Sin use_net, el LP es el del agente (lineal, descarga de la hora actual limitada a la
    carga neta de la hora anterior). Con use_net, es el LP del oráculo restringido a L horas.
    λ es el óptimo para la ventana evaluada (otra trampa: el agente real no lo conoce).
    """

    def __init__(self, df: pd.DataFrame, L: int = 24, use_net: bool = False, v_quantile: float = 0.40):
        self.L, self.use_net, self.vq = L, use_net, v_quantile
        self.pos = {ts: i for i, ts in enumerate(df.index)}
        self.price = df["precio_eur_per_kwh"].to_numpy()
        self.co2 = df["co2_kg_per_kwh"].to_numpy()
        self.net = (df["load_kw"] - df["pv_kw"]).to_numpy()
        self.lam = 0.30 / 0.70 * ratio(df)

    def act(self, obs: dict) -> float:
        i, soc = self.pos[obs["timestamp"]], obs["soc_kwh"]
        j = min(i + self.L, len(self.price))
        w = self.co2[i:j] + self.lam * self.price[i:j]
        v = float(np.quantile(w[:24], self.vq)) if j - i >= self.L else 0.0
        dades = obs["noves_dades_5min"]
        d0_max = None if self.use_net else min(P_MAX, max(0.0, dades["ultim_consum_kw"] - dades["ultima_solar_kw"]))
        p = _lp(w, self.net[i:j] if self.use_net else None, soc, v, d0_max)
        max_c = max(0.0, min(P_MAX, (CAP - soc) / ETA - 1e-6))
        max_d = max(0.0, min(P_MAX, soc * ETA - 1e-6))
        return float(max(-max_d, min(max_c, p)))


def _lp(w, net, soc0, v, d0_max):
    """Primera acción del LP de H = len(w) horas. net=None: la red siempre absorbe la descarga."""
    H = len(w)
    I = sparse.identity(H, format="csr")
    D = I - sparse.eye(H, k=-1, format="csr")
    b_eq = np.zeros(H)
    b_eq[0] = soc0
    if net is None:  # variables [c, d, s]
        A_eq = sparse.hstack([-ETA * I, I / ETA, D], format="csr")
        cost = np.concatenate([w, -w, np.zeros(H)])
        bounds = [(0, P_MAX)] * (2 * H) + [(0, CAP)] * H
        bounds[H] = (0, d0_max)
        kw = {}
    else:  # variables [c, d, g, s], g = importación de la red >= max(0, net + c - d)
        Z = sparse.csr_matrix((H, H))
        A_eq = sparse.hstack([-ETA * I, I / ETA, Z, D], format="csr")
        cost = np.concatenate([np.zeros(2 * H), w, np.zeros(H)])
        bounds = [(0, P_MAX)] * (2 * H) + [(0, None)] * H + [(0, CAP)] * H
        kw = {"A_ub": sparse.hstack([I, -I, -I, Z], format="csr"), "b_ub": -net}
    cost[-1] -= v
    sol = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs", **kw)
    return sol.x[0] - sol.x[H]


def raw_score(agent, df: pd.DataFrame) -> float:
    """Score oficial sin redondear (0,7·ΔCO2% + 0,3·ΔCoste% − 1·infracciones)."""
    env = CampusBatteryEnv(df, spec=SPEC)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    grid = np.asarray(env.grid)
    base = np.maximum(df["load_kw"].to_numpy() - df["pv_kw"].to_numpy(), 0.0)
    co2, price = df["co2_kg_per_kwh"].to_numpy(), df["precio_eur_per_kwh"].to_numpy()
    d_co2 = (base - grid) @ co2 / (base @ co2)
    d_cost = (base - grid) @ price / (base @ price)
    return 70 * d_co2 + 30 * d_cost - (env.clipped + env.violations)


VARIANTS = {
    "agente": None,
    "+ λ real": dict(L=24),
    "+ λ real + precio/CO2 48 h": dict(L=48),
    "+ λ real + precio/CO2 72 h": dict(L=72),
    "+ λ real + carga neta 24 h": dict(L=24, use_net=True),
}


def headroom_week(args) -> dict:
    agent_path, label, week = args
    row = {"semana": label}
    for name, kw in VARIANTS.items():
        agent = load_agent_class(agent_path)() if kw is None else CheatAgent(week, **kw)
        row[name] = raw_score(agent, week)
    row["oráculo (semana entera)"] = oracle(week)["score_oficial"]
    return row


def weeks(df: pd.DataFrame, step_days: int | None = None, min_rows: int = 150):
    """step_days=None: una semana por mes (validación). Si no, ventanas de 7 días cada step_days."""
    if step_days is None:
        return [(str(m), g) for m, g in df.groupby(df.index.to_period("M"))]
    out = []
    for s in pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq=f"{step_days}D"):
        g = df[(df.index >= s) & (df.index < s + pd.Timedelta(days=7))]
        if len(g) >= min_rows:
            out.append((str(s.date()), g))
    return out


__all__ = ["CheatAgent", "VARIANTS", "headroom_week", "raw_score", "ratio", "weeks"]
