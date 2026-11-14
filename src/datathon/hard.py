"""Mode Difícil: simulador local, evaluación por semanas y oráculo LP.

Cambios respecto al modo estándar (según la pestaña "Mode Difícil" de la plataforma):
- Batería 6.000 kWh / ±1.500 kW (η 90 %, SOC inicial 50 %).
- Solar ×12 (`pv_kw * 12`, hasta ≈1.700 kW): a mediodía hay excedentes que se vierten a 0 € / 0 kg.
- `preus_properes_24h` y `co2_properes_24h` traen solo 3 horas (la actual y las 2 siguientes).
- `noves_dades_5min` añade la meteo de la hora anterior: irradiancia_wm2, temp_ambient_c,
  temp_panel_c, vent_ms. `obs` incluye "cap_max_kwh" y "power_max_kw".

Dos formas de evaluar (el score siempre se calcula sobre el agregado de todas las semanas,
con una base común de CO2 y coste):
- continuous=True (lo que dice el FAQ oficial): UNA instancia del agente y UNA simulación con
  todas las semanas seguidas. El SOC no se reinicia entre semanas (empieza al 50 % solo en la
  primera hora) y `obs["timestamp"]` salta de un mes al siguiente. En la primera hora de cada
  semana, `noves_dades_5min` trae la última hora de la semana anterior y los vectores de
  precio/CO2 del final de una semana ya incluyen las primeras horas de la siguiente.
- continuous=False (por defecto, histórico): cada semana es una simulación independiente
  (SOC reiniciado al 50 %, vectores truncados al final de la semana).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .evaluate import OFFICIAL_PENALTY, load_agent_class
from environment import CampusBatteryEnv  # importado vía sys.path en evaluate.py

PV_SCALE = 12.0
HORIZON = 3
SPEC_HARD = {
    "capacity_kwh": 6000.0,
    "power_kw": 1500.0,
    "efficiency": 0.9,
    "soc_init": 0.5,
    "soc_min": 0.0,
    "soc_max": 1.0,
    "dt_hours": 1,
    "degradation": False,
}
METEO_KEYS = {
    "irradiancia_wm2": "irradiancia_wm2",
    "temp_amb_c": "temp_ambient_c",
    "temp_panel_c": "temp_panel_c",
    "viento_ms": "vent_ms",
}


def hard_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla horaria con la solar escalada ×12 (lo que ve el simulador del Mode Difícil)."""
    out = df.copy()
    out["pv_kw"] = out["pv_kw"] * PV_SCALE
    return out


class HardEnv(CampusBatteryEnv):
    """Simulador oficial con las observaciones del Mode Difícil. `df` ya debe tener pv ×12."""

    def __init__(self, df: pd.DataFrame, horizon: int = HORIZON):
        self.horizon = horizon
        super().__init__(df, spec=SPEC_HARD)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        h_end = min(self.t + self.horizon, self.n)
        obs["preus_properes_24h"] = self.df["precio_eur_per_kwh"].iloc[self.t:h_end].tolist()
        obs["co2_properes_24h"] = self.df["co2_kg_per_kwh"].iloc[self.t:h_end].tolist()
        prev = self.df.iloc[self.t - 1] if self.t > 0 else self.df.iloc[self.t]
        for col, key in METEO_KEYS.items():
            if col in self.df.columns:
                obs["noves_dades_5min"][key] = float(prev[col])
        obs["cap_max_kwh"] = self.cap
        obs["power_max_kw"] = self.power
        return obs


def weeks(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Bloques de evaluación: un bloque por mes (en validación, los últimos 7 días de cada mes)."""
    return [g for _, g in df.groupby(df.index.to_period("M"))]


def _score(grid, price, co2, base, infr) -> dict:
    co2_base, cost_base = float(base @ co2), float(base @ price)
    d_co2 = 100 * (co2_base - float(grid @ co2)) / co2_base
    d_cost = 100 * (cost_base - float(grid @ price)) / cost_base
    raw = 0.7 * d_co2 + 0.3 * d_cost
    return {
        "score_oficial": round(max(0.0, raw - OFFICIAL_PENALTY * infr), 3),
        "co2_estalviat_pct": round(d_co2, 3),
        "cost_estalviat_pct": round(d_cost, 3),
        "infraccions": int(infr),
    }


def _evaluate_continuous(agent_cls, df: pd.DataFrame, env_cls=HardEnv, **env_kw) -> dict:
    """Una sola instancia del agente y una sola simulación con todas las semanas seguidas
    (SOC continuo y saltos de `timestamp` entre semanas), como describe el FAQ oficial."""
    t0 = time.perf_counter()
    agent = agent_cls()
    env = env_cls(df, **env_kw)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    grid = np.asarray(env.grid)
    price = df["precio_eur_per_kwh"].to_numpy()
    co2 = df["co2_kg_per_kwh"].to_numpy()
    base = np.maximum(df["load_kw"].to_numpy() - df["pv_kw"].to_numpy(), 0.0)
    infr_step = np.asarray(env.infraccions_per_step)
    per_week = {}
    month = df.index.to_period("M")
    for m in month.unique():
        sel = np.asarray(month == m)
        per_week[str(m)] = _score(grid[sel], price[sel], co2[sel], base[sel],
                                  int(infr_step[sel].sum()))["score_oficial"]
    res = _score(grid, price, co2, base, env.clipped + env.violations)
    res["por_semana"] = per_week
    res["segundos"] = time.perf_counter() - t0
    res["acciones"] = pd.Series(env.actions, index=df.index)
    return res


def evaluate_hard(agent_cls, df: pd.DataFrame, fresh_agent: bool = True, env_cls=HardEnv,
                  week_hook=None, continuous: bool = False, **env_kw) -> dict:
    """Score agregado de todas las semanas.

    continuous=True: una instancia y una simulación continua (FAQ oficial); `week_hook` no
    se usa (el aviso de fin de bloque, si hace falta, se fija antes de llamar).
    continuous=False: cada semana simulada por separado. fresh_agent=True: una instancia nueva
    del agente por semana (el agente no tiene historia al empezar la semana)."""
    df = hard_frame(df)
    if continuous:
        return _evaluate_continuous(agent_cls, df, env_cls=env_cls, **env_kw)
    grids, prices, co2s, bases, actions = [], [], [], [], []
    infr, t0 = 0, time.perf_counter()
    per_week = {}
    agent = None if fresh_agent else agent_cls()
    for g in weeks(df):
        if week_hook is not None:
            week_hook(g)
        a = agent_cls() if fresh_agent else agent
        env = env_cls(g, **env_kw)
        obs, done = env.reset(), False
        while not done:
            obs, done = env.step(a.act(obs))
        grid = np.asarray(env.grid)
        price = g["precio_eur_per_kwh"].to_numpy()
        co2 = g["co2_kg_per_kwh"].to_numpy()
        base = np.maximum(g["load_kw"].to_numpy() - g["pv_kw"].to_numpy(), 0.0)
        w_infr = env.clipped + env.violations
        per_week[str(g.index[0].to_period("M"))] = _score(grid, price, co2, base, w_infr)["score_oficial"]
        grids.append(grid), prices.append(price), co2s.append(co2), bases.append(base)
        actions.append(pd.Series(env.actions, index=g.index))
        infr += w_infr
    res = _score(np.concatenate(grids), np.concatenate(prices), np.concatenate(co2s),
                 np.concatenate(bases), infr)
    res["por_semana"] = per_week
    res["segundos"] = time.perf_counter() - t0
    res["acciones"] = pd.concat(actions)
    return res


def _lp_week(net, w, cap, p_max, eta, soc0):
    """min Σ w·g, g ≥ net + c − d, g ≥ 0, dinámica de SOC exacta."""
    n = len(net)
    I = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((n, n))
    A_ub = sparse.hstack([I, -I, -I, Z])
    D = I - sparse.eye(n, k=-1, format="csr")
    A_eq = sparse.hstack([-eta * I, I / eta, Z, D])
    b_eq = np.zeros(n)
    b_eq[0] = soc0
    cost = np.concatenate([np.zeros(2 * n), w, np.zeros(n)])
    bounds = [(0, p_max)] * (2 * n) + [(0, None)] * n + [(0, cap)] * n
    sol = linprog(cost, A_ub=A_ub, b_ub=-net, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x[:n] - sol.x[n:2 * n], sol.x[2 * n:3 * n]


def oracle_hard(df: pd.DataFrame, continuous: bool = False) -> dict:
    """Óptimo con información perfecta, con los pesos de la base agregada. Semana a semana
    (SOC al 50 % en cada una) o, con continuous=True, todas seguidas con el SOC continuo."""
    df = hard_frame(df)
    net_all = (df["load_kw"] - df["pv_kw"]).to_numpy()
    base_all = np.maximum(net_all, 0.0)
    price_all = df["precio_eur_per_kwh"].to_numpy()
    co2_all = df["co2_kg_per_kwh"].to_numpy()
    w_all = 70 * co2_all / float(base_all @ co2_all) + 30 * price_all / float(base_all @ price_all)
    cap, p_max = SPEC_HARD["capacity_kwh"], SPEC_HARD["power_kw"]
    eta = SPEC_HARD["efficiency"] ** 0.5
    soc0 = SPEC_HARD["soc_init"] * cap
    if continuous:
        a, grid = _lp_week(net_all, w_all, cap, p_max, eta, soc0)
        res = _score(grid, price_all, co2_all, base_all, 0)
        res["acciones"] = pd.Series(a, index=df.index)
        return res
    grids, actions, i = [], [], 0
    for g in weeks(df):
        n = len(g)
        a, grid = _lp_week(net_all[i:i + n], w_all[i:i + n], cap, p_max, eta, soc0)
        grids.append(grid)
        actions.append(pd.Series(a, index=g.index))
        i += n
    res = _score(np.concatenate(grids), price_all, co2_all, base_all, 0)
    res["acciones"] = pd.concat(actions)
    return res


def evaluate_path(agent_path, df, **kw) -> dict:
    return evaluate_hard(load_agent_class(agent_path), df, **kw)


class PerturbedHardEnv(HardEnv):
    """HardEnv con observaciones degradadas (la física y el score no cambian)."""

    def __init__(self, df, nan_prob=0.0, str_timestamp=False, as_array=False, bad_obs_at=None,
                 drop_soc=False, drop_meteo=False, drop_spec=False, seed=0, **kw):
        self.nan_prob, self.str_timestamp, self.as_array = nan_prob, str_timestamp, as_array
        self.bad_obs_at, self.drop_soc, self.drop_meteo, self.drop_spec = bad_obs_at, drop_soc, drop_meteo, drop_spec
        self.rng = np.random.default_rng(seed)
        super().__init__(df, **kw)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        for key in ("preus_properes_24h", "co2_properes_24h"):
            v = np.asarray(obs[key], dtype=float)
            if self.nan_prob:
                v[self.rng.random(len(v)) < self.nan_prob] = np.nan
            obs[key] = v if self.as_array else v.tolist()
        if self.nan_prob:
            d = obs["noves_dades_5min"]
            for k in list(d):
                if self.rng.random() < self.nan_prob:
                    d[k] = float("nan")
        if self.drop_meteo:
            for k in METEO_KEYS.values():
                obs["noves_dades_5min"].pop(k, None)
        if self.drop_spec:
            obs.pop("cap_max_kwh", None)
            obs.pop("power_max_kw", None)
        if self.bad_obs_at == self.t:
            obs["noves_dades_5min"] = [1.0]
        if self.drop_soc:
            obs.pop("soc_kwh")
        if self.str_timestamp:
            obs["timestamp"] = str(obs["timestamp"])
        return obs
