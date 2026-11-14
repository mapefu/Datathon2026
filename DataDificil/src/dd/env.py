"""Simuladores Estándar y Difícil sobre el `CampusBatteryEnv` oficial, y ejecutores de los protocolos.

`SimEnv` solo cambia `_get_obs`: recorta los vectores de mercado a `K` horas (el oficial siempre da 24 e
ignora `forecast_horizon`) y, en Difícil, añade la meteo de la fila t−1 y `cap_max_kwh`/`power_max_kw`.
La física, las infracciones y el score son los del fichero oficial sin tocar.

Protocolos:
- P1: una simulación y un agente nuevos por semana (SOC al 50 %).
- P2: una sola simulación con las semanas seguidas, una sola instancia, SOC heredado (FAQ oficial).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MODES, STARTER_KIT

if str(STARTER_KIT) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT))
from environment import CampusBatteryEnv  # noqa: E402  (simulador oficial)

METEO_KEYS = {"irradiancia_wm2": "irradiancia_wm2", "temp_amb_c": "temp_ambient_c",
              "temp_panel_c": "temp_panel_c", "viento_ms": "vent_ms"}


def spec_of(mode: str) -> dict:
    m = MODES[mode]
    return {"capacity_kwh": m["cap"], "power_kw": m["pmax"], "efficiency": 0.90, "soc_init": 0.50,
            "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": m["K"],
            "degradation": False}


def sim_frame(sim: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Tabla que ve el simulador del modo: solar ×S."""
    out = sim.copy()
    out["pv_kw"] = out["pv_kw"] * MODES[mode]["S"]
    return out


class SimEnv(CampusBatteryEnv):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.mode = mode
        self.K = MODES[mode]["K"]
        super().__init__(df, spec=spec_of(mode))

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        h_end = min(self.t + self.K, self.n)
        obs["preus_properes_24h"] = self.df["precio_eur_per_kwh"].iloc[self.t:h_end].tolist()
        obs["co2_properes_24h"] = self.df["co2_kg_per_kwh"].iloc[self.t:h_end].tolist()
        if self.mode == "hard":
            prev = self.df.iloc[self.t - 1] if self.t > 0 else self.df.iloc[self.t]
            for col, key in METEO_KEYS.items():
                obs["noves_dades_5min"][key] = float(prev[col]) if col in self.df.columns else 0.0
            obs["cap_max_kwh"] = self.cap
            obs["power_max_kw"] = self.power
        return obs


# ----------------------------------------------------------------------------- agentes
def load_agent_module(path: str | Path, overrides: dict | None = None):
    """Importa un agent.py como módulo nuevo (estado de módulo aislado) y aplica overrides."""
    path = Path(path).resolve()
    name = f"agent_{abs(hash((str(path), time.perf_counter_ns())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in (overrides or {}).items():
        setattr(mod, k, v)
    return mod


def run(agent, df_mode: pd.DataFrame, mode: str) -> dict:
    """Simula `agent` sobre `df_mode` (solar ya escalada). Devuelve series por paso."""
    env = SimEnv(df_mode, mode)
    obs, done = env.reset(), False
    times = []
    while not done:
        t0 = time.perf_counter()
        a = agent.act(obs)
        times.append(time.perf_counter() - t0)
        obs, done = env.step(a)
    return {"grid": np.asarray(env.grid), "infr": np.asarray(env.infraccions_per_step),
            "actions": np.asarray(env.actions, dtype=float), "soc": np.asarray(env.soc_series),
            "times": np.asarray(times), "env": env}


def week_id(g: pd.DataFrame) -> str:
    return str(g.index[0].date())


def run_p1(agent_path, weeks: list[pd.DataFrame], mode: str, overrides: dict | None = None) -> dict:
    from .score import week_stats

    out = {}
    for g in weeks:
        mod = load_agent_module(agent_path, overrides)
        gm = sim_frame(g, mode)
        t0 = time.perf_counter()
        r = run(mod.BatteryAgent(), gm, mode)
        out[week_id(g)] = week_stats(r["grid"], gm, int(r["infr"].sum()), time.perf_counter() - t0)
    return out


def run_p2(agent_path, weeks: list[pd.DataFrame], mode: str, overrides: dict | None = None,
           keep: bool = False) -> dict:
    from .score import week_stats

    mod = load_agent_module(agent_path, overrides)
    gm = sim_frame(pd.concat(weeks), mode)
    t0 = time.perf_counter()
    agent = mod.BatteryAgent()
    init_s = time.perf_counter() - t0
    r = run(agent, gm, mode)
    out, i = {}, 0
    for g in weeks:
        n = len(g)
        out[week_id(g)] = week_stats(r["grid"][i:i + n], gm.iloc[i:i + n], int(r["infr"][i:i + n].sum()),
                                     float(r["times"][i:i + n].sum()) + (init_s if i == 0 else 0.0))
        i += n
    if keep:
        out["_trace"] = {"actions": r["actions"], "soc": r["soc"], "grid": r["grid"], "index": gm.index}
    return out
