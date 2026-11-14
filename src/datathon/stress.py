"""Escenarios de estrés: ¿qué pasa si el servidor no se comporta como el simulador local?

- Previsiones de 24 h con error (ruido multiplicativo que crece con el horizonte).
- NaN sueltos en las previsiones y `timestamp` como string.
- Formatos inesperados: previsiones como np.array, una observación malformada, SOC ausente.
- Distintas formas de evaluar: continuo, cada semana por separado.
- Tiempo de ejecución.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluate import SPEC, _official, load_agent_class, oracle
from environment import CampusBatteryEnv  # importado vía sys.path en evaluate.py


class PerturbedEnv(CampusBatteryEnv):
    """Simulador oficial con observaciones degradadas. La física y el score no cambian."""

    def __init__(self, df, noise=0.0, nan_prob=0.0, str_timestamp=False, seed=0, correlated=False,
                 as_array=False, bad_obs_at=None, drop_soc=False):
        """correlated=True: el error de una hora objetivo es el mismo en todas las previsiones
        que la incluyen (como un pronóstico real que se equivoca de forma persistente).
        as_array: previsiones como np.ndarray en vez de list.
        bad_obs_at: en ese paso, `noves_dades_5min` llega con un tipo inválido (una lista).
        drop_soc: se quita `soc_kwh` (quedan soc_pct y soc_percent)."""
        self.as_array, self.bad_obs_at, self.drop_soc = as_array, bad_obs_at, drop_soc
        self.noise, self.nan_prob, self.str_timestamp = noise, nan_prob, str_timestamp
        self.rng = np.random.default_rng(seed)
        self.correlated = correlated
        self.z = {key: self.rng.standard_normal(len(df)) for key in ("preus_properes_24h", "co2_properes_24h")}
        super().__init__(df, spec=SPEC)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        for key in ("preus_properes_24h", "co2_properes_24h"):
            v = np.asarray(obs[key], dtype=float)
            if self.noise:
                # La hora actual se conoce exacta (es preu_actual); el error crece con el horizonte.
                k = np.arange(len(v))
                z = self.z[key][self.t : self.t + len(v)] if self.correlated else self.rng.standard_normal(len(v))
                v = v * (1 + self.noise * np.sqrt(k / 23) * z)
            if self.nan_prob:
                v[self.rng.random(len(v)) < self.nan_prob] = np.nan
            obs[key] = v if self.as_array else v.tolist()
        if self.bad_obs_at == self.t:
            obs["noves_dades_5min"] = [obs["noves_dades_5min"]["ultim_consum_kw"]]
        if self.drop_soc:
            obs.pop("soc_kwh")
        if self.str_timestamp:
            obs["timestamp"] = str(obs["timestamp"])
        return obs


def run(agent_cls, df: pd.DataFrame, **kw) -> dict:
    env = PerturbedEnv(df, **kw)
    agent = agent_cls()
    obs, done = env.reset(), False
    t0 = time.perf_counter()
    while not done:
        obs, done = env.step(agent.act(obs))
    res = env.evaluate_results()
    res["score_oficial"] = _official(res)
    res["segundos"] = time.perf_counter() - t0
    return res


def weekly(agent_cls, df: pd.DataFrame) -> float:
    """Media del score con cada semana (mes) evaluada por separado, con el SOC reiniciado."""
    return float(np.mean([run(agent_cls, g)["score_oficial"] for _, g in df.groupby(df.index.to_period("M"))]))


def scenarios(val: pd.DataFrame, full: pd.DataFrame, seeds=(0, 1, 2)) -> list[tuple[str, callable]]:
    """Lista de (nombre, función(agent_cls) -> (score, infracciones))."""
    def noisy(n, correlated=False):
        def f(A):
            rs = [run(A, val, noise=n, seed=s, correlated=correlated) for s in seeds]
            return np.mean([r["score_oficial"] for r in rs]), max(r["infraccions"] for r in rs)
        return f

    def single(df, **kw):
        def f(A):
            r = run(A, df, **kw)
            return r["score_oficial"], r["infraccions"]
        return f

    return [
        ("val exacto", single(val)),
        ("año completo", single(full)),
        ("semanas separadas", lambda A: (weekly(A, val), 0)),
        ("ruido 10%", noisy(0.10)),
        ("ruido 20%", noisy(0.20)),
        ("ruido 30%", noisy(0.30)),
        ("ruido 20% correlado", noisy(0.20, correlated=True)),
        ("ruido 30% correlado", noisy(0.30, correlated=True)),
        ("NaN 5% + timestamp str", single(val, nan_prob=0.05, str_timestamp=True)),
        ("NaN 20%", single(val, nan_prob=0.20)),
        ("previsiones np.array", single(val, as_array=True)),
        ("1 obs malformada (h 50)", single(val, bad_obs_at=50)),
        ("sin soc_kwh", single(val, drop_soc=True)),
    ]


def oracle_reference(val: pd.DataFrame, full: pd.DataFrame) -> dict:
    return {"val exacto": oracle(val)["score_oficial"], "año completo": oracle(full)["score_oficial"]}


def timing(agent_cls, val: pd.DataFrame, hours: int = 2016) -> float:
    """Segundos estimados para el test oficial (2.016 h), extrapolando desde val."""
    r = run(agent_cls, val)
    return r["segundos"] * hours / len(val)


__all__ = ["PerturbedEnv", "run", "weekly", "scenarios", "oracle_reference", "timing", "load_agent_class"]
