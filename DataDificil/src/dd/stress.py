"""Observaciones degradadas para las pruebas de robustez (la física y el score no cambian)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .env import METEO_KEYS, SimEnv


class StressEnv(SimEnv):
    def __init__(self, df, mode, nan_prob=0.0, str_timestamp=False, as_array=False, bad_obs_at=None,
                 drop_soc=False, drop_spec=False, drop_meteo=False, full_lists=False, shift_years=0, seed=0):
        self.nan_prob, self.str_timestamp, self.as_array = nan_prob, str_timestamp, as_array
        self.bad_obs_at, self.drop_soc, self.drop_spec = bad_obs_at, drop_soc, drop_spec
        self.drop_meteo, self.full_lists, self.shift_years = drop_meteo, full_lists, shift_years
        self.rng = np.random.default_rng(seed)
        super().__init__(df, mode)
        if full_lists:
            self.K = 24

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        for key in ("preus_properes_24h", "co2_properes_24h"):
            v = np.asarray(obs[key], dtype=float)
            if self.nan_prob:
                v[self.rng.random(len(v)) < self.nan_prob] = np.nan
            obs[key] = v if self.as_array else v.tolist()
        d = obs["noves_dades_5min"]
        if self.nan_prob:
            for k in list(d):
                if self.rng.random() < self.nan_prob:
                    d[k] = float("nan")
        if self.drop_meteo:
            for k in METEO_KEYS.values():
                d.pop(k, None)
        if self.drop_spec:
            obs.pop("cap_max_kwh", None)
            obs.pop("power_max_kw", None)
        if self.bad_obs_at == self.t:
            obs["noves_dades_5min"] = [1.0]
            obs["preus_properes_24h"] = "x"
        if self.drop_soc:
            obs.pop("soc_kwh", None)
        if self.shift_years:
            obs["timestamp"] = obs["timestamp"] + pd.DateOffset(years=self.shift_years)
        if self.str_timestamp:
            obs["timestamp"] = str(obs["timestamp"])
        return obs
