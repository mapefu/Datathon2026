"""Error de previsión fuera de muestra en las semanas de un pliegue (modelos entrenados sin ellas).

La memoria es la verdad completa, pero `build_features` solo lee sensores hasta t−1 y mercado hasta
t+2: es exactamente lo que el agente habrá observado (train antes del bloque + telemetría del bloque).
El calendario y las climatologías son las del bundle del pliegue (sin las semanas evaluadas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import fold_dir, fold_weeks
from .data import load_full, load_sim
from .dataset import agent_module, memory
from .paths import idx_of

BUCKETS = {"h0": (0, 1), "h1-2": (1, 3), "h3": (3, 4), "h3-7": (3, 8), "h8-23": (8, 24), "h3-23": (3, 24)}


def evaluate(fold: str, tag: str = "default", mode: str = "hard") -> dict:
    ag = agent_module()
    _, C, arrs, params = ag.load_bundle(fold_dir(fold, tag))
    forest = ag.Forest(arrs, params["models"])
    full, sim = load_full(), load_sim()
    M = memory(full)
    S = 12.0 if mode == "hard" else 1.0
    zt, s0 = arrs["Z_" + mode], arrs["S0_" + mode]
    rec = []
    for g in fold_weeks(sim, fold):
        for ts in g.index:
            t = idx_of(ts)
            X = ag.build_features(M, C, t, 3, meteo=(mode == "hard"))
            f = ag.point_forecast(forest, X, C, M, t, market=True)
            sig = np.maximum(forest.predict(X, [f"sigma_{mode}"])[f"sigma_{mode}"], 5.0)
            hs = np.arange(24)
            tau = t + hs
            y = {k: np.array([full[k].iat[i] if i < len(full) else np.nan for i in tau]) for k in ("load", "pv", "price", "co2")}
            Xb = X[:24]
            base_load = np.where(np.isfinite(Xb[:, ag.FI["L_same"]]), Xb[:, ag.FI["L_same"]], Xb[:, ag.FI["L24"]])
            k1 = Xb[:, ag.FI["k1"]]
            base_pv = np.where(np.isfinite(k1), k1, 0.6) * Xb[:, ag.FI["cs_tau"]]
            for h in hs:
                net_true = y["load"][h] - S * y["pv"][h]
                net_hat = f["load"][h] - S * f["pv"][h]
                day = Xb[h, ag.FI["cs_tau"]] > 0
                r = net_true - net_hat
                rec.append({
                    "week": str(g.index[0].date()), "t": t, "h": h, "exact": bool(f["exact"][h]),
                    "e_net": r, "e_net_base": net_true - (base_load[h] - S * base_pv[h]),
                    "e_load": y["load"][h] - f["load"][h], "e_pv": S * (y["pv"][h] - f["pv"][h]),
                    "e_price": y["price"][h] - f["price"][h] if h >= 3 else np.nan,
                    "e_price_base": y["price"][h] - Xb[h, ag.FI["p_xfer_yd"]] if h >= 3 else np.nan,
                    "e_co2": y["co2"][h] - f["co2"][h] if h >= 3 else np.nan,
                    "e_co2_base": y["co2"][h] - Xb[h, ag.FI["c_xfer_yd"]] if h >= 3 else np.nan,
                    "u": r / sig[h] if np.isfinite(r) else np.nan, "reg": int(day),
                })
    df = pd.DataFrame(rec)
    out = {"fold": fold, "mode": mode, "rows": len(df)}
    for b, (lo, hi) in BUCKETS.items():
        d = df[(df.h >= lo) & (df.h < hi)]
        out[b] = {k: float(d[k].abs().mean()) for k in ("e_net", "e_net_base", "e_load", "e_pv", "e_price",
                                                        "e_price_base", "e_co2", "e_co2_base")}
    d = df[np.isfinite(df.u) & ~df.exact]
    cov = []
    for j in range(9):
        thr = np.where(d.reg == 1, zt[1, j], zt[0, j])
        cov.append(float(np.mean(d.u <= thr)))
    out["coverage"] = cov
    out["exact_rows"] = int(df.exact.sum())
    out["exact_mae"] = float(df[df.exact].e_net.abs().mean()) if df.exact.any() else None
    return out
