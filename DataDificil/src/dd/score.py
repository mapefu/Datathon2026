"""Score oficial, estadísticas por semana, comparación emparejada y oráculo LP.

El score de un conjunto de semanas es el oficial de `environment.py` sobre el agregado (base común):
    Score = max(0, 70·ΔCO2 + 30·ΔCoste − 0,2·infracciones)   [en puntos, 0–100]
Se informa también la variante del enunciado (−1,0 por infracción); el objetivo es 0 infracciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .paths import ETA, MODES

PEN_ENV, PEN_DOC = 0.2, 1.0


def week_stats(grid, g: pd.DataFrame, infr: int, seconds: float) -> dict:
    """g: tabla del modo (solar ya escalada)."""
    base = np.maximum((g["load_kw"] - g["pv_kw"]).to_numpy(), 0.0)
    p, c = g["precio_eur_per_kwh"].to_numpy(), g["co2_kg_per_kwh"].to_numpy()
    grid = np.asarray(grid, dtype=float)
    return {"gc": float(grid @ c), "gp": float(grid @ p), "bc": float(base @ c), "bp": float(base @ p),
            "infr": int(infr), "s": float(seconds), "n": int(len(g))}


def aggregate(stats: dict) -> dict:
    stats = {k: v for k, v in stats.items() if not k.startswith("_")}
    s = {k: sum(v[k] for v in stats.values()) for k in ("gc", "gp", "bc", "bp", "infr", "s", "n")}
    dco2 = 100 * (s["bc"] - s["gc"]) / s["bc"]
    dcost = 100 * (s["bp"] - s["gp"]) / s["bp"]
    raw = 0.7 * dco2 + 0.3 * dcost
    return {"score": max(0.0, raw - PEN_ENV * s["infr"]), "score_doc": max(0.0, raw - PEN_DOC * s["infr"]),
            "dco2": dco2, "dcost": dcost, "infr": s["infr"], "ms_h": 1000 * s["s"] / max(s["n"], 1),
            "weeks": len(stats), "hours": s["n"]}


def contributions(stats: dict) -> dict:
    """Aportación de cada semana al score agregado (suman el score sin penalización)."""
    stats = {k: v for k, v in stats.items() if not k.startswith("_")}
    BC = sum(v["bc"] for v in stats.values())
    BP = sum(v["bp"] for v in stats.values())
    return {k: 70 * (v["bc"] - v["gc"]) / BC + 30 * (v["bp"] - v["gp"]) / BP for k, v in stats.items()}


def paired(a: dict, b: dict) -> dict:
    """Diferencia emparejada por semana (a − b) de la aportación al score agregado de a ∪ b, más la
    diferencia de los scores agregados."""
    ks = sorted(set(k for k in a if not k.startswith("_")) & set(b))
    ca, cb = contributions({k: a[k] for k in ks}), contributions({k: b[k] for k in ks})
    d = np.array([ca[k] - cb[k] for k in ks])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"sum": float(d.sum()), "mean": float(d.mean()), "se_sum": se * len(d), "se": se,
            "wins": float((d > 0).mean()), "n": len(d),
            "agg": aggregate({k: a[k] for k in ks})["score"] - aggregate({k: b[k] for k in ks})["score"],
            "accept": bool(d.mean() > 2 * se and (d > 0).mean() >= 0.6)}


# ----------------------------------------------------------------------------- oráculo
def lp_plan(net: np.ndarray, w: np.ndarray, cap: float, pmax: float, soc0: float) -> tuple[np.ndarray, np.ndarray]:
    """min Σ w·g  s.a. g ≥ net + c − d, g ≥ 0, s_k = s_{k−1} + η·c_k − d_k/η, 0 ≤ s ≤ cap, 0 ≤ c,d ≤ pmax.
    Devuelve (acción c − d, importación g)."""
    n = len(net)
    I = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((n, n))
    D = I - sparse.eye(n, k=-1, format="csr")
    A_ub = sparse.hstack([I, -I, -I, Z], format="csr")
    A_eq = sparse.hstack([-ETA * I, I / ETA, Z, D], format="csr")
    b_eq = np.zeros(n)
    b_eq[0] = soc0
    cost = np.concatenate([np.zeros(2 * n), w, np.zeros(n)])
    bounds = [(0, pmax)] * (2 * n) + [(0, None)] * n + [(0, cap)] * n
    sol = linprog(cost, A_ub=A_ub, b_ub=-net, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x[:n] - sol.x[n:2 * n], sol.x[2 * n:3 * n]


def oracle(weeks_by_fold: dict[str, list[pd.DataFrame]], mode: str) -> dict:
    """Información perfecta. P1: por semana con SOC al 50 %; P2: semanas de cada pliegue seguidas con SOC
    continuo. Pesos con la base agregada de todas las semanas (el óptimo no depende de la escala)."""
    from .env import sim_frame, week_id

    m = MODES[mode]
    cap, pmax, soc0 = m["cap"], m["pmax"], 0.5 * m["cap"]
    allw = pd.concat([sim_frame(g, mode) for gs in weeks_by_fold.values() for g in gs])
    base = np.maximum((allw.load_kw - allw.pv_kw).to_numpy(), 0.0)
    BC, BP = float(base @ allw.co2_kg_per_kwh.to_numpy()), float(base @ allw.precio_eur_per_kwh.to_numpy())

    def w_of(g):
        return 70 * g.co2_kg_per_kwh.to_numpy() / BC + 30 * g.precio_eur_per_kwh.to_numpy() / BP

    res = {"P1": {}, "P2": {}}
    for gs in weeks_by_fold.values():
        ws = [sim_frame(g, mode) for g in gs]
        for g in ws:
            _, grid = lp_plan((g.load_kw - g.pv_kw).to_numpy(), w_of(g), cap, pmax, soc0)
            res["P1"][week_id(g)] = week_stats(grid, g, 0, 0.0)
        c = pd.concat(ws)
        _, grid = lp_plan((c.load_kw - c.pv_kw).to_numpy(), w_of(c), cap, pmax, soc0)
        i = 0
        for g in ws:
            res["P2"][week_id(g)] = week_stats(grid[i:i + len(g)], g, 0, 0.0)
            i += len(g)
    return res
