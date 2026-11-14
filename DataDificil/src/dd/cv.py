"""Validación cruzada sin fuga.

Semanas pseudo-test por mes con datos de simulación: pliegue A = últimos 7 días disponibles, pliegue B =
los 7 anteriores (la misma definición que el CV previo del equipo, así los números son comparables).
Para el pliegue X, todas las variables de sus semanas se ponen a NaN en `hourly_full` antes de entrenar
y de construir la memoria del agente: quedan como un hueco real, igual que el test en el servidor.

Pliegues de robustez (no deciden parámetros):
- COLD: semanas A de octubre, enero y mayo, sin carga/solar/meteo de los 30 días anteriores (≈ noviembre).
- NOPV: semana A de julio, con la solar de los 8 días anteriores sustituida por la imputación desde la
  irradiancia (≈ final de agosto).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import FOLDS_DIR, REPORTS, idx_of

FOLDS = ("A", "B")
SENSORS = ("load", "pv", "pv_mem", "irr", "tamb", "tpan")
ALL_VARS = SENSORS + ("price", "co2")


def cv_weeks(sim: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    out = {"A": [], "B": []}
    for _, g in sim.groupby(sim.index.to_period("M")):
        days = g.index.normalize()
        last = days.max()
        late = g[days >= last - pd.Timedelta(days=6)]
        mid = g[(days >= last - pd.Timedelta(days=13)) & (days <= last - pd.Timedelta(days=7))]
        if len(late) >= 120:
            out["A"].append(late)
        if len(mid) >= 120:
            out["B"].append(mid)
    return out


def robustness_weeks(sim: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    A = {str(g.index[0].to_period("M")): g for g in cv_weeks(sim)["A"]}
    return {"COLD": [A["2025-10"], A["2026-01"], A["2026-05"]], "NOPV": [A["2026-07"]]}


def week_span(g: pd.DataFrame) -> tuple[int, int]:
    """(idx inicial, idx final) del bloque, ambos incluidos, en calendario continuo."""
    return idx_of(g.index[0].normalize()), idx_of(g.index[-1].normalize() + pd.Timedelta(hours=23))


def fold_weeks(sim: pd.DataFrame, fold: str) -> list[pd.DataFrame]:
    if fold in FOLDS:
        return cv_weeks(sim)[fold]
    return robustness_weeks(sim)[fold]


def masked_full(full: pd.DataFrame, sim: pd.DataFrame, fold: str) -> pd.DataFrame:
    """`hourly_full` sin la información de las semanas del pliegue (y, en COLD/NOPV, lo extra)."""
    out = full.copy()
    if fold == "full":
        return out
    weeks = fold_weeks(sim, fold)
    for g in weeks:
        a, b = week_span(g)
        out.iloc[a:b + 1, [out.columns.get_loc(c) for c in ALL_VARS]] = np.nan
        if fold == "COLD":
            out.iloc[max(0, a - 30 * 24):a, [out.columns.get_loc(c) for c in SENSORS]] = np.nan
        if fold == "NOPV":
            lo = max(0, a - 8 * 24)
            out.iloc[lo:a, out.columns.get_loc("pv")] = np.nan     # pv_mem conserva la imputación
            from .data import fit_pv_irr

            coef = fit_pv_irr(out)
            seg = out.iloc[lo:a]
            est = (coef[0] * seg.irr + coef[1] * seg.irr * (seg.tpan.fillna(25) - 25) + coef[2] * seg.irr ** 2)
            out.iloc[lo:a, out.columns.get_loc("pv_mem")] = est.clip(0, 150).where(seg.irr > 0, 0.0).to_numpy()
    return out


def blocks_of(weeks: list[pd.DataFrame]) -> list[tuple[int, int]]:
    return [week_span(g) for g in weeks]


def fold_dir(fold: str, tag: str = "default") -> Path:
    return FOLDS_DIR / tag / fold


# ----------------------------------------------------------------------------- ejecución
def _job(args):
    from .env import run_p1, run_p2

    proto, agent_path, weeks, mode, overrides = args
    if proto == "P1":
        out = {}
        for g in weeks:
            ov = dict(overrides, BLOCKS=blocks_of([g]))
            out.update(run_p1(agent_path, [g], mode, ov))
        return out
    return run_p2(agent_path, weeks, mode, dict(overrides, BLOCKS=blocks_of(weeks)))


def run_cv(name: str, agent_path, mode: str, sim: pd.DataFrame, bundle_tag: str = "default",
           folds=FOLDS, protocols=("P1", "P2"), overrides: dict | None = None, workers: int = 4,
           save: bool = True) -> dict:
    """Ejecuta el agente en P1/P2 con los modelos de cada pliegue (overrides BUNDLE_DIR y BLOCKS)."""
    overrides = dict(overrides or {})
    res = {"name": name, "mode": mode, "agent": str(agent_path), "bundle_tag": bundle_tag,
           "overrides": {k: v for k, v in overrides.items() if isinstance(v, (int, float, str, bool))}}
    jobs = []
    for f in folds:
        ov = dict(overrides)
        d = fold_dir(f, bundle_tag)
        if d.exists():
            ov["BUNDLE_DIR"] = str(d)
        weeks = fold_weeks(sim, f)
        if "P1" in protocols:
            # P1 se reparte en trozos de 3 semanas para paralelizar
            for i in range(0, len(weeks), 3):
                jobs.append(("P1", (("P1", agent_path, weeks[i:i + 3], mode, ov))))
        if "P2" in protocols:
            jobs.append(("P2", ("P2", agent_path, weeks, mode, ov)))
    t0 = time.time()
    for p in protocols:
        res[p] = {}
    if workers <= 1:
        for p, a in jobs:
            res[p].update(_job(a))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [(p, ex.submit(_job, a)) for p, a in jobs]
            for p, fu in futs:
                res[p].update(fu.result())
    res["wall_s"] = time.time() - t0
    if save:
        out = REPORTS / "cv"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{mode}_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def load_result(mode: str, name: str) -> dict:
    return json.loads((REPORTS / "cv" / f"{mode}_{name}.json").read_text(encoding="utf-8"))


def report(res: dict, ref: dict | None = None, orc: dict | None = None) -> str:
    from .score import aggregate, paired

    lines = []
    for p in ("P1", "P2"):
        if p not in res:
            continue
        a = aggregate(res[p])
        line = (f"{res.get('name', '?'):>18s} {p}: score {a['score']:7.3f}  ΔCO2 {a['dco2']:6.2f}%  "
                f"ΔCoste {a['dcost']:6.2f}%  infr {a['infr']}  {a['ms_h']:.2f} ms/h  ({a['weeks']} sem)")
        if orc is not None and p in orc:
            line += f"  | oráculo {aggregate({k: orc[p][k] for k in res[p] if k in orc[p]})['score']:.3f}"
        if ref is not None and p in ref and ref is not res:
            d = paired(res[p], ref[p])
            line += (f"\n{'':>22s}vs {ref.get('name')}: Δagregado {d['agg']:+.3f}  Δ(suma semanas) "
                     f"{d['sum']:+.3f} ± {d['se_sum']:.3f}  gana {100 * d['wins']:.0f} % de {d['n']}  "
                     f"→ {'ACEPTAR' if d['accept'] else 'rechazar'}")
        lines.append(line)
    return "\n".join(lines)
