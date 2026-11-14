"""Fase 1 · Construye las tablas horarias y comprueba los criterios de aceptación de los datos.

    .venv/bin/python scripts/00_build_data.py [--rebuild]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dd.data import fit_pv_irr, load_full, load_sim  # noqa: E402
from dd.paths import MODES, N_IDX  # noqa: E402

PREV_HOURLY = Path("/home/marc/Projects/Datathon2026/data/processed/hourly.csv")


def check(cond: bool, msg: str) -> None:
    print(("  OK   " if cond else "  FALLA ") + msg)
    if not cond:
        check.failed = True


check.failed = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    full = load_full(args.rebuild)
    sim = load_sim(args.rebuild)

    print(f"hourly_full: {len(full)} filas, {full.index[0]} → {full.index[-1]}")
    print("  horas finitas por variable:", {c: int(full[c].notna().sum()) for c in full.columns})
    print(f"hourly_sim: {len(sim)} filas, {sim.index[0]} → {sim.index[-1]}")
    check(len(full) == N_IDX, f"hourly_full tiene {N_IDX} filas")
    check(len(sim) == 5993, "hourly_sim tiene 5.993 filas")
    check(sim.index[0] == pd.Timestamp("2025-09-01 00:00") and sim.index[-1] == pd.Timestamp("2026-08-16 22:00"),
          "hourly_sim va de 2025-09-01 00:00 a 2026-08-16 22:00")

    if PREV_HOURLY.exists():
        prev = pd.read_csv(PREV_HOURLY, parse_dates=["Fecha"], index_col="Fecha")
        cols = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]
        same_idx = prev.index.equals(sim.index)
        diff = float(np.nanmax(np.abs(prev[cols].to_numpy() - sim[cols].to_numpy()))) if same_idx else np.inf
        check(same_idx and diff < 1e-9, f"idéntica a la tabla horaria del repo previo (máx |Δ| = {diff:.2e})")

    expected = {"std": (551823, 939634, 0.2517), "hard": (468035, 766495, 0.2617)}
    for mode, (c_ref, e_ref, lam_ref) in expected.items():
        S = MODES[mode]["S"]
        base = (sim.load_kw - S * sim.pv_kw).clip(lower=0)
        C, E = float(base @ sim.co2_kg_per_kwh), float(base @ sim.precio_eur_per_kwh)
        lam = 0.3 / 0.7 * C / E
        check(abs(C / c_ref - 1) < 1e-3 and abs(E / e_ref - 1) < 1e-3 and abs(lam - lam_ref) < 5e-4,
              f"{mode}: CO2 base {C:,.0f} kg, coste base {E:,.0f} €, λ = {lam:.4f}")
        net = sim.load_kw - S * sim.pv_kw
        print(f"         carga neta ≤ 0 en {100 * (net <= 0).mean():.1f} % de las horas; "
              f"< potencia máx. en {100 * (net < MODES[mode]['pmax']).mean():.1f} %")

    h = full.load
    d = (h - h.shift(1)).abs()
    night = d[h.index.hour.isin([1, 2, 3, 4, 5])].dropna()
    check(float(night.mean()) == 0.0, f"contador congelado: MAE persistencia 1–5 h = {night.mean():.3f} kW")
    print("  MAE persistencia por hora:", d.groupby(h.index.hour).mean().round(1).to_dict())

    coef = fit_pv_irr(full)
    dd_ = full[["pv", "irr", "tpan"]].dropna()
    dd_ = dd_[dd_.irr > 20]
    pred = coef[0] * dd_.irr + coef[1] * dd_.irr * (dd_.tpan - 25) + coef[2] * dd_.irr ** 2
    r2 = 1 - ((pred - dd_.pv) ** 2).sum() / ((dd_.pv - dd_.pv.mean()) ** 2).sum()
    check(r2 > 0.98, f"solar ≈ {coef[0]:.4f}·irr {coef[1]:+.2e}·irr·(tpan−25) {coef[2]:+.2e}·irr² (R² {r2:.4f})")
    imp = full.pv_mem.notna() & full.pv.isna()
    print(f"  horas de solar imputadas para la memoria: {int(imp.sum())} "
          f"({full.index[imp].min()} → {full.index[imp].max()})")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
