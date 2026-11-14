"""CSV de 5 min → tablas horarias.

- `hourly_full`: todas las horas de 2025-09-01 00:00 a 2026-09-01 23:00 (8.808), unión externa con NaN
  por variable. Es la memoria que viaja en el bundle del agente (incluye noviembre y agosto 17–24, donde
  hay mercado y/o contador aunque falte la solar).
- `hourly_sim`: filas con las 4 variables del simulador finitas (lo que hace el organizador con dropna),
  con los nombres de columna oficiales. 5.993 filas.

Las fórmulas de consumo y solar son las del tutorial oficial: `(-Contador).clip(0, 5000)` y suma de las
columnas `POTENCIA ACTIVA` sin `FASE` de los inversores, `.clip(lower=0)`, `resample("1h").mean()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import BASE, END, N_IDX, PROC, RAW

LOAD_COL = "Contador - POTENCIA ACTIVA EXPORTADA (kW)"
METEO_COLS = {
    "Atersa - IRRADIANCIA (W/m2)": "irr",
    "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)": "tamb",
    "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)": "tpan",
}
FULL_PATH = PROC / "hourly_full.csv"
SIM_PATH = PROC / "hourly_sim.csv"
SIM_COLS = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]


def full_index() -> pd.DatetimeIndex:
    idx = pd.date_range(BASE, END, freq="h", name="Fecha")
    assert len(idx) == N_IDX
    return idx


def _read(name: str, usecols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RAW / name, usecols=["Fecha", *usecols], parse_dates=["Fecha"])
    return df.set_index("Fecha").sort_index()


def build_full() -> pd.DataFrame:
    idx = full_index()
    cont = _read("contador.csv", [LOAD_COL])
    load = (-cont[LOAD_COL]).clip(lower=0, upper=5000).resample("1h").mean()

    inv_head = pd.read_csv(RAW / "inversor.csv", nrows=0).columns
    inv_cols = [c for c in inv_head if "POTENCIA ACTIVA" in c and "FASE" not in c]   # filtro oficial
    inv = _read("inversor.csv", inv_cols)
    pv = inv[inv_cols].sum(axis=1).clip(lower=0).resample("1h").mean()

    met = _read("meteo.csv", list(METEO_COLS)).rename(columns=METEO_COLS).resample("1h").mean()

    pvpc = pd.read_csv(RAW / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(RAW / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

    full = pd.DataFrame(index=idx)
    full["load"] = load.reindex(idx)
    full["pv"] = pv.reindex(idx)
    for c in ("irr", "tamb", "tpan"):
        full[c] = met[c].reindex(idx)
    full["price"] = pvpc["precio_eur_per_kwh"].reindex(idx)
    full["co2"] = co2["co2_kg_per_kwh"].reindex(idx)
    full["pv_mem"] = impute_pv(full)
    return full


def fit_pv_irr(full: pd.DataFrame) -> np.ndarray:
    """pv ≈ a·irr + b·irr·(tpan − 25) + c·irr² (escala ×1). Mínimos cuadrados con irr > 20."""
    d = full[["pv", "irr", "tpan"]].dropna()
    d = d[d.irr > 20]
    A = np.c_[d.irr, d.irr * (d.tpan - 25.0), d.irr ** 2]
    coef, *_ = np.linalg.lstsq(A, d.pv.to_numpy(), rcond=None)
    return coef


def impute_pv(full: pd.DataFrame) -> pd.Series:
    """Copia de la solar para la memoria del agente: los huecos de inversor con irradiancia medida
    (16/08 23:00 – 24/08) se rellenan con el ajuste físico, recortado a [0, 150] kW."""
    coef = fit_pv_irr(full)
    tp = full.tpan.fillna(25.0)
    est = (coef[0] * full.irr + coef[1] * full.irr * (tp - 25.0) + coef[2] * full.irr ** 2).clip(0, 150)
    est = est.where(full.irr > 0, 0.0).where(full.irr.notna())
    return full.pv.where(full.pv.notna(), est)


def to_sim(full: pd.DataFrame) -> pd.DataFrame:
    sim = pd.DataFrame({
        "load_kw": full["load"], "pv_kw": full["pv"],
        "precio_eur_per_kwh": full["price"], "co2_kg_per_kwh": full["co2"],
        "irradiancia_wm2": full["irr"], "temp_amb_c": full["tamb"], "temp_panel_c": full["tpan"],
        "viento_ms": 0.0,
    }, index=full.index)
    sim = sim.dropna(subset=SIM_COLS)
    for c in ("irradiancia_wm2", "temp_amb_c", "temp_panel_c"):
        sim[c] = sim[c].fillna(0.0)
    return sim


def load_full(rebuild: bool = False) -> pd.DataFrame:
    if FULL_PATH.exists() and not rebuild:
        return pd.read_csv(FULL_PATH, parse_dates=["Fecha"], index_col="Fecha")
    full = build_full()
    PROC.mkdir(parents=True, exist_ok=True)
    full.to_csv(FULL_PATH)
    return full


def load_sim(rebuild: bool = False) -> pd.DataFrame:
    if SIM_PATH.exists() and not rebuild:
        return pd.read_csv(SIM_PATH, parse_dates=["Fecha"], index_col="Fecha")
    sim = to_sim(load_full(rebuild))
    sim.to_csv(SIM_PATH)
    return sim
