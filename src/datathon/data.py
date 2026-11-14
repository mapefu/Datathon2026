"""Carga y preprocesado de los datos de entrenamiento a tabla horaria.

Reproduce el preprocesado del tutorial oficial (starter_kit/tutorial.ipynb) para
`load_kw` y `pv_kw`, que es lo que consume el simulador, y añade variables
meteorológicas útiles para modelos de predicción.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "train"
PROCESSED_DIR = ROOT / "data" / "processed"
HOURLY_PATH = PROCESSED_DIR / "hourly.csv"

SIM_COLS = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]

_LOAD_COL = "Contador - POTENCIA ACTIVA EXPORTADA (kW)"
_PV_COLS = [f"Inversor {i} - POTENCIA ACTIVA (kW)" for i in (1, 2, 3)]
_METEO_COLS = {
    "Atersa - IRRADIANCIA (W/m2)": "irradiancia_wm2",
    "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)": "temp_amb_c",
    "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)": "temp_panel_c",
    "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)": "viento_ms",
}


def _read(name: str, usecols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / name, usecols=["Fecha", *usecols], parse_dates=["Fecha"])
    return df.set_index("Fecha").sort_index()


def build_hourly() -> pd.DataFrame:
    """Construye la tabla horaria desde los CSV crudos de 5 minutos."""
    contador = _read("contador.csv", [_LOAD_COL])
    load = (-contador[_LOAD_COL]).clip(lower=0, upper=5000).resample("1h").mean()

    inversor = _read("inversor.csv", _PV_COLS)
    pv = inversor[_PV_COLS].fillna(0).sum(axis=1).clip(lower=0).resample("1h").mean()

    meteo = _read("meteo.csv", list(_METEO_COLS)).rename(columns=_METEO_COLS)
    meteo = meteo.resample("1h").mean()

    pvpc = pd.read_csv(RAW_DIR / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(RAW_DIR / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

    df = pd.DataFrame(
        {
            "load_kw": load,
            "pv_kw": pv,
            "precio_eur_per_kwh": pvpc["precio_eur_per_kwh"],
            "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
        }
    )
    # Igual que el tutorial: solo horas con las 4 variables del simulador.
    df = df.dropna(subset=SIM_COLS)
    df = df.join(meteo, how="left")
    df.index.name = "Fecha"
    return df


def load_hourly(rebuild: bool = False) -> pd.DataFrame:
    """Devuelve la tabla horaria, usando la caché de data/processed si existe."""
    if HOURLY_PATH.exists() and not rebuild:
        return pd.read_csv(HOURLY_PATH, parse_dates=["Fecha"], index_col="Fecha")
    df = build_hourly()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HOURLY_PATH)
    return df


def validation_mask(df: pd.DataFrame, days: int = 7) -> pd.Series:
    """Últimos `days` días disponibles de cada mes.

    El test oculto son los últimos 7 días de cada mes (excluidos de train), así
    que los 7 días inmediatamente anteriores son la mejor imitación local.
    """
    dates = df.index.normalize()
    month = df.index.to_period("M")
    last_day = pd.Series(dates, index=df.index).groupby(month).transform("max")
    return pd.Series(dates >= (last_day - pd.Timedelta(days=days - 1)).values, index=df.index)


def split(df: pd.DataFrame, days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val) con val = últimos `days` días de cada mes."""
    m = validation_mask(df, days)
    return df[~m], df[m]
