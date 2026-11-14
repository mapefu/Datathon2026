"""Calendario del campus y climatologías precalculadas para cada índice horario.

Fijas (no dependen de los datos): hora, día de la semana, tipo de día, nivel de actividad, periodo
tarifario 2.0TD, desfase UTC y horas de noche con el contador congelado.
Dependientes del pliegue (se calculan solo con los datos de train visibles): solar de cielo despejado
empírica `cs`, nivel típico de carga `lvl_load` y escalones tarifarios `toll`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import BASE, N_CAL, N_IDX

# Festivos nacionales de fecha fija: sin escalones de tarifa (todo el día en P3).
NATIONAL = {"2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25",
            "2026-01-01", "2026-01-06", "2026-05-01", "2026-08-15"}
# Festivos del campus (tipo de día 2): nacionales + autonómicos/locales + cierres puntuales.
CAMPUS_HOLIDAYS = NATIONAL | {"2025-09-11", "2025-09-24", "2025-12-26", "2026-04-03", "2026-04-06",
                              "2026-05-25", "2026-06-24"}
CLOSURES = [("2025-12-24", "2026-01-06"), ("2026-08-01", "2026-08-23")]      # actividad 0,3
REDUCED = [("2025-12-22", "2025-12-23"), ("2026-03-30", "2026-04-02")]       # actividad 0,6
NIGHT_HOURS = (23, 0, 1, 2, 3, 4, 5)


def _in_ranges(dates: pd.DatetimeIndex, ranges) -> np.ndarray:
    out = np.zeros(len(dates), bool)
    for a, b in ranges:
        out |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return out


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.dayofweek + 1) % 7)


def utc_offset(ts: pd.DatetimeIndex) -> np.ndarray:
    """+2 desde el último domingo de marzo 02:00 hasta el último de octubre 03:00 (hora local)."""
    off = np.ones(len(ts), np.int64)
    for y in np.unique(ts.year):
        a = _last_sunday(y, 3) + pd.Timedelta(hours=2)
        b = _last_sunday(y, 10) + pd.Timedelta(hours=3)
        off[(ts >= a) & (ts < b)] = 2
    return off


def static_calendar() -> dict[str, np.ndarray]:
    ts = pd.date_range(BASE, periods=N_CAL, freq="h")
    dates = ts.normalize()
    ds = dates.strftime("%Y-%m-%d")
    hour, dow = ts.hour.to_numpy(), ts.day_of_week.to_numpy()
    nat = np.isin(ds, list(NATIONAL))
    hol = np.isin(ds, list(CAMPUS_HOLIDAYS))
    daytype = np.where(hol | (dow == 6), 2, np.where(dow == 5, 1, 0))
    act = np.where(_in_ranges(dates, CLOSURES), 0.3, np.where(_in_ranges(dates, REDUCED), 0.6, 1.0))
    weekend = (dow >= 5) | nat
    period = np.where(weekend | (hour < 8), 3,
                      np.where(((hour >= 10) & (hour < 14)) | ((hour >= 18) & (hour < 22)), 1, 2))
    off = utc_offset(ts)
    day = ((ts - BASE) // pd.Timedelta(days=1)).to_numpy().astype(np.int64)
    utc_abs = np.arange(N_CAL) - off + 24                     # ≥ 0
    return {
        "hour": hour.astype(np.float64), "dow": dow.astype(np.float64),
        "daytype": daytype.astype(np.float64), "act": act.astype(np.float64),
        "period": period.astype(np.float64), "night": np.isin(hour, NIGHT_HOURS).astype(np.float64),
        "utc_off": off.astype(np.float64), "hour_utc": (utc_abs % 24).astype(np.float64),
        "day": day.astype(np.float64), "year": ts.year.to_numpy().astype(np.float64),
        "month": ts.month.to_numpy().astype(np.float64),
    }


def clear_sky(pv_x1: np.ndarray, cal: dict) -> np.ndarray:
    """Solar de cielo despejado empírica (kW, escala ×1) para los N_CAL índices.

    Por hora UTC entera (absorbe el cambio de hora): envolvente de la serie diaria = máximo móvil de ±10
    días → mediana móvil de 15 → media móvil de 15, interpolada en los días sin datos tratando el año como
    circular."""
    off = cal["utc_off"].astype(np.int64)
    utc_abs = np.arange(N_CAL) - off + 24
    uday, uhour = utc_abs // 24, utc_abs % 24
    D = int(uday.max()) + 1
    grid = np.full((24, D), np.nan)
    vals = np.full(N_CAL, np.nan)
    vals[:N_IDX] = pv_x1
    ok = np.isfinite(vals)
    grid[uhour[ok], uday[ok]] = vals[ok]
    env = np.zeros((24, D))
    for u in range(24):
        x = pd.Series(np.concatenate([grid[u]] * 3))
        if x.notna().sum() == 0:
            continue
        e = x.rolling(21, center=True, min_periods=3).max()
        e = e.rolling(15, center=True, min_periods=3).median()
        e = e.rolling(15, center=True, min_periods=3).mean()
        e = e.interpolate(limit_direction="both")
        env[u] = e.to_numpy()[D:2 * D]
    cs = np.clip(env[uhour, uday], 0.0, None)
    cs[cs < 0.5] = 0.0
    return cs


def day_max(x: np.ndarray, cal: dict) -> np.ndarray:
    """Máximo de `x` dentro del mismo día natural, para cada índice."""
    s = pd.Series(x).groupby(cal["day"].astype(np.int64)).transform("max")
    return s.to_numpy()


def level_load(load: np.ndarray, cal: dict, radius: int = 35) -> np.ndarray:
    """Mediana de la carga por (tipo de día, actividad normal/reducida, hora) en los días de train a ±35
    días (distancia circular de un año), excluyendo el propio día. Respaldos: sin radio y solo hora."""
    vals = np.full(N_CAL, np.nan)
    vals[:N_IDX] = load
    day = cal["day"].astype(np.int64)
    normal = cal["act"] >= 0.99
    keys = cal["daytype"].astype(np.int64) * 100 + normal * 50 + cal["hour"].astype(np.int64)
    out = np.full(N_CAL, np.nan)
    for k in np.unique(keys):
        members = np.flatnonzero(keys == k)
        src = members[np.isfinite(vals[members])]
        for i in members:
            dd = np.abs(day[src] - day[i]) % 365
            dd = np.minimum(dd, 365 - dd)
            sel = src[(dd <= radius) & (day[src] != day[i])]
            if len(sel) == 0:
                sel = src[day[src] != day[i]]
            if len(sel):
                out[i] = float(np.median(vals[sel]))
    # respaldo final: mediana por hora del día
    bad = ~np.isfinite(out)
    if bad.any():
        prof = pd.Series(vals).groupby(cal["hour"].astype(np.int64)).median()
        out[bad] = prof.reindex(cal["hour"][bad].astype(np.int64)).to_numpy()
    return out


def tolls(price: np.ndarray, cal: dict) -> np.ndarray:
    """Escalón tarifario por (año − 2025, periodo 1..3) en €/kWh, estimado por diferencias en
    diferencias: salto de precio en las fronteras de periodo en laborables menos el mismo salto en fines
    de semana. P3 = 0."""
    out = np.zeros((2, 4))
    p = np.full(N_CAL, np.nan)
    p[:N_IDX] = price
    dp = np.r_[np.nan, np.diff(p)]
    hour = cal["hour"].astype(np.int64)
    wk = cal["period"][np.clip(np.arange(N_CAL) - hour + 12, 0, N_CAL - 1)] != 3   # el día es laborable
    for yi, y in enumerate((2025, 2026)):
        ysel = cal["year"] == y

        def did(h):
            a = dp[ysel & wk & (hour == h)]
            b = dp[ysel & ~wk & (hour == h)]
            return np.nanmean(a) - np.nanmean(b)

        d12 = np.nanmean([did(10), -did(14), did(18), -did(22)])
        t2 = np.nanmean([did(8), -did(0)])
        out[yi, 2], out[yi, 1] = t2, t2 + d12
    return out
