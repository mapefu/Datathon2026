"""BatteryAgent — Mode Difícil (6.000 kWh / ±1.500 kW, solar ×12, mercado a 3 h).

Control predictivo (MPC) con previsiones:
1. Cada hora se actualiza una memoria con lo observado (consumo, solar y meteo de la hora
   anterior; precio y CO2 de las 3 horas conocidas).
2. Cuatro modelos LightGBM (consumo, solar, precio, CO2) entrenados en data/train predicen las
   próximas 24 h (un modelo por variable con el horizonte como variable de entrada).
3. Un LP de 24 h decide la potencia minimizando Σ w_k·max(0, carga_neta_k + c_k − d_k), con
   w_k = co2_k + λ·precio_k. Los excedentes solares cargan gratis y la descarga no supera la
   carga neta prevista (lo que sobrase se vertería a 0 € / 0 kg).
4. Se ejecuta solo la primera hora y se recorta a los límites físicos (Safety Clamp).

El test son los últimos 7 días de cada mes y, según el FAQ oficial, se evalúan con UNA instancia
y el SOC continuo de una semana a la siguiente. El horizonte termina a fin de mes, pero la energía
que queda se conserva para la semana siguiente. Solo se vacía la batería en la última semana del
periodo o si se detecta que el SOC se reinicia entre semanas. Si faltan modelos o datos, se usan
perfiles estacionales; cualquier error acaba en 0 kW dentro de los límites físicos.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from scipy import sparse
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    sparse = linprog = None
import base64

ETA = math.sqrt(0.9)
CAP_DEFAULT = 6000.0
P_DEFAULT = 1500.0
H_PLAN = 24
KNOWN = 3                  # horas de precio/CO2 conocidas (actual + 2)
LAMBDA = 0.265             # (0,3/0,7)·CO2_base/Coste_base en train con solar ×12
V_QUANTILE = 0.30          # valor terminal de la energía almacenada (cuantil de w)
NET_SHRINK = 0.0           # kW que se restan a la carga neta prevista (prudencia al descargar)
N_SCEN_H = 24             # horas con escenarios de carga neta
N_SCEN = 10                # escenarios equiprobables por hora
USE_SIGMA = True           # escenarios escalados por el error previsto (cielo variable vs despejado)
LP_OPTIONS = {"presolve": False}   # LP pequeño: el presolve solo añade tiempo
MARGIN = 1e-4              # kW de margen frente a los límites físicos
MODEL_FILE = "modelos_dificil.json"
KEEP_AT_WEEK_END = True    # no vaciar la batería al acabar cada semana de test (SOC continuo, FAQ)
FINAL_MONTH = (2026, 8)    # última semana del periodo (datos sep-2025 … ago-2026): ahí sí se vacía
TIME_SOFT_S = 40.0         # tiempo acumulado en act() tras el cual se usan menos escenarios
TIME_HARD_S = 52.0         # ... y tras el cual el LP va sin escenarios
_ELAPSED = [0.0]           # tiempo acumulado en act() por todas las instancias (límite del servidor)

VARS = ("load", "pv", "irr", "tamb", "tpan", "price", "co2")

# Festivos (Catalunya/Barcelona) y cierres de la UPC conocidos en el periodo del reto.
HOLIDAYS = {
    "2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08",
    "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29",
    "2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06",
    "2026-04-03", "2026-04-06", "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15",
    "2026-09-11", "2026-09-24",
}

FEATURES = [
    "h", "hour", "dow", "month", "doy_s", "doy_c", "offday", "hour_t", "age",
    "load1", "load2", "load24", "load48", "load168", "load_m24", "load_off_m",
    "pv1", "pv24", "pv48", "pv_max24", "pv_sum24", "irr1", "irr24", "irr_max24", "tamb1", "tpan1",
    "p0", "p1", "p2", "p24", "p168", "p_m24",
    "c0", "c1", "c2", "c24", "c168", "c_m24",
    "cosz", "ghi_cs", "cosz1", "kt1", "kpv1", "pv_cs_pers", "kpv_day", "pv_cs_day",
    "p_shift", "c_shift", "p48", "c48", "p_d1", "c_d1",
    "p_yd", "c_yd", "p_wd", "c_wd", "p_slope", "c_slope", "load_yd",
    "p_md", "c_md", "load_md",
]

LAT = math.radians(41.39)   # Campus Nord (Barcelona)
LON = 2.11


_DST_CACHE = {}


def _utc_offset(t) -> int:
    """Hora local de Madrid: +2 entre el último domingo de marzo y el último de octubre."""
    import calendar
    y = t.year
    if y not in _DST_CACHE:
        def last_sunday(month):
            return max(w[calendar.SUNDAY] for w in calendar.monthcalendar(y, month))
        _DST_CACHE[y] = ((3, last_sunday(3), 2), (10, last_sunday(10), 3))
    start, end = _DST_CACHE[y]
    cur = (t.month, t.day, t.hour)
    return 2 if start <= cur < end else 1


def solar_cosz(t) -> float:
    """Coseno del ángulo cenital al inicio de la hora `t` (hora local, h:00). La tabla horaria
    etiqueta cada hora por su inicio, así que el centro real del intervalo es h:30. Se probó a
    calcularlo en h:30 y reentrenar: sin diferencia apreciable (±0,03), así que se mantiene."""
    hr_utc = t.hour - _utc_offset(t)
    doy = t.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (hr_utc - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
           + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hr_utc * 60 + eot + 4 * LON
    ha = math.radians(tst / 4 - 180)
    return math.sin(LAT) * math.sin(dec) + math.cos(LAT) * math.cos(dec) * math.cos(ha)


def ghi_clear(cosz: float) -> float:
    """Irradiancia global de cielo despejado (modelo de Haurwitz), W/m2."""
    return 1098.0 * cosz * math.exp(-0.057 / cosz) if cosz > 0.01 else 0.0


def _num(x, default=float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _vec(x) -> np.ndarray:
    if x is None:
        return np.zeros(0)
    try:
        return np.asarray(list(x), dtype=float).ravel()
    except (TypeError, ValueError):
        return np.asarray([_num(e) for e in x], dtype=float)


def _to_datetime(ts):
    import pandas as pd
    try:
        t = pd.Timestamp(ts if not isinstance(ts, str) else str(ts))
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        return t
    except Exception:
        return None


def _key(t) -> int:
    """Hora absoluta (horas desde epoch, sin zona)."""
    return int(round(t.value / 3.6e12))


def episode_end_key(t) -> int | None:
    """Última hora de la evaluación: el test son los últimos 7 días de cada mes, así que
    el episodio acaba a las 23:00 del último día del mes. (Los scripts locales la sustituyen.)"""
    import pandas as pd
    try:
        end = (t + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=23)
        if end < t:
            return None
        return _key(end)
    except Exception:
        return None


_EPOCH = None
_CAL_CACHE = {}


def calendar_row(key: int):
    """(hora, día semana, mes, doy_s, doy_c, festivo, cosz, ghi_cs) de la hora absoluta `key`."""
    global _EPOCH
    r = _CAL_CACHE.get(key)
    if r is None:
        import datetime as _dt
        if _EPOCH is None:
            _EPOCH = _dt.datetime(1970, 1, 1)
        ts = _EPOCH + _dt.timedelta(hours=key)
        doy = ts.timetuple().tm_yday
        off = 1.0 if (ts.weekday() >= 5 or ts.strftime("%Y-%m-%d") in HOLIDAYS) else 0.0
        cz = solar_cosz(ts)
        r = (ts.hour, ts.weekday(), ts.month, math.sin(2 * math.pi * doy / 365.25),
             math.cos(2 * math.pi * doy / 365.25), off, cz, ghi_clear(cz))
        if len(_CAL_CACHE) > 200_000:
            _CAL_CACHE.clear()
        _CAL_CACHE[key] = r
    return r


class History:
    """Memoria horaria (clave = hora absoluta) y constructor de variables de los modelos.

    Se usa igual en el agente y en el entrenamiento (src/datathon/hard_train.py)."""

    def __init__(self):
        self.d = {v: {} for v in VARS}
        self.first_key = None

    def put(self, var, key, val):
        if val is not None and math.isfinite(val):
            self.d[var][key] = float(val)

    def get(self, var, key):
        return self.d[var].get(key, float("nan"))

    def _window(self, var, k0, k1):
        dv = self.d[var]
        vals = [dv[k] for k in range(k0, k1 + 1) if k in dv]
        return vals

    def _mean_delta(self, var, tau, ref):
        """Media, en los días anteriores disponibles (hasta 7), del cambio entre la hora `ref`
        (último valor conocido) y la hora objetivo `tau` a la misma distancia."""
        d = self.d[var]
        acc = []
        for j in range(1, 8):
            a, b = d.get(tau - 24 * j), d.get(ref - 24 * j)
            if a is not None and b is not None and tau - 24 * j <= ref:
                acc.append(a - b)
        return sum(acc) / len(acc) if acc else float("nan")

    def features(self, key: int, t, H: int) -> np.ndarray:
        """Matriz (H, len(FEATURES)) para las horas key..key+H-1 decidiendo en `key`.
        Solo usa consumo/solar/meteo hasta key-1 y precio/CO2 hasta key+2."""
        nan = float("nan")
        g = self.get
        if self.first_key is None:
            self.first_key = key
        age = key - self.first_key
        L1, L2 = g("load", key - 1), g("load", key - 2)
        lw = self._window("load", key - 24, key - 1)
        load_m24 = float(np.mean(lw)) if len(lw) >= 6 else nan
        pw = self._window("pv", key - 24, key - 1)
        pv_max24 = float(np.max(pw)) if len(pw) >= 6 else nan
        pv_sum24 = float(np.sum(pw)) * 24 / len(pw) if len(pw) >= 6 else nan
        iw = self._window("irr", key - 24, key - 1)
        irr_max24 = float(np.max(iw)) if len(iw) >= 6 else nan
        prw = self._window("price", key - 21, key + 2)
        p_m24 = float(np.mean(prw)) if prw else nan
        cw = self._window("co2", key - 21, key + 2)
        c_m24 = float(np.mean(cw)) if cw else nan
        p0, p1, p2 = g("price", key), g("price", key + 1), g("price", key + 2)
        c0, c1, c2 = g("co2", key), g("co2", key + 1), g("co2", key + 2)
        pv1, irr1 = g("pv", key - 1), g("irr", key - 1)
        tamb1, tpan1 = g("tamb", key - 1), g("tpan", key - 1)
        hour_t = t.hour
        cz1, cs1 = calendar_row(key - 1)[6:8]
        kt1 = irr1 / cs1 if cs1 > 50 and math.isfinite(irr1) else nan
        kpv1 = pv1 / cs1 if cs1 > 50 and math.isfinite(pv1) else nan
        # índice de claridad de la solar en las últimas 24 h (energía real / energía de cielo despejado)
        num = den = 0.0
        for k in range(key - 24, key):
            v = self.d["pv"].get(k)
            if v is None:
                continue
            c = calendar_row(k)[7]
            if c > 50:
                num += v
                den += c
        kpv_day = num / den if den > 500 else nan
        # nivel de hoy respecto a ayer en las horas conocidas
        dp = [g("price", key + i) - g("price", key + i - 24) for i in range(KNOWN)]
        dc = [g("co2", key + i) - g("co2", key + i - 24) for i in range(KNOWN)]
        dp = [x for x in dp if math.isfinite(x)]
        dc = [x for x in dc if math.isfinite(x)]
        p_d1 = float(np.mean(dp)) if dp else nan
        c_d1 = float(np.mean(dc)) if dc else nan

        # consumo medio de la última jornada del mismo tipo (laborable / festivo)
        rows = np.full((H, len(FEATURES)), nan)
        for h in range(H):
            tau = key + h
            hour_, dow_, month_, doy_s, doy_c, off, cz, cs = calendar_row(tau)
            def lag(var, back, limit):
                k = tau - back
                return g(var, k) if k <= limit else nan
            # consumo de la última jornada del mismo tipo (laborable / festivo)
            same_off = []
            for j in range(1, 8):
                k = tau - 24 * j
                if k > key - 1:
                    continue
                if calendar_row(k)[5] == off:
                    v = g("load", k)
                    if math.isfinite(v):
                        same_off.append(v)
                        break
            rows[h] = [
                h, hour_, dow_, month_, doy_s, doy_c,
                off, hour_t, age,
                L1, L2, lag("load", 24, key - 1), lag("load", 48, key - 1), lag("load", 168, key - 1),
                load_m24, same_off[0] if same_off else nan,
                pv1, lag("pv", 24, key - 1), lag("pv", 48, key - 1), pv_max24, pv_sum24,
                irr1, lag("irr", 24, key - 1), irr_max24, tamb1, tpan1,
                p0, p1, p2, lag("price", 24, key + 2), lag("price", 168, key + 2), p_m24,
                c0, c1, c2, lag("co2", 24, key + 2), lag("co2", 168, key + 2), c_m24,
                cz, cs, cz1, kt1, kpv1, cs * kpv1, kpv_day, cs * kpv_day,
                lag("price", 24, key + 2) + p_d1, lag("co2", 24, key + 2) + c_d1,
                lag("price", 48, key + 2), lag("co2", 48, key + 2), p_d1, c_d1,
                # cambio entre la última hora conocida y la hora objetivo, ayer y hace una semana
                lag("price", 24, key + 2) - g("price", key + 2 - 24),
                lag("co2", 24, key + 2) - g("co2", key + 2 - 24),
                lag("price", 168, key + 2) - g("price", key + 2 - 168),
                lag("co2", 168, key + 2) - g("co2", key + 2 - 168),
                p2 - p0, c2 - c0,
                lag("load", 24, key - 1) - g("load", key - 1 - 24),
                self._mean_delta("price", tau, key + 2), self._mean_delta("co2", tau, key + 2),
                self._mean_delta("load", tau, key - 1),
            ]
        return rows


def _arr(b64: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype).copy()


class Forest:
    """Conjunto de árboles de LightGBM evaluado solo con numpy (sin depender de LightGBM en el
    servidor). Nodos internos: índice ≥ 0; hojas: −(índice de hoja) − 1. missing_type:
    0 = None, 1 = Zero, 2 = NaN (misma regla que NumericalDecision de LightGBM)."""

    def __init__(self, d: dict):
        feat = _arr(d["feat"], np.int32)
        thr = _arr(d["thr"], np.float64)
        dleft = _arr(d["dleft"], np.int8).astype(bool)
        mtype = _arr(d["mtype"], np.int8)
        left = _arr(d["left"], np.int32)
        right = _arr(d["right"], np.int32)
        leaf = _arr(d["leaf"], np.float64)
        roots = _arr(d["roots"], np.int32)
        # Se renumera: nodos internos 0..ni-1, hojas ni..ni+nl-1. Cada hoja apunta a sí misma
        # (umbral +inf), así el recorrido es un bucle de gathers sin máscaras.
        ni, nl = len(feat), len(leaf)
        fix = lambda a: np.where(a >= 0, a, ni + (-a - 1)).astype(np.int32)
        idx_leaf = np.arange(ni, ni + nl, dtype=np.int32)
        self.feat = np.concatenate([feat, np.zeros(nl, np.int32)])
        self.thr = np.concatenate([thr, np.full(nl, np.inf)])
        self.dleft = np.concatenate([dleft, np.ones(nl, bool)])
        self.mtype = np.concatenate([mtype, np.zeros(nl, np.int8)])
        self.left = np.concatenate([fix(left), idx_leaf])
        self.right = np.concatenate([fix(right), idx_leaf])
        self.value = np.concatenate([np.zeros(ni), leaf])
        self.roots = fix(roots)
        self.ni = ni
        self.depth = int(d["depth"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        node = np.repeat(self.roots[:, None], n, axis=1).ravel()
        rows = np.tile(np.arange(n), len(self.roots))
        act = np.flatnonzero(node < self.ni)
        while act.size:
            nd = node[act]
            x = X[rows[act], self.feat[nd]]
            mt = self.mtype[nd]
            isnan = x != x
            x0 = np.where(isnan & (mt != 2), 0.0, x)
            missing = ((mt == 1) & (np.abs(x0) <= 1e-35)) | ((mt == 2) & isnan)
            go_left = np.where(missing, self.dleft[nd], x0 <= self.thr[nd])
            nxt = np.where(go_left, self.left[nd], self.right[nd])
            node[act] = nxt
            act = act[nxt < self.ni]
        return self.value[node].reshape(len(self.roots), n).sum(axis=0)


class _Models:
    """Modelos de previsión (árboles en numpy) + perfiles estacionales de reserva, de MODEL_FILE."""

    def __init__(self):
        self.boosters = {}
        self.specs = {}
        self.profiles = {}
        self.net_q = {}
        self.net_zq = {}
        self.mg = None
        self._cache = None
        self.lam = LAMBDA
        data = None
        for base in self._dirs():
            p = base / MODEL_FILE
            try:
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    break
            except Exception:
                data = None
        if not data:
            return
        self.profiles = data.get("profiles", {})
        self.net_q = data.get("net_quantiles", {})
        self.net_zq = data.get("net_zq", {})
        self.lam = float(data.get("lambda", LAMBDA))
        for name, m in data.get("models", {}).items():
            try:
                self.boosters[name] = Forest(m["forest"])
                self.specs[name] = m
            except Exception:
                pass

    @staticmethod
    def _dirs():
        out = []
        try:
            out.append(Path(__file__).resolve().parent)
        except Exception:
            pass
        out.append(Path.cwd())
        return out

    def net_scenarios(self, h, hour):
        """Errores equiprobables (real − previsto) de la carga neta a horizonte h y hora del día."""
        try:
            return np.asarray(self.net_q[str(h)][str(hour)], dtype=float)
        except Exception:
            return None

    def profile(self, name, keys):
        """Perfil estacional mes × (laborable/festivo) × hora para las horas absolutas `keys`."""
        prof = self.profiles.get(name)
        if not prof:
            return np.full(len(keys), np.nan)
        out = []
        for k in keys:
            hour_, _, month_, _, _, off, _, _ = calendar_row(k)
            tab = prof.get(str(month_)) or next(iter(prof.values()))
            row = tab.get("1" if off else "0") or tab.get("0")
            out.append(float(row[hour_]) if row is not None else np.nan)
        return np.asarray(out, dtype=float)

    def _merge(self):
        """Une todos los bosques en uno (columnas globales de FEATURES) para recorrerlos a la vez:
        el coste lo marcan las llamadas a numpy por nivel, no el número de árboles."""
        feat, thr, dleft, mtype, left, right, value, leafm, roots, starts = [], [], [], [], [], [], [], [], [], []
        names, off, nt = [], 0, 0
        for name, f in self.boosters.items():
            cols = np.asarray([FEATURES.index(c) for c in self.specs[name].get("cols", FEATURES)])
            nn = len(f.feat)
            feat.append(cols[f.feat])
            thr.append(f.thr)
            dleft.append(f.dleft)
            mtype.append(f.mtype)
            left.append(f.left + off)
            right.append(f.right + off)
            value.append(f.value)
            leafm.append(np.arange(nn) >= f.ni)
            roots.append(f.roots + off)
            starts.append(nt)
            names.append(name)
            off += nn
            nt += len(f.roots)
        cat = np.concatenate
        self.mg = dict(feat=cat(feat), thr=cat(thr), dleft=cat(dleft), mtype=cat(mtype), left=cat(left),
                       right=cat(right), value=cat(value), leaf=cat(leafm), roots=cat(roots),
                       starts=np.asarray(starts), names=names)

    def predict_all(self, X) -> dict:
        """Predicciones de todos los modelos (con su base sumada) sobre la matriz de FEATURES."""
        if self._cache is not None and self._cache[0] is X:
            return self._cache[1]
        if self.mg is None:
            self._merge()
        g = self.mg
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        T = len(g["roots"])
        node = np.repeat(g["roots"][:, None], n, axis=1).ravel()
        rows = np.tile(np.arange(n), T)
        act = np.flatnonzero(~g["leaf"][node])
        while act.size:
            nd = node[act]
            x = X[rows[act], g["feat"][nd]]
            mt = g["mtype"][nd]
            isnan = x != x
            x0 = np.where(isnan & (mt != 2), 0.0, x)
            missing = ((mt == 1) & (np.abs(x0) <= 1e-35)) | ((mt == 2) & isnan)
            go_left = np.where(missing, g["dleft"][nd], x0 <= g["thr"][nd])
            nxt = np.where(go_left, g["left"][nd], g["right"][nd])
            node[act] = nxt
            act = act[~g["leaf"][nxt]]
        sums = np.add.reduceat(g["value"][node].reshape(T, n), g["starts"], axis=0)
        out = {}
        for i, name in enumerate(g["names"]):
            y = sums[i].copy()
            base = self.specs[name].get("base")
            if base:
                y = y + X[:, FEATURES.index(base)]
            out[name] = y
        self._cache = (X, out)
        return out

    def predict(self, name, X):
        """Predicción del modelo `name` sobre la matriz completa de FEATURES.
        Cada modelo usa sus columnas y, si tiene base, predice la diferencia respecto a ella."""
        if name not in self.boosters:
            return None
        return self.predict_all(X)[name]


class _LP:
    """LP de horizonte H con escenarios de carga neta en las primeras horas.

    Variables [c, d, g, s, e]:  min Σ_k w_k·g_k + Σ_{k<S} w_k·mean_j e_kj − v·s_H
    s.a. c_k − d_k − g_k ≤ −net_k             (horas k ≥ S: previsión puntual)
         c_k − d_k − e_kj ≤ −(net_k + r_kj)    (horas k < S: un escenario j por error r_kj)
         s_k − s_{k−1} − η·c_k + d_k/η = 0,  0 ≤ c,d ≤ P,  g,e ≥ 0,  0 ≤ s ≤ CAP.
    La acción (c_k, d_k) es la misma en todos los escenarios de su hora.
    """

    def __init__(self):
        self.cache = {}

    def _mats(self, H, S, K):
        key = (H, S, K)
        if key not in self.cache:
            I = sparse.identity(H, format="csr")
            Z = sparse.csr_matrix((H, H))
            D = I - sparse.eye(H, k=-1, format="csr")
            rows = [sparse.hstack([I, -I, -I, Z, sparse.csr_matrix((H, S * K))], format="csr")]
            if S:
                # fila (k, j): c_k − d_k − e_kj
                sel = sparse.csr_matrix((np.ones(S * K), (np.arange(S * K), np.repeat(np.arange(S), K))),
                                        shape=(S * K, H))
                rows.append(sparse.hstack([sel, -sel, sparse.csr_matrix((S * K, 2 * H)),
                                           -sparse.identity(S * K, format="csr")], format="csr"))
            A_ub = sparse.vstack(rows, format="csr")
            A_eq = sparse.hstack([-ETA * I, I / ETA, Z, D, sparse.csr_matrix((H, S * K))], format="csr")
            self.cache[key] = (A_ub, A_eq)
        return self.cache[key]

    def first_action(self, w, net, soc, cap, p_max, v, scen=None) -> float:
        """scen: matriz (S, K) con la carga neta de cada escenario para las horas 0..S-1."""
        H = len(w)
        S, K = (scen.shape if scen is not None else (0, 0))
        S = min(S, H)
        if scen is not None:
            scen = scen[:S]
        A_ub, A_eq = self._mats(H, S, K)
        b_eq = np.zeros(H)
        b_eq[0] = soc
        wg = np.asarray(w, dtype=float).copy()
        wg[:S] = 0.0  # en las horas con escenarios cuenta la media de los escenarios
        we = np.repeat(np.asarray(w[:S], dtype=float) / max(K, 1), K)
        cost = np.concatenate([np.zeros(2 * H), wg, np.zeros(H), we])
        cost[3 * H + H - 1] -= v
        b_ub = -np.asarray(net, dtype=float)
        if S:
            b_ub = np.concatenate([b_ub, -scen.ravel()])
        bounds = [(0.0, p_max)] * (2 * H) + [(0.0, None)] * H + [(0.0, cap)] * H + [(0.0, None)] * (S * K)
        sol = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
                      options=LP_OPTIONS)
        if sol.status != 0 or sol.x is None:
            raise RuntimeError("LP")
        return float(sol.x[0] - sol.x[H])


class BatteryAgent:
    def __init__(self):
        self.hist = History()
        self.models = _Models()
        self.lp = _LP()
        self.prev_key = None
        self.prev_soc_pred = None
        self.cap = CAP_DEFAULT
        self.p_max = P_DEFAULT
        self.end_key = None
        self.vec_len_max = 0
        self.reset_detected = False

    # ------------------------------------------------------------------ util
    def _limits(self, soc):
        max_c = max(0.0, min(self.p_max, (self.cap - soc) / ETA - MARGIN))
        max_d = max(0.0, min(self.p_max, soc * ETA - MARGIN))
        return max_c, max_d

    def _read_spec(self, obs):
        cap = _num(obs.get("cap_max_kwh"))
        p = _num(obs.get("power_max_kw"))
        if not (cap > 0):
            s_kwh, s_pct = _num(obs.get("soc_kwh")), _num(obs.get("soc_pct"))
            cap = s_kwh / s_pct if (s_pct > 1e-3 and s_kwh > 0) else CAP_DEFAULT
        self.cap = cap
        self.p_max = p if p > 0 else P_DEFAULT

    def _soc(self, obs):
        soc = _num(obs.get("soc_kwh"))
        if math.isnan(soc):
            pct = _num(obs.get("soc_pct"))
            if not math.isnan(pct):
                soc = pct * self.cap
        if math.isnan(soc):
            pct = _num(obs.get("soc_percent"))
            if not math.isnan(pct):
                soc = pct / 100.0 * self.cap
        if math.isnan(soc) and self.prev_soc_pred is not None:
            soc = self.prev_soc_pred
        if math.isnan(soc):
            return None
        return min(self.cap, max(0.0, soc))

    # ------------------------------------------------------------------ main
    def act(self, obs) -> float:
        t0 = time.perf_counter()
        try:
            return self._act(obs)
        finally:
            _ELAPSED[0] += time.perf_counter() - t0

    def _act(self, obs) -> float:
        try:
            self._read_spec(obs)
            soc = self._soc(obs)
        except Exception:
            soc = None
        if soc is None:
            self.prev_soc_pred = None
            return 0.0
        try:
            p = self._decide(obs, soc)
        except Exception:
            p = 0.0
        if not math.isfinite(p):
            p = 0.0
        max_c, max_d = self._limits(soc)
        p = max(-max_d, min(max_c, p))
        if abs(p) < 1e-3:
            p = 0.0
        self.prev_soc_pred = soc + (p * ETA if p >= 0 else p / ETA)
        return float(p)

    def observe(self, obs):
        """Actualiza la memoria. Devuelve (key, timestamp, vector de precios, vector de CO2)."""
        t = _to_datetime(obs.get("timestamp"))
        if t is None:
            raise ValueError("timestamp")
        key = _key(t)
        dades = obs.get("noves_dades_5min")
        if not isinstance(dades, dict):
            dades = {}
        H = self.hist
        # La telemetría es la de la fila anterior de la simulación, es decir, la hora de la
        # observación anterior (prev_key). Normalmente prev_key = key − 1. Tras un salto de
        # timestamp (hueco de datos, cambio de semana de test o cambio de hora) pertenece a la
        # última hora antes del salto: se guarda en su hora real y también en key − 1 como
        # último valor conocido, para que las variables de retardo 1 no queden vacías (medido
        # en validación: dejarlas en NaN tras un hueco empeora la previsión).
        # En el primer paso el simulador repite la hora actual y no se guarda.
        if self.prev_key is not None and self.prev_key < key:
            tel = (("load", "ultim_consum_kw"), ("pv", "ultima_solar_kw"), ("irr", "irradiancia_wm2"),
                   ("tamb", "temp_ambient_c"), ("tpan", "temp_panel_c"))
            for k_tel in {self.prev_key, key - 1}:
                for var, field in tel:
                    H.put(var, k_tel, _num(dades.get(field)))
        vp, vc = _vec(obs.get("preus_properes_24h")), _vec(obs.get("co2_properes_24h"))
        price, co2 = _num(obs.get("preu_actual")), _num(obs.get("co2_actual"))
        H.put("price", key, price)
        H.put("co2", key, co2)
        # si llegan más de 3 horas (p. ej. 24 h como en el Mode Estàndard) se aprovechan todas
        for i in range(1, min(len(vp), H_PLAN + 1)):
            H.put("price", key + i, vp[i])
        for i in range(1, min(len(vc), H_PLAN + 1)):
            H.put("co2", key + i, vc[i])
        if math.isnan(price) and len(vp):
            H.put("price", key, vp[0])
        if math.isnan(co2) and len(vc):
            H.put("co2", key, vc[0])
        self.prev_key = key
        return key, t, vp, vc

    def forecast(self, key, t, H):
        """Previsión de (precio, co2, consumo, solar) para key..key+H-1."""
        X = self.hist.features(key, t, H)
        self._X = X
        keys = range(key, key + H)
        out = {}
        for name in ("load", "pv", "price", "co2"):
            try:
                yhat = self.models.predict(name, X)
            except Exception:
                yhat = None
            if yhat is None or not np.isfinite(yhat).all():
                prof = self.models.profile(name, keys)
                yhat = prof if yhat is None else np.where(np.isfinite(yhat), yhat, prof)
            out[name] = yhat
        for name in ("price", "co2"):
            for h in range(H):
                v = self.hist.get(name, key + h)
                if math.isfinite(v):
                    out[name][h] = v
        # rellenos finales
        for name, default in (("price", 0.12), ("co2", 0.08), ("load", 1000.0), ("pv", 0.0)):
            v = out[name]
            bad = ~np.isfinite(v)
            if bad.all():
                v[:] = default
            elif bad.any():
                i = np.arange(len(v))
                v[bad] = np.interp(i[bad], i[~bad], v[~bad])
        out["pv"] = np.maximum(out["pv"], 0.0)
        out["load"] = np.maximum(out["load"], 0.0)
        return out

    def _decide(self, obs, soc) -> float:
        prev_key, prev_soc = self.prev_key, self.prev_soc_pred
        key, t, vp, vc = self.observe(obs)
        # Tras un salto de semana, ¿se ha reiniciado el SOC al 50 %? Si es así, el servidor
        # evalúa cada semana por separado y conviene vaciar la batería al final de cada una.
        if (prev_key is not None and key - prev_key > 2 and prev_soc is not None
                and abs(soc - prev_soc) > 1.0 and abs(soc - 0.5 * self.cap) < 1.0):
            self.reset_detected = True
        H = H_PLAN
        truncated = False
        end = episode_end_key(t)
        if end is not None and end - key + 1 <= H:
            H = max(1, end - key + 1)
            # Fin de la semana de test. Según el FAQ, la simulación sigue con la semana del mes
            # siguiente sin reiniciar el SOC, así que la energía que quede vale lo mismo que
            # cualquier otra hora (valor terminal normal). Se vacía en la última semana del
            # periodo o si se ha visto que el SOC se reinicia entre semanas.
            truncated = (not KEEP_AT_WEEK_END or self.reset_detected
                         or (t.year, t.month) == FINAL_MONTH)
        # vectores más cortos de lo normal: la evaluación acaba antes
        n_vec = min(len(vp), len(vc))
        self.vec_len_max = max(self.vec_len_max, n_vec)
        if 0 < n_vec < min(self.vec_len_max, KNOWN) and n_vec < H:
            H, truncated = n_vec, True
        f = self.forecast(key, t, H)
        w = f["co2"] + self.models.lam * f["price"]
        net = f["load"] - f["pv"]
        net = np.where(net > 0, np.maximum(net - NET_SHRINK, 0.0), net)
        v = 0.0 if truncated else float(np.quantile(w, V_QUANTILE)) * ETA
        scen = None
        n_scen_h = N_SCEN_H
        if _ELAPSED[0] > TIME_HARD_S:
            n_scen_h = 0
        elif _ELAPSED[0] > TIME_SOFT_S:
            n_scen_h = min(N_SCEN_H, 3)
        if n_scen_h > 0 and self.models.net_q:
            raw = f["load"] - f["pv"]
            rows = []
            sig = None
            if USE_SIGMA and self.models.net_zq and "net_sigma" in self.models.boosters:
                try:
                    sig = self.models.predict("net_sigma", self._X)
                    sig = np.maximum(sig, self.models.specs["net_sigma"].get("floor", 5.0))
                    if not np.isfinite(sig).all():
                        sig = None
                except Exception:
                    sig = None
            ghi_col = FEATURES.index("ghi_cs")
            for h in range(min(n_scen_h, H)):
                if sig is not None:
                    try:
                        z = np.asarray(self.models.net_zq[str(h)]["1" if self._X[h, ghi_col] > 0 else "0"])
                        r = sig[h] * z
                    except Exception:
                        r = None
                else:
                    r = self.models.net_scenarios(h, (t.hour + h) % 24)
                if r is None or len(r) != N_SCEN:
                    rows = []
                    break
                rows.append(raw[h] + r)
            if rows:
                scen = np.vstack(rows)
        return self.lp.first_action(w, net, soc, self.cap, self.p_max, v, scen)
