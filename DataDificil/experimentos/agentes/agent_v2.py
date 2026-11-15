"""BatteryAgent — Datathon UPC Sostenible 2026 (Modo Estándar y Modo Difícil).

Módulo único de runtime. Contiene (en este orden):
1. Constantes y utilidades numéricas.
2. Carga del bundle (memoria horaria de train, calendario, climatologías y bosques) desde
   `Path(__file__).parent` (o `BUNDLE_DIR`, que fija el arnés local de validación cruzada).
3. `build_features`: variables de previsión para una hora de decisión, con límites de causalidad
   estrictos (sensores hasta t−1, mercado hasta t+K−1). Es la MISMA función que usa el entrenamiento.
4. `night_rule` y `pv_from_k`: post-proceso compartido con el entrenamiento.
5. `Forest`: bosques de LightGBM exportados a arrays y evaluados con numpy (sin LightGBM en el servidor).
6. `BatteryAgent` (Fase 3).
"""

from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from scipy import sparse
    from scipy.optimize import linprog
except Exception:  # pragma: no cover  (sin scipy → regla de reserva)
    sparse = linprog = None

# ── RawTree observability (offline-safe) ──────────────────────────────────
try:
    from rawtree_logger import RawTreeLogger
    _rt = RawTreeLogger()
except Exception:  # pragma: no cover  (sin rawtree_logger → sin observabilidad)
    _rt = None  # type: ignore[assignment]

# ----------------------------------------------------------------------------- 1. constantes
ETA = math.sqrt(0.90)
EPS_CLAMP = 1e-4
H_PLAN = 24                        # horizonte del LP
K_SCEN = 9                         # escenarios de carga neta por paso
SPREAD_MODE = "horizon"            # "horizon": σ del horizonte; "decision": σ de una hora (re-decisión)
V_Q = {"hard": 0.30, "std": 0.40}  # cuantil de w para el valor terminal de la energía
V_ETA = {"hard": True, "std": False}
SIG_MULT = {"hard": 1.0, "std": 1.0}   # escala de σ de la carga neta
Z_SHIFT = {"hard": 0.0, "std": 0.0}    # desplazamiento de los cuantiles estandarizados (en σ)
TRUTH = None                       # diagnóstico local: {idx: (load, pv×1, precio, co2)}
TRUTH_KEYS = ()                    # "net0", "net", "price", "co2"
MKT_ALPHA = {"price": 1.0, "co2": 1.0}  # escala de la previsión de mercado respecto al último dato conocido
Z_FINE = 0                          # >0: interpolar los 9 cuantiles z a Z_FINE niveles equiprobables
EPS_LP = 1e-5                      # coste por kWh movido: evita cargar y descargar a la vez
CROSS_BLOCK = True                 # al final de un bloque, planificar con el arranque del bloque siguiente
NIGHT_EXACT = True                 # σ = 0 en las horas de noche con el contador congelado observado
JULY_LAST_IF_NO_NOV = True         # sin noviembre en la evaluación, julio se trata como último bloque
TIME_SOFT_S, TIME_MID_S, TIME_HARD_S = 38.0, 46.0, 52.0
LAM = {"hard": 0.2617, "std": 0.2517}
BASE_YMDH = (2025, 9, 1, 0)        # índice 0 = 2025-09-01 00:00 (hora local, como `Fecha`)
BUNDLE_FILE = "bundle.npz"
PARAMS_FILE = "params.json"
BUNDLE_DIR = None                  # el arnés local lo sustituye por el directorio de modelos del pliegue
BLOCKS = None                      # [(idx_inicio, idx_fin)] de los bloques de test; None → los de params.json
H_FEAT = 27                        # filas de variables por decisión: h = 0..26
MEM_VARS = ("load", "pv", "irr", "tamb", "price", "co2")
CAL_VARS = ("hour", "dow", "daytype", "act", "period", "night", "utc_off", "hour_utc", "day", "year",
            "cs", "cs_dmax", "lvl")

NAN = float("nan")

CAL_FEATS = ["h", "hour_tau", "hour_t", "dow_tau", "daytype_tau", "daytype_t", "act_tau", "night_tau",
             "night_anchor", "period_tau", "period_a", "hour_utc_tau"]
LOAD_FEATS = ["L_base", "L1", "L2", "dL", "L24", "L48", "L168", "L_same", "ramp_yd", "ramp_wk",
              "ramp_md", "L_m24", "F_prev", "tamb1", "tamb_m24", "lvl_tau", "lvl_t1"]
PV_FEATS = ["cs_tau", "cs_rel", "k1", "k2", "dk", "k_day", "k_yday", "k_24", "k24", "cs1"]
MKT_STEMS = ["0", "1", "2", "slope", "24", "48", "168", "lvl_d1", "shape_yd", "shape_wk", "shape_md",
             "xfer_yd", "m24", "min24", "max24"]
MKT_FEATS = [f"p_{s}" for s in MKT_STEMS] + [f"c_{s}" for s in MKT_STEMS] + ["p24_adj"]
MKT_LAGS = (3, 6, 12)                    # x[a − d] − x[a]: tendencia del mercado en las últimas horas
LAG_FEATS = [f"{px}_l{d}" for px in ("p", "c") for d in MKT_LAGS]
# variables añadidas a los modelos de mercado ({x}: propia, {y}: la otra serie)
MKT_EXTRA = ["k1", "cs_tau", "hour_utc_tau", "{y}_24", "{x}_l3", "{x}_l6", "{x}_l12", "{y}_l3", "{y}_l6"]
FEATS = CAL_FEATS + LOAD_FEATS + PV_FEATS + MKT_FEATS + LAG_FEATS
FI = {f: i for i, f in enumerate(FEATS)}
METEO_FEATS = ("tamb1", "tamb_m24")      # ausentes en Modo Estándar


def _num(x, default=NAN) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _take(a: np.ndarray, idx, lo: int, hi: int) -> np.ndarray:
    """a[idx] donde lo ≤ idx ≤ hi y dentro del array; NaN en el resto (límite de causalidad)."""
    idx = np.asarray(idx, dtype=np.int64)
    ok = (idx >= max(lo, 0)) & (idx <= min(hi, a.shape[0] - 1))
    out = np.full(idx.shape, NAN)
    out[ok] = a[idx[ok]]
    return out


def _fmean(v: np.ndarray, axis=None, min_n: int = 1):
    """Media de los valores finitos (NaN si hay menos de min_n)."""
    ok = np.isfinite(v)
    n = ok.sum(axis=axis)
    s = np.where(ok, v, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_n, s / np.maximum(n, 1), NAN)


# ----------------------------------------------------------------------------- 2. bundle
def bundle_dir() -> Path:
    if BUNDLE_DIR is not None:
        return Path(BUNDLE_DIR)
    return Path(__file__).resolve().parent


def load_bundle(d: Path | None = None) -> tuple[dict, dict, dict, dict]:
    """(memoria, calendario, arrays del bosque y tablas, params). Lanza excepción si falta algo."""
    d = Path(d) if d is not None else bundle_dir()
    params = json.loads((d / PARAMS_FILE).read_text(encoding="utf-8"))
    with np.load(d / BUNDLE_FILE, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    M = {v: arrs.pop("M_" + v).astype(np.float64) for v in MEM_VARS}
    C = {v: arrs.pop("C_" + v).astype(np.float64) for v in CAL_VARS}
    C["toll"] = arrs.pop("C_toll").astype(np.float64)
    return M, C, arrs, params


# ----------------------------------------------------------------------------- 3. variables
def build_features(M: dict, C: dict, t: int, k_obs: int = 3, hs=None, floor: int | None = None,
                   meteo: bool = True) -> np.ndarray:
    """Matriz (len(hs), len(FEATS)) para las horas objetivo τ = t + h decidiendo en la hora t.

    Lecturas permitidas: sensores (load, pv, irr, tamb) con índice en [floor, t−1]; mercado (price, co2)
    con índice ≤ a = t + k_obs − 1. `floor` emula un arranque en frío (sin sensores anteriores)."""
    hs = np.arange(H_FEAT) if hs is None else np.asarray(hs, dtype=np.int64)
    nh = len(hs)
    t = int(t)
    tau = t + hs
    ncal = C["hour"].shape[0]
    tc = np.clip(tau, 0, ncal - 1)
    lo = 0 if floor is None else int(floor)
    sh = t - 1                                   # último índice de sensores visible
    a = t + max(int(k_obs), 1) - 1               # último índice de mercado visible
    X = np.full((nh, len(FEATS)), NAN)

    def cal(name, i):
        return C[name][min(max(int(i), 0), ncal - 1)]

    # --- calendario
    X[:, FI["h"]] = hs
    X[:, FI["hour_tau"]] = C["hour"][tc]
    X[:, FI["hour_t"]] = cal("hour", t)
    X[:, FI["dow_tau"]] = C["dow"][tc]
    X[:, FI["daytype_tau"]] = C["daytype"][tc]
    X[:, FI["daytype_t"]] = cal("daytype", t)
    X[:, FI["act_tau"]] = C["act"][tc]
    X[:, FI["night_tau"]] = C["night"][tc]
    X[:, FI["night_anchor"]] = cal("night", t - 1)
    X[:, FI["period_tau"]] = C["period"][tc]
    X[:, FI["period_a"]] = cal("period", a)
    X[:, FI["hour_utc_tau"]] = C["hour_utc"][tc]

    # --- consumo
    L = M["load"]
    l3 = _take(L, [t - 1, t - 2, t - 3], lo, sh)
    fin = np.flatnonzero(np.isfinite(l3))
    L1 = l3[fin[0]] if len(fin) else NAN
    L2 = l3[1]
    lvl_t1 = cal("lvl", t - 1)
    X[:, FI["L1"]] = L1
    X[:, FI["L2"]] = L2
    X[:, FI["dL"]] = L1 - L2
    X[:, FI["L_base"]] = L1 if math.isfinite(L1) else lvl_t1
    X[:, FI["lvl_t1"]] = lvl_t1
    X[:, FI["lvl_tau"]] = C["lvl"][tc]
    X[:, FI["L24"]] = _take(L, tau - 24, lo, sh)
    X[:, FI["L48"]] = _take(L, tau - 48, lo, sh)
    X[:, FI["L168"]] = _take(L, tau - 168, lo, sh)
    J = tau[:, None] - 24 * np.arange(1, 8)[None, :]
    LJ = _take(L, J, lo, sh)
    same = (C["daytype"][np.clip(J, 0, ncal - 1)] == C["daytype"][tc][:, None]) & np.isfinite(LJ)
    has = same.any(axis=1)
    first = np.argmax(same, axis=1)
    X[:, FI["L_same"]] = np.where(has, LJ[np.arange(nh), first], NAN)
    ref = _take(L, t - 1 - 24 * np.arange(1, 8), lo, sh)
    X[:, FI["ramp_yd"]] = LJ[:, 0] - ref[0]
    X[:, FI["ramp_wk"]] = X[:, FI["L168"]] - _take(L, [t - 1 - 168], lo, sh)[0]
    X[:, FI["ramp_md"]] = _fmean(LJ - ref[None, :], axis=1)
    X[:, FI["L_m24"]] = _fmean(_take(L, np.arange(t - 24, t), lo, sh), min_n=6)
    back = np.arange(t - 1, t - 49, -1)
    lb = _take(L, back, lo, sh)
    nb = C["night"][np.clip(back, 0, ncal - 1)] == 1
    okb = np.flatnonzero(nb & np.isfinite(lb))
    X[:, FI["F_prev"]] = lb[okb[0]] if len(okb) else NAN
    if meteo:
        T = M["tamb"]
        X[:, FI["tamb1"]] = _take(T, [t - 1], lo, sh)[0]
        X[:, FI["tamb_m24"]] = _fmean(_take(T, np.arange(t - 24, t), lo, sh), min_n=6)

    # --- solar (escala ×1, índice de claridad k = pv / cs)
    P, cs = M["pv"], C["cs"]
    cs_tau = cs[tc]
    X[:, FI["cs_tau"]] = cs_tau
    dmax = C["cs_dmax"][tc]
    X[:, FI["cs_rel"]] = np.where(dmax > 0, cs_tau / np.where(dmax > 0, dmax, 1.0), 0.0)

    def kidx(idx):
        idx = np.asarray(idx, dtype=np.int64)
        c = cs[np.clip(idx, 0, ncal - 1)]
        v = _take(P, idx, lo, sh)
        return np.where(c > 5, v / np.where(c > 5, c, 1.0), NAN)

    def kagg(i0, i1, min_n):
        idx = np.arange(i0, i1)
        if len(idx) == 0:
            return NAN
        c = cs[np.clip(idx, 0, ncal - 1)]
        v = _take(P, idx, lo, sh)
        ok = (c > 5) & np.isfinite(v)
        return float(v[ok].sum() / c[ok].sum()) if ok.sum() >= min_n else NAN

    k12 = kidx([t - 1, t - 2])
    X[:, FI["k1"]] = k12[0]
    X[:, FI["k2"]] = k12[1]
    X[:, FI["dk"]] = k12[0] - k12[1]
    ds = t - int(cal("hour", t))
    X[:, FI["k_day"]] = kagg(ds, t, 2)
    X[:, FI["k_yday"]] = kagg(ds - 24, ds, 3)
    X[:, FI["k_24"]] = kagg(t - 24, t, 3)
    X[:, FI["k24"]] = kidx(tau - 24)
    X[:, FI["cs1"]] = cal("cs", t - 1)

    # --- mercado (precio "p", CO2 "c")
    j7 = 24 * np.arange(1, 8)
    for var, px in (("price", "p"), ("co2", "c")):
        V = M[var]
        x012 = _take(V, [t, t + 1, a], 0, a)
        x0, x1, x2 = x012
        X[:, FI[f"{px}_0"]] = x0
        X[:, FI[f"{px}_1"]] = x1
        X[:, FI[f"{px}_2"]] = x2
        X[:, FI[f"{px}_slope"]] = x2 - x0
        x24 = _take(V, tau - 24, 0, a)
        X[:, FI[f"{px}_24"]] = x24
        X[:, FI[f"{px}_48"]] = _take(V, tau - 48, 0, a)
        x168 = _take(V, tau - 168, 0, a)
        X[:, FI[f"{px}_168"]] = x168
        kk = np.arange(t, min(a, t + 2) + 1)
        X[:, FI[f"{px}_lvl_d1"]] = _fmean(_take(V, kk, 0, a) - _take(V, kk - 24, 0, a))
        ra = _take(V, a - j7, 0, a)
        X[:, FI[f"{px}_shape_yd"]] = x24 - ra[0]
        X[:, FI[f"{px}_shape_wk"]] = x168 - _take(V, [a - 168], 0, a)[0]
        X[:, FI[f"{px}_shape_md"]] = _fmean(_take(V, tau[:, None] - j7[None, :], 0, a) - ra[None, :], axis=1)
        X[:, FI[f"{px}_xfer_yd"]] = x2 + X[:, FI[f"{px}_shape_yd"]]
        lags = _take(V, [a - d for d in MKT_LAGS], 0, a) - x2
        for d, v in zip(MKT_LAGS, lags):
            X[:, FI[f"{px}_l{d}"]] = v
        w = _take(V, np.arange(a - 23, a + 1), 0, a)
        okw = np.isfinite(w)
        X[:, FI[f"{px}_m24"]] = _fmean(w, min_n=6)
        X[:, FI[f"{px}_min24"]] = w[okw].min() if okw.sum() >= 6 else NAN
        X[:, FI[f"{px}_max24"]] = w[okw].max() if okw.sum() >= 6 else NAN
    i24 = np.clip(tau - 24, 0, ncal - 1)
    toll = C["toll"]
    yr = lambda i: np.clip(C["year"][i].astype(np.int64) - 2025, 0, toll.shape[0] - 1)  # noqa: E731
    X[:, FI["p24_adj"]] = (X[:, FI["p_24"]] - toll[yr(i24), C["period"][i24].astype(np.int64)]
                           + toll[yr(tc), C["period"][tc].astype(np.int64)])
    return X


# ----------------------------------------------------------------------------- 4. post-proceso
def pv_from_k(k_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Solar ×1 desde el índice de claridad previsto. Con cs ≤ 5 kW se usa la claridad del día."""
    cs = X[:, FI["cs_tau"]]
    kd = X[:, FI["k_day"]]
    kd = np.where(np.isfinite(kd), kd, 0.6)
    k = np.where(cs > 5, np.clip(np.nan_to_num(k_hat, nan=0.6), 0.0, 1.25), np.clip(kd, 0.0, 1.25))
    return np.where(cs > 0, k * cs, 0.0)


def night_rule(load_hat: np.ndarray, X: np.ndarray, C: dict, t: int, M: dict, floor: int | None = None):
    """Contador congelado de 22:05 a 06:00: si la hora anterior es de noche (o son las 23 h) y su
    consumo es un dato observado, las horas de noche consecutivas desde h = 0 valen exactamente ese dato
    (exact = True). Las noches futuras toman un perfil plano igual a la previsión de sus 23 h."""
    load_hat = load_hat.copy()
    hs = X[:, FI["h"]].astype(np.int64)
    night = X[:, FI["night_tau"]] == 1
    exact = np.zeros(len(hs), bool)
    lo = 0 if floor is None else floor
    l1 = _take(M["load"], [t - 1], lo, t - 1)[0]
    if math.isfinite(l1) and (X[0, FI["night_anchor"]] == 1 or X[0, FI["hour_t"]] == 23):
        k = 0
        while k < len(hs) and hs[k] == k and night[k]:
            load_hat[k] = l1
            exact[k] = True
            k += 1
    hour = X[:, FI["hour_tau"]].astype(np.int64)
    tau = t + hs
    nid = (tau - hour + np.where(hour == 23, 24, 0)) // 24      # noche = día en que acaba (horas 23 → 5)
    for n in np.unique(nid[night & ~exact]):
        rows = np.flatnonzero((nid == n) & night & ~exact)
        start = rows[hour[rows] == 23]
        if len(start):
            load_hat[rows] = load_hat[start[0]]
    return load_hat, exact


SENSOR_MODELS = ("load@0", "load@1", "load@3", "pv@0", "pv@1", "pv@3")
MKT_MODELS = ("price@3", "price@8", "co2@3", "co2@8")


def _bands(P: dict, names, n: int) -> np.ndarray:
    y = np.full(n, NAN)
    for name in names:
        v = P.get(name)
        if v is not None:
            y = np.where(np.isfinite(v), v, y)
    return y


def point_forecast(forest, X: np.ndarray, C: dict, M: dict, t: int, floor: int | None = None,
                   market: bool = True, n: int | None = None, k_known: int = 3, extra=()) -> dict:
    """Previsión puntual por banda de horizonte: consumo (kW), solar ×1 (kW), precio y CO2.
    El mercado de las horas conocidas queda NaN (se toma de obs). Incluye la regla de la noche.
    `n` limita las filas evaluadas a h < n (y el mercado a h ≥ k_known); `extra` añade otros modelos
    (p. ej. sigma) en la misma pasada por los árboles."""
    names = list(SENSOR_MODELS) + (list(MKT_MODELS) if market else []) + list(extra)
    rows_of = None
    if n is not None:
        h = X[:, FI["h"]]
        rows_of = {}
        for name in names:
            m = forest.models[forest.mi[name]]
            sel = (h >= m["h_lo"]) & (h < m["h_hi"]) & (h < n)
            if name in MKT_MODELS:
                sel &= h >= k_known
            rows_of[name] = np.flatnonzero(sel)
    P = forest.predict(X, names, rows_of)
    nr = X.shape[0]
    load = _bands(P, ("load@0", "load@1", "load@3"), nr)
    pv = pv_from_k(_bands(P, ("pv@0", "pv@1", "pv@3"), nr), X)
    load, exact = night_rule(load, X, C, t, M, floor)
    out = {"load": load, "pv": pv, "exact": exact}
    if market:
        out["price"] = _bands(P, ("price@3", "price@8"), nr)
        out["co2"] = _bands(P, ("co2@3", "co2@8"), nr)
    for name in extra:
        out[name] = P[name]
    return out


# ----------------------------------------------------------------------------- 5. bosques
class Forest:
    """Bosques de LightGBM fundidos en un solo conjunto de arrays (columnas globales de FEATS).

    Nodos internos 0..ni−1 y hojas a continuación; cada hoja apunta a sí misma. missing_type:
    0 None, 1 Zero, 2 NaN (misma regla que NumericalDecision de LightGBM)."""

    def __init__(self, arrs: dict, models: list[dict]):
        self.feat = arrs["F_feat"].astype(np.int64)
        self.thr = arrs["F_thr"].astype(np.float64)
        self.dleft = arrs["F_dleft"].astype(bool)
        self.mtype = arrs["F_mtype"].astype(np.int8)
        self.left = arrs["F_left"].astype(np.int64)
        self.right = arrs["F_right"].astype(np.int64)
        self.value = arrs["F_value"].astype(np.float64)
        self.leaf = arrs["F_leaf"].astype(bool)
        self.roots = arrs["F_roots"].astype(np.int64)
        self.tree_model = arrs["F_tree_model"].astype(np.int64)
        # dirección de un NaN en cada nodo: NaN → rama por defecto; None → se trata como 0,0
        self.has_zero = bool((self.mtype == 1).any())
        self.nan_left = np.where(self.mtype == 2, self.dleft, 0.0 <= self.thr)
        self.models = models
        self.mi = {m["name"]: i for i, m in enumerate(models)}
        self.trees_of = {m["name"]: np.flatnonzero(self.tree_model == i) for i, m in enumerate(models)}

    def predict(self, X: np.ndarray, names, rows_of: dict | None = None) -> dict:
        """{nombre: predicción (con su base sumada)} para los modelos `names`. `rows_of[nombre]` limita
        las filas evaluadas (el resto queda NaN); por defecto, las filas de la banda de h del modelo."""
        n = X.shape[0]
        h = X[:, FI["h"]]
        t_list, r_list = [], []
        sel_rows = {}
        for name in names:
            m = self.models[self.mi[name]]
            if rows_of is not None and name in rows_of:
                rows = np.asarray(rows_of[name], dtype=np.int64)
            else:
                rows = np.flatnonzero((h >= m["h_lo"]) & (h < m["h_hi"]))
            sel_rows[name] = rows
            tr = self.trees_of[name]
            if len(rows) == 0 or len(tr) == 0:
                continue
            t_list.append(np.repeat(tr, len(rows)))
            r_list.append(np.tile(rows, len(tr)))
        out = {}
        if t_list:
            tt = np.concatenate(t_list)
            rr = np.concatenate(r_list)
            node = self.roots[tt]
            act = np.flatnonzero(~self.leaf[node])
            Xf = np.ascontiguousarray(X, dtype=np.float64).ravel()
            base = rr * X.shape[1]
            while act.size:
                nd = node[act]
                x = Xf[base[act] + self.feat[nd]]
                if self.has_zero:
                    mt = self.mtype[nd]
                    isnan = x != x
                    x0 = np.where(isnan & (mt != 2), 0.0, x)
                    missing = ((mt == 1) & (np.abs(x0) <= 1e-35)) | ((mt == 2) & isnan)
                    go_left = np.where(missing, self.dleft[nd], x0 <= self.thr[nd])
                else:
                    go_left = np.where(x != x, self.nan_left[nd], x <= self.thr[nd])
                nxt = np.where(go_left, self.left[nd], self.right[nd])
                node[act] = nxt
                act = act[~self.leaf[nxt]]
            nm = len(self.models)
            sums = np.bincount(self.tree_model[tt] * n + rr, weights=self.value[node],
                               minlength=nm * n).reshape(nm, n)
        for name in names:
            m = self.models[self.mi[name]]
            y = np.full(n, NAN)
            rows = sel_rows[name]
            if len(rows):
                y[rows] = sums[self.mi[name], rows] if t_list else 0.0
                if m.get("base"):
                    y[rows] += X[rows, FI[m["base"]]]
            out[name] = y
        return out


# ----------------------------------------------------------------------------- 6. LP
class _LP:
    """min Σ_k Σ_j (w_k/K)·e_kj + ε·Σ(c+d) − v_T·s_{H−1}
    s.a. c_k − d_k − e_kj ≤ −n_kj (sin premio por verter), s_k − s_{k−1} − η·c_k + d_k/η = 0,
         0 ≤ c,d ≤ P (paso 0: límites del Safety Clamp), 0 ≤ s ≤ C, e ≥ 0.
    Variables x = [c (H) | d (H) | s (H) | e (H·K)]; matrices CSR en caché por (H, K, C, P)."""

    def __init__(self):
        self.cache = {}

    def _mats(self, H, K, cap, pmax):
        key = (H, K, cap, pmax)
        m = self.cache.get(key)
        if m is None:
            I = sparse.identity(H, format="csr")
            D = I - sparse.eye(H, k=-1, format="csr")
            R = sparse.kron(I, np.ones((K, 1)), format="csr")
            A_ub = sparse.hstack([R, -R, sparse.csr_matrix((H * K, H)), -sparse.identity(H * K, format="csr")],
                                 format="csr")
            A_eq = sparse.hstack([-ETA * I, I / ETA, D, sparse.csr_matrix((H, H * K))], format="csr")
            n = 3 * H + H * K
            bounds = np.zeros((n, 2))
            bounds[:2 * H, 1] = pmax
            bounds[2 * H:3 * H, 1] = cap
            bounds[3 * H:, 1] = np.inf
            m = (A_ub, A_eq, bounds)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[key] = m
        return m

    def solve(self, w, N, soc, v_T, cap, pmax, c0_max, d0_max) -> float:
        H, K = N.shape
        A_ub, A_eq, bounds = self._mats(H, K, cap, pmax)
        b = bounds.copy()
        b[0, 1] = c0_max
        b[H, 1] = d0_max
        c = np.zeros(3 * H + H * K)
        c[:2 * H] = EPS_LP
        c[3 * H - 1] = -v_T
        c[3 * H:] = np.repeat(np.asarray(w, dtype=float) / K, K)
        b_eq = np.zeros(H)
        b_eq[0] = soc
        sol = linprog(c, A_ub=A_ub, b_ub=-N.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=b, method="highs",
                      options={"presolve": False})
        if sol.status != 0 or sol.x is None:
            raise RuntimeError("LP")
        return float(sol.x[0] - sol.x[H])


def _vec(x) -> np.ndarray:
    if x is None:
        return np.zeros(0)
    try:
        return np.asarray(list(x), dtype=float).ravel()
    except (TypeError, ValueError):
        try:
            return np.asarray([_num(e) for e in x], dtype=float)
        except TypeError:
            return np.zeros(0)


_BASE_DT = datetime.datetime(*BASE_YMDH)


def _to_dt(ts):
    """pd.Timestamp, datetime, np.datetime64 o texto → datetime sin zona (hora local)."""
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    elif isinstance(ts, np.datetime64):
        ts = datetime.datetime.fromisoformat(str(ts.astype("datetime64[s]")))
    elif isinstance(ts, str):
        ts = datetime.datetime.fromisoformat(ts.strip().replace("T", " ")[:19])
    if not isinstance(ts, datetime.datetime):
        if isinstance(ts, datetime.date):
            ts = datetime.datetime(ts.year, ts.month, ts.day)
        else:
            return None
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


# ----------------------------------------------------------------------------- 7. agente
class BatteryAgent:
    """MPC con previsiones: memoria horaria → variables → bosques → escenarios de carga neta → LP de 24 h
    → primera acción → Safety Clamp. Detecta el modo (Estándar/Difícil) en la primera llamada."""

    def __init__(self):
        self.mode = None
        self.cap, self.pmax, self.S, self.lam = 2000.0, 500.0, 1.0, LAM["std"]
        self.prev_idx = None
        self.prev_soc_pred = None
        self.kmax_seen = 0
        self.p1_reset = False
        self.nov_seen = False
        self.next_cache = {}
        self.elapsed = 0.0
        self.lp = _LP() if linprog is not None else None
        self.ok = False
        try:
            M, C, arrs, params = load_bundle()
            self.M, self.C, self.params = M, C, params
            self.forest = Forest(arrs, params["models"])
            self.Z = {m: np.asarray(arrs["Z_" + m], dtype=float) for m in ("hard", "std")}
            if Z_FINE:
                q9 = (2 * np.arange(1, 10) - 1) / 18.0
                qf = (2 * np.arange(1, Z_FINE + 1) - 1) / (2.0 * Z_FINE)
                lq9 = np.log(q9 / (1 - q9))
                lqf = np.log(qf / (1 - qf))
                def _fine(z):
                    out = np.interp(lqf, lq9, z)
                    lo, hi = lqf < lq9[0], lqf > lq9[-1]
                    out[lo] = z[0] + (lqf[lo] - lq9[0]) * (z[1] - z[0]) / (lq9[1] - lq9[0])
                    out[hi] = z[-1] + (lqf[hi] - lq9[-1]) * (z[-1] - z[-2]) / (lq9[-1] - lq9[-2])
                    return out
                self.Z = {m: np.array([_fine(r) for r in v]) for m, v in self.Z.items()}
            self.S0 = {m: np.asarray(arrs["S0_" + m], dtype=float) for m in ("hard", "std")}
            self.prof = params.get("profiles", {})
            self.blocks = [tuple(int(v) for v in b) for b in (BLOCKS if BLOCKS is not None else params["blocks"])]
            self.default_blocks = BLOCKS is None
            self.n_idx = self.M["load"].shape[0]
            self.ok = True
        except Exception:
            self.M = self.C = self.forest = None
            self.blocks, self.default_blocks, self.prof = [], True, {}
            self.n_idx = 0

    # ------------------------------------------------------------------ entrada / salida
    def act(self, obs) -> float:
        t0 = datetime.datetime.now()
        try:
            return self._act(obs)
        finally:
            self.elapsed += (datetime.datetime.now() - t0).total_seconds()

    def _act(self, obs) -> float:
        try:
            soc, k_obs, vp, vc = self._read(obs)
        except Exception:
            return 0.0
        if soc is None:
            self.prev_soc_pred = None
            return 0.0
        try:
            p = self._decide(obs, soc, k_obs, vp, vc)
        except Exception:
            try:
                p = self._rule_obs(obs, soc, vp, vc)
            except Exception:
                p = 0.0
        p = self._clamp(_num(p, 0.0), soc)
        self.prev_soc_pred = soc + (p * ETA if p >= 0 else p / ETA)
        return float(p)

    def _limits(self, soc):
        max_c = max(0.0, min(self.pmax, (self.cap - soc) / ETA - EPS_CLAMP))
        max_d = max(0.0, min(self.pmax, soc * ETA - EPS_CLAMP))
        return max_c, max_d

    def _clamp(self, p, soc):
        max_c, max_d = self._limits(soc)
        p = max(-max_d, min(max_c, p))
        return 0.0 if abs(p) < 1e-3 else p

    def _read(self, obs):
        """Lee capacidad, potencia y SOC; detecta el modo en la primera llamada."""
        vp, vc = _vec(obs.get("preus_properes_24h")), _vec(obs.get("co2_properes_24h"))
        k_obs = int(min(len(vp), len(vc)))
        soc_kwh, soc_pct = _num(obs.get("soc_kwh")), _num(obs.get("soc_pct"))
        cap = _num(obs.get("cap_max_kwh"))
        if not cap > 0:
            cap = soc_kwh / soc_pct if (soc_pct > 0.01 and soc_kwh > 0) else NAN
        if self.mode is None:
            if not cap > 0:
                cap = 6000.0 if 0 < k_obs <= 3 else 2000.0
            self.mode = "hard" if cap >= 4000.0 else "std"
            self.S = 12.0 if self.mode == "hard" else 1.0
            self.lam = LAM[self.mode]
        if not cap > 0:
            cap = 6000.0 if self.mode == "hard" else 2000.0
        # una capacidad deducida de soc_kwh/soc_pct arrastra error de redondeo (2000,0000000000002), y una
        # potencia 1e-13 kW por encima de la nominal ya cuenta como recorte en el simulador
        cap = round(cap, 3)
        pmax = _num(obs.get("power_max_kw"))
        if not pmax > 0:
            pmax = {2000.0: 500.0, 6000.0: 1500.0}.get(cap, round(cap / 4.0, 3))
        self.cap = cap
        self.pmax = pmax
        soc = soc_kwh
        if math.isnan(soc) and not math.isnan(soc_pct):
            soc = soc_pct * cap
        if math.isnan(soc):
            pc = _num(obs.get("soc_percent"))
            soc = pc / 100.0 * cap if not math.isnan(pc) else NAN
        if math.isnan(soc) and self.prev_soc_pred is not None:
            soc = self.prev_soc_pred
        if math.isnan(soc):
            return None, k_obs, vp, vc
        return min(self.cap, max(0.0, soc)), k_obs, vp, vc

    # ------------------------------------------------------------------ bloques
    def _block_of(self, idx):
        for i, (a, b) in enumerate(self.blocks):
            if a <= idx <= b:
                return i
        return None

    def _is_last_block(self, bi) -> bool:
        if bi is None:
            return False
        if bi == len(self.blocks) - 1:
            return True
        if self.default_blocks and JULY_LAST_IF_NO_NOV and not self.nov_seen:
            d = _BASE_DT + datetime.timedelta(hours=int(self.blocks[bi][0]))
            return (d.year, d.month) == (2026, 7)
        return False

    # ------------------------------------------------------------------ decisión
    def _observe(self, obs, idx, k_obs, vp, vc, bi):
        tel = obs.get("noves_dades_5min")
        tel = tel if isinstance(tel, dict) else {}
        M = self.M
        if self.prev_idx is not None and self.prev_idx < idx and 0 <= self.prev_idx < self.n_idx:
            k = self.prev_idx
            for var, key, scale in (("load", "ultim_consum_kw", 1.0), ("pv", "ultima_solar_kw", self.S),
                                    ("irr", "irradiancia_wm2", 1.0), ("tamb", "temp_ambient_c", 1.0)):
                v = _num(tel.get(key))
                if not math.isnan(v):
                    M[var][k] = v / scale
        end = self.blocks[bi][1] if bi is not None else idx + k_obs - 1
        for i in range(k_obs):
            j = idx + i
            if j > end or j >= self.n_idx:
                break
            if not math.isnan(vp[i]):
                M["price"][j] = vp[i]
            if not math.isnan(vc[i]):
                M["co2"][j] = vc[i]

    def _decide(self, obs, soc, k_obs, vp, vc) -> float:
        dt = _to_dt(obs.get("timestamp"))
        if dt is None or not self.ok or self.lp is None:
            return self._rule_obs(obs, soc, vp, vc)
        idx = int(round((dt - _BASE_DT).total_seconds() / 3600.0))
        if not (24 <= idx < self.n_idx):
            return self._rule_obs(obs, soc, vp, vc)
        if dt.month == 11:
            self.nov_seen = True
        new_block = self.prev_idx is None or idx - self.prev_idx > 72
        if new_block and self.prev_idx is not None and self.prev_soc_pred is not None:
            if abs(soc - self.prev_soc_pred) > 1.0 and abs(soc - 0.5 * self.cap) < 1.0:
                self.p1_reset = True
        bi = self._block_of(idx)
        self._observe(obs, idx, k_obs, vp, vc, bi)
        if self.prev_idx is None or idx > self.prev_idx:
            self.prev_idx = idx
        self.kmax_seen = max(self.kmax_seen, k_obs)

        # horizonte
        left = (self.blocks[bi][1] - idx + 1) if bi is not None else H_PLAN
        ending = 0 < k_obs < self.kmax_seen
        if ending:
            H, v_zero, cross = max(1, k_obs), True, False
        elif left < H_PLAN and (self.p1_reset or self._is_last_block(bi)):
            H, v_zero, cross = left, True, False
        else:
            H, v_zero = H_PLAN, False
            cross = CROSS_BLOCK and left < H_PLAN and bi is not None and bi + 1 < len(self.blocks)
            if left < H_PLAN and not cross:
                H = left
        n_cur = min(H, left)

        mode = self.mode
        f, sig, reg = self._forecast(idx, n_cur, k_obs)
        if cross and H > n_cur:
            g = self._next_block(bi + 1)
            m = H - n_cur
            for k in ("load", "pv", "price", "co2"):
                f[k] = np.concatenate([f[k], g[k][:m]])
            f["exact"] = np.concatenate([f["exact"], np.zeros(m, bool)])
            sig = np.concatenate([sig, g["sig"][:m]])
            reg = np.concatenate([reg, g["reg"][:m]])
        H = len(f["load"])
        if TRUTH is not None:
            for k in range(H):
                tr = TRUTH.get(idx + k)
                if tr is None:
                    continue
                if ("net" in TRUTH_KEYS or ("net0" in TRUTH_KEYS and k == 0)):
                    f["load"][k], f["pv"][k] = tr[0], tr[1]
                    sig[k] = 0.0
                if "price" in TRUTH_KEYS:
                    f["price"][k] = tr[2]
                if "co2" in TRUTH_KEYS or ("co2a" in TRUTH_KEYS and k < 8) or ("co2b" in TRUTH_KEYS and k >= 8):
                    f["co2"][k] = tr[3]
        price, co2 = f["price"][:H].copy(), f["co2"][:H].copy()
        nk = min(k_obs, H)
        price[:nk] = np.where(np.isfinite(vp[:nk]), vp[:nk], price[:nk])
        co2[:nk] = np.where(np.isfinite(vc[:nk]), vc[:nk], co2[:nk])
        self._fill_market(price, co2, idx)
        if nk > 0:
            for arr, key in ((price, "price"), (co2, "co2")):
                a_ = MKT_ALPHA[key]
                if a_ != 1.0:
                    arr[nk:] = arr[nk - 1] + a_ * (arr[nk:] - arr[nk - 1])
        w = co2 + self.lam * price
        net = f["load"] - self.S * f["pv"]
        if NIGHT_EXACT:
            sig = np.where(f["exact"], 0.0, sig)
        l1 = self.M["load"][idx - 1]
        if math.isfinite(l1) and l1 < 1.0:
            net[0] = -self.S * f["pv"][0]          # corte del contador: persiste en la hora actual
            sig[0] = 0.0
        z = self.Z[mode] + Z_SHIFT[mode]
        sig = sig * SIG_MULT[mode]
        N = net[:, None] + sig[:, None] * z[reg.astype(np.int64)]
        K = K_SCEN
        if self.elapsed > TIME_HARD_S:
            return self._rule(w, net, sig, soc)
        if self.elapsed > TIME_MID_S:
            K = 1
        elif self.elapsed > TIME_SOFT_S:
            K = 3
        if K < N.shape[1]:
            cols = {1: [4], 3: [1, 4, 7], 5: [0, 2, 4, 6, 8]}.get(K, list(range(N.shape[1])))
            N = N[:, cols]
        if v_zero:
            v_T = 0.0
        else:
            v_T = float(np.quantile(w, V_Q[mode])) * (ETA if V_ETA[mode] else 1.0)
        c0, d0 = self._limits(soc)
        t_lp = time.monotonic()
        try:
            p = self.lp.solve(w, N, soc, v_T, self.cap, self.pmax, c0, d0)
        except Exception:
            p = self._rule(w, net, sig, soc)
        lp_ms = (time.monotonic() - t_lp) * 1000

        # ── RawTree logging (offline-safe) ───────────────────────────────
        try:
            pfloat = float(p) if p is not None else 0.0
            _rt.log_decision(
                timestamp=obs.get("timestamp", ""),
                idx=idx, soc_kwh=soc, soc_pct=soc / self.cap * 100 if self.cap > 0 else 0,
                cap_kwh=self.cap, power_kw=pfloat,
                price=float(price[0]) if len(price) else 0, co2=float(co2[0]) if len(co2) else 0,
                load_net=float(net[0]) if len(net) else 0,
                mode=mode, horizon=H, scenarios=N.shape[1],
                lp_time=lp_ms / 1000, elapsed_total=self.elapsed,
            )
        except Exception:  # pragma: no cover  (no bloquea si fallan)
            pass

        return p

    def _forecast(self, idx, n, k_obs):
        """Previsión de las n próximas horas del bloque actual + σ y régimen (0 noche, 1 día)."""
        meteo = self.mode == "hard"
        X = build_features(self.M, self.C, idx, max(1, min(k_obs, 3)) if k_obs else 3, meteo=meteo)
        sname = f"sigma_{self.mode}"
        f = point_forecast(self.forest, X, self.C, self.M, idx, market=n > k_obs, n=n, k_known=k_obs,
                           extra=(sname,))
        sig = self._sigma(X, f[sname])
        reg = (X[:, FI["cs_tau"]] > 0).astype(np.int64)
        out = {k: f[k][:n] for k in ("load", "pv", "exact")}
        for k in ("price", "co2"):
            out[k] = f[k][:n] if k in f else np.full(n, NAN)
        self._fill_sensors(out, X[:n])
        return out, sig[:n], reg[:n]

    def _sigma(self, X, s_model):
        """σ de la carga neta por paso. "horizon": la del modelo a cada horizonte; "decision": la de una
        hora (tabla por hora del día y régimen), porque cada hora se vuelve a decidir con el error de h = 0."""
        if SPREAD_MODE == "decision":
            hour = X[:, FI["hour_tau"]].astype(np.int64)
            reg = (X[:, FI["cs_tau"]] > 0).astype(np.int64)
            sig = self.S0[self.mode][hour, reg].copy()
            if math.isfinite(s_model[0]):
                sig[0] = s_model[0]
        else:
            sig = np.asarray(s_model, dtype=float)
        sig = np.where(np.isfinite(sig), sig, 100.0)
        return np.maximum(sig, 5.0)

    def _next_block(self, bi):
        """Previsión de las 24 primeras horas del bloque bi con la historia anterior a su inicio
        (sensores hasta tb−1; mercado con ancla en tb−1). Se calcula una vez por bloque."""
        g = self.next_cache.get(bi)
        if g is not None:
            return g
        tb = self.blocks[bi][0]
        meteo = self.mode == "hard"
        Xs = build_features(self.M, self.C, tb, 3, hs=np.arange(24), meteo=meteo)
        sname = f"sigma_{self.mode}"
        fs = point_forecast(self.forest, Xs, self.C, self.M, tb, market=False, extra=(sname,))
        Xm = build_features(self.M, self.C, tb - 3, 3, hs=np.arange(3, 27), meteo=meteo)
        fm = self.forest.predict(Xm, list(MKT_MODELS))
        price = _bands(fm, ("price@3", "price@8"), 24)
        co2 = _bands(fm, ("co2@3", "co2@8"), 24)
        g = {"load": fs["load"], "pv": fs["pv"], "price": price, "co2": co2}
        self._fill_sensors(g, Xs)
        self._fill_market(g["price"], g["co2"], tb)
        g["sig"] = self._sigma(Xs, fs[sname])
        g["reg"] = (Xs[:, FI["cs_tau"]] > 0).astype(np.int64)
        self.next_cache[bi] = g
        return g

    def _fill_sensors(self, f, X):
        bad = ~np.isfinite(f["load"])
        if bad.any():
            lv = X[:, FI["lvl_tau"]]
            f["load"] = np.where(bad, np.where(np.isfinite(lv), lv, 1000.0), f["load"])
        f["pv"] = np.where(np.isfinite(f["pv"]), np.maximum(f["pv"], 0.0), 0.0)
        f["load"] = np.maximum(f["load"], 0.0)

    def _fill_market(self, price, co2, idx):
        for arr, key, default in ((price, "price", 0.12), (co2, "co2", 0.08)):
            bad = ~np.isfinite(arr)
            if bad.any():
                for k in np.flatnonzero(bad):
                    j = min(idx + int(k), self.C["hour"].shape[0] - 1)
                    try:
                        wk = 1 if self.C["dow"][j] >= 5 else 0
                        arr[k] = float(self.prof[key][wk][int(self.C["hour"][j])])
                    except Exception:
                        arr[k] = default
                arr[~np.isfinite(arr)] = default

    # ------------------------------------------------------------------ reglas de reserva
    def _rule(self, w, net, sig, soc) -> float:
        """Regla determinista: cargar a fondo en el 20 % de horas más limpias/baratas y descargar hasta la
        carga neta prudente en el 30 % más caro."""
        w = np.asarray(w, dtype=float)
        lo, hi = np.quantile(w, 0.20), np.quantile(w, 0.70)
        if w[0] <= lo:
            return self.pmax
        if w[0] >= hi:
            return -min(self.pmax, max(0.0, float(net[0] - sig[0])))
        return 0.0

    def _rule_obs(self, obs, soc, vp, vc) -> float:
        """Sin modelos ni fecha válida: la regla con el mercado de obs y la carga neta de la última hora."""
        n = min(len(vp), len(vc))
        if n == 0:
            return 0.0
        w = vc[:n] + self.lam * vp[:n]
        if not np.isfinite(w).all() or n < 3:
            return 0.0
        tel = obs.get("noves_dades_5min")
        tel = tel if isinstance(tel, dict) else {}
        net0 = _num(tel.get("ultim_consum_kw"), 0.0) - _num(tel.get("ultima_solar_kw"), 0.0)
        return self._rule(w, np.array([net0]), np.array([0.25 * abs(net0) + 50.0]), soc)
