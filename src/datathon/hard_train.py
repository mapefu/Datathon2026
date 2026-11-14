"""Entrenamiento de los modelos de previsión del Mode Difícil.

Para que las variables sean idénticas en entrenamiento e inferencia, se reproduce el agente:
se recorren bloques de 168 h (memoria nueva en cada bloque, como una semana de test) llamando
a `BatteryAgent.observe` y a `History.features` del propio agent.py, y se guardan las
variables junto con los valores reales de las 24 h siguientes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT
from .hard import HardEnv, hard_frame

AGENT_DIR = ROOT / "submission_dificil"
TARGETS = {"load": "load_kw", "pv": "pv_kw", "price": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh"}
LGB_PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
N_ROUNDS = 220

# Sin día del año, mes ni edad: con un solo año de datos, los árboles memorizan cada fecha
# (MAE de consumo 23 kW en train frente a 65 kW en validación).
_CAL = ["h", "hour", "dow", "offday", "hour_t"]
MODEL_SPECS = {
    "load": {"base": "load1", "cols": _CAL + [
        "load1", "load2", "load_yd", "load_md", "load24", "load48", "load168", "load_m24", "load_off_m", "tamb1"]},
    "pv": {"base": None, "cols": _CAL + [
        "cosz", "ghi_cs", "cosz1", "kt1", "kpv1", "pv_cs_pers", "kpv_day", "pv_cs_day", "pv1", "pv24", "pv48", "pv_max24", "pv_sum24", "irr1", "irr24", "irr_max24",
        "tamb1", "tpan1"]},
    "price": {"base": "p2", "cols": _CAL + ["p0", "p1", "p2", "p24", "p48", "p168", "p_m24", "p_shift", "p_d1",
                                            "p_yd", "p_wd", "p_slope", "c_yd", "p_md", "c_md",
                                            "c2", "c_m24", "c24", "kpv_day"]},
    "co2": {"base": "c2", "cols": _CAL + ["c0", "c1", "c2", "c24", "c48", "c168", "c_m24", "c_shift", "c_d1",
                                          "c_yd", "c_wd", "c_slope", "p_yd", "c_md", "p_md",
                                          "p2", "p_m24", "p24", "kpv_day", "kt1"]},
}


def load_agent_module(path: Path = AGENT_DIR / "agent.py"):
    spec = importlib.util.spec_from_file_location(f"hard_agent_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def blocks(df: pd.DataFrame, length: int = 168, offset: int = 0) -> list[pd.DataFrame]:
    """Tramos contiguos (sin huecos) troceados en bloques de `length` horas."""
    gap = df.index.to_series().diff() != pd.Timedelta(hours=1)
    out = []
    for _, seg in df.groupby(gap.cumsum()):
        start = offset % length if len(seg) > length else 0
        if start:
            out.append(seg.iloc[:start])
        for i in range(start, len(seg), length):
            out.append(seg.iloc[i:i + length])
    return [b for b in out if len(b) >= 24]


def replay(mod, block: pd.DataFrame, H: int):
    """Variables y objetivos de cada paso del bloque, tal como los vería el agente."""
    agent = mod.BatteryAgent.__new__(mod.BatteryAgent)
    agent.hist = mod.History()
    agent.prev_key = None
    env = HardEnv(block)
    obs = env.reset()
    full = block
    keys = (full.index - pd.Timestamp(0)) // pd.Timedelta(hours=1)
    pos = {int(k): i for i, k in enumerate(keys)}
    Xs, Ys = [], []
    t = 0
    while obs is not None:
        key, ts, _, _ = agent.observe(obs)
        X = agent.hist.features(key, ts, H)
        Y = np.full((H, len(TARGETS)), np.nan)
        for h in range(H):
            i = pos.get(key + h)
            if i is not None:
                Y[h] = full.iloc[i][list(TARGETS.values())].to_numpy(dtype=float)
        Xs.append(X)
        Ys.append(Y)
        env.t += 1  # avanzamos sin acción: las variables no dependen de la batería
        obs = env._get_obs()
        t += 1
    return np.vstack(Xs), np.vstack(Ys)


def dataset(mod, df: pd.DataFrame, H: int = 24, offsets=(0, 84)):
    import hashlib
    txt = (AGENT_DIR / "agent.py").read_text(encoding="utf-8")
    # solo el código que define las variables (festivos, FEATURES, History, observe)
    src = (txt[txt.index("HOLIDAYS = {"):txt.index("class _Models")]
           + txt[txt.index("    def observe("):txt.index("    def forecast(")]).encode()
    meta = repr((len(df), str(df.index[0]), str(df.index[-1]), H, offsets)).encode()
    tag = hashlib.md5(src + meta).hexdigest()[:12]
    cache = ROOT / "data" / "processed" / f"hard_ds_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["Y"]
    X, Y = _dataset(mod, df, H, offsets)
    np.savez_compressed(cache, X=X, Y=Y)
    return X, Y


def _dataset(mod, df, H, offsets):
    df = hard_frame(df)
    Xs, Ys = [], []
    for off in offsets:
        for b in blocks(df, offset=off):
            X, Y = replay(mod, b, H)
            Xs.append(X)
            Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)


def profiles(df: pd.DataFrame, mod) -> dict:
    """Perfiles de reserva: mes × (laborable/festivo) × hora."""
    df = hard_frame(df)
    off = ((df.index.dayofweek >= 5) | df.index.strftime("%Y-%m-%d").isin(mod.HOLIDAYS)).astype(int)
    out = {}
    for name, col in TARGETS.items():
        tab = {}
        g = df.groupby([df.index.month, off, df.index.hour])[col].mean()
        glob = df.groupby([off, df.index.hour])[col].mean()
        for m in range(1, 13):
            tab[str(m)] = {}
            for o in (0, 1):
                row = []
                for h in range(24):
                    v = g.get((m, o, h), np.nan)
                    if not np.isfinite(v):
                        v = glob.get((o, h), np.nan)
                    row.append(float(v) if np.isfinite(v) else 0.0)
                tab[str(m)][str(o)] = row
        out[name] = tab
    return out


def export_forest(booster) -> dict:
    """Árboles de LightGBM → arrays planos (base64) que el agente evalúa con numpy."""
    import base64

    feat, thr, dleft, mtype, left, right, leaf, roots = [], [], [], [], [], [], [], []
    depth = 0
    mt_code = {"None": 0, "Zero": 1, "NaN": 2}

    def walk(nd, d):
        nonlocal depth
        depth = max(depth, d)
        if "leaf_value" in nd and "split_feature" not in nd:
            leaf.append(float(nd["leaf_value"]))
            return -(len(leaf) - 1) - 1
        if nd.get("decision_type", "<=") != "<=":
            raise ValueError("solo splits numéricos")
        i = len(feat)
        feat.append(int(nd["split_feature"]))
        thr.append(float(nd["threshold"]))
        dleft.append(1 if nd["default_left"] else 0)
        mtype.append(mt_code[nd["missing_type"]])
        left.append(0)
        right.append(0)
        left[i] = walk(nd["left_child"], d + 1)
        right[i] = walk(nd["right_child"], d + 1)
        return i

    for t in booster.dump_model()["tree_info"]:
        roots.append(walk(t["tree_structure"], 0))

    def enc(a, dt):
        return base64.b64encode(np.asarray(a, dtype=dt).tobytes()).decode("ascii")

    return {"feat": enc(feat, np.int32), "thr": enc(thr, np.float64), "dleft": enc(dleft, np.int8),
            "mtype": enc(mtype, np.int8), "left": enc(left, np.int32), "right": enc(right, np.int32),
            "leaf": enc(leaf, np.float64), "roots": enc(roots, np.int32), "depth": depth}


def fit_models(mod, X, Y, rows=None, n_rounds: int = N_ROUNDS, check: bool = True, boosters=None) -> dict:
    """Entrena los cuatro modelos con las filas `rows` (máscara booleana; None = todas)."""
    import lightgbm as lgb

    feats = mod.FEATURES
    hcol = feats.index("h")
    models = {}
    for j, name in enumerate(TARGETS):
        spec = MODEL_SPECS[name]
        cols = [feats.index(c) for c in spec["cols"]]
        y = Y[:, j].copy()
        if spec.get("base"):
            y = y - X[:, feats.index(spec["base"])]
        m = np.isfinite(y)
        if rows is not None:
            m &= rows
        if name in ("price", "co2"):
            m &= X[:, hcol] >= mod.KNOWN
        ds = lgb.Dataset(X[m][:, cols], y[m], feature_name=spec["cols"], free_raw_data=False)
        b = lgb.train(LGB_PARAMS, ds, num_boost_round=n_rounds)
        forest = export_forest(b)
        if check:  # la evaluación en numpy debe coincidir con LightGBM (NaN incluidos)
            Xc = X[m][:, cols][:: max(1, m.sum() // 5000)]
            diff = np.max(np.abs(mod.Forest(forest).predict(Xc) - b.predict(Xc)))
            if diff > 1e-6:
                raise RuntimeError(f"{name}: numpy ≠ LightGBM ({diff})")
        models[name] = {**spec, "forest": forest}
        if boosters is not None:
            boosters[name] = b
    return models


def _lgb_predict(mod, models, boosters, name, X):
    """Predicción con LightGBM (rápida para muchas filas); idéntica a la del agente."""
    spec = models[name]
    cols = [mod.FEATURES.index(c) for c in spec["cols"]]
    y = boosters[name].predict(X[:, cols])
    if spec.get("base"):
        y = y + X[:, mod.FEATURES.index(spec["base"])]
    return y


def _predictor(mod, models: dict):
    m = mod._Models.__new__(mod._Models)
    m.boosters, m.specs, m.profiles, m.lam = {}, {}, {}, mod.LAMBDA
    m.mg, m._cache = None, None
    for name, spec in models.items():
        m.boosters[name] = mod.Forest(spec["forest"])
        m.specs[name] = spec
    return m


def net_quantiles(mod, X, Y, S: int, K: int, n_rounds: int = N_ROUNDS) -> dict:
    """Cuantiles del error de la carga neta (real − prevista) fuera de muestra, por horizonte h<S
    y hora del día. Dos pliegues: meses pares / impares."""
    feats = mod.FEATURES
    month = X[:, feats.index("month")]
    fold = (month % 2).astype(int)
    pred = np.full(len(X), np.nan)
    for f in (0, 1):
        bst = {}
        models = fit_models(mod, X, Y, rows=fold != f, n_rounds=n_rounds, check=False, boosters=bst)
        te = fold == f
        pred[te] = (_lgb_predict(mod, models, bst, "load", X[te]) - _lgb_predict(mod, models, bst, "pv", X[te]))
    true = Y[:, 0] - Y[:, 1]
    r = true - pred
    h = X[:, feats.index("h")].astype(int)
    hour = X[:, feats.index("hour")].astype(int)
    probs = (np.arange(K) + 0.5) / K
    out = {}
    for hh in range(S):
        out[str(hh)] = {}
        for hr in range(24):
            m = (h == hh) & (hour == hr) & np.isfinite(r)
            out[str(hh)][str(hr)] = np.quantile(r[m], probs).round(2).tolist() if m.sum() > 20 else [0.0] * K
    return out, r


SIGMA_COLS = sorted(set(MODEL_SPECS["load"]["cols"]) | set(MODEL_SPECS["pv"]["cols"]), key=lambda c: c)
SIGMA_FLOOR = 5.0


def net_sigma(mod, X, r, S: int, K: int, n_rounds: int = N_ROUNDS) -> tuple[dict, dict]:
    """Modelo de la magnitud del error de carga neta (|r| esperado según las condiciones: cielo
    variable, horizonte, hora...) y cuantiles del error normalizado z = r / σ̂ por horizonte y
    día/noche. Escenarios del agente: previsión + σ̂·z."""
    import lightgbm as lgb

    feats = mod.FEATURES
    cols = [feats.index(c) for c in SIGMA_COLS]
    m = np.isfinite(r)
    ds = lgb.Dataset(X[m][:, cols], np.abs(r[m]), feature_name=SIGMA_COLS, free_raw_data=False)
    b = lgb.train({**LGB_PARAMS, "min_data_in_leaf": 200}, ds, num_boost_round=n_rounds // 2)
    sig = np.maximum(b.predict(X[:, cols]), SIGMA_FLOOR)
    forest = export_forest(b)
    Xc = X[:, cols][:: max(1, len(X) // 5000)]
    if np.max(np.abs(mod.Forest(forest).predict(Xc) - b.predict(Xc))) > 1e-6:
        raise RuntimeError("net_sigma: numpy ≠ LightGBM")
    z = r / sig
    h = X[:, feats.index("h")].astype(int)
    day = (X[:, feats.index("ghi_cs")] > 0).astype(int)
    probs = (np.arange(K) + 0.5) / K
    zq = {}
    for hh in range(S):
        zq[str(hh)] = {}
        for dd in (0, 1):
            mm = (h == hh) & (day == dd) & np.isfinite(z)
            zq[str(hh)][str(dd)] = np.quantile(z[mm], probs).round(4).tolist() if mm.sum() > 50 else [0.0] * K
    return {"base": None, "cols": SIGMA_COLS, "forest": forest, "floor": SIGMA_FLOOR}, zq


def train(df: pd.DataFrame, out_path: Path = AGENT_DIR / "modelos_dificil.json", H: int = 24,
          n_rounds: int = N_ROUNDS, verbose: bool = True) -> dict:
    mod = load_agent_module()
    X, Y = dataset(mod, df, H)
    models = fit_models(mod, X, Y, n_rounds=n_rounds)
    quant, resid = net_quantiles(mod, X, Y, S=mod.N_SCEN_H, K=mod.N_SCEN, n_rounds=n_rounds)
    models["net_sigma"], zq = net_sigma(mod, X, resid, S=mod.N_SCEN_H, K=mod.N_SCEN, n_rounds=n_rounds)
    dfh = hard_frame(df)
    base = np.maximum(dfh.load_kw - dfh.pv_kw, 0)
    lam = 0.3 / 0.7 * float(base @ dfh.co2_kg_per_kwh) / float(base @ dfh.precio_eur_per_kwh)
    data = {"lambda": lam, "features": mod.FEATURES, "models": models, "net_quantiles": quant, "net_zq": zq,
            "profiles": profiles(df, mod)}
    out_path.write_text(json.dumps(data), encoding="utf-8")
    if verbose:
        print(f"λ = {lam:.4f}; guardado {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return data


def forecast_errors(df_val: pd.DataFrame, H: int = 24) -> pd.DataFrame:
    """MAE por horizonte en validación: modelo vs perfil estacional."""
    mod = load_agent_module()
    models = mod._Models()
    X, Y = dataset(mod, df_val, H, offsets=(0,))
    hcol = mod.FEATURES.index("h")
    rows = {}
    for j, name in enumerate(TARGETS):
        pred = models.predict(name, X)
        m = np.isfinite(Y[:, j])
        e = np.abs(pred - Y[:, j])
        rows[name] = pd.Series(e[m]).groupby(X[m, hcol].astype(int)).mean()
    return pd.DataFrame(rows)
