"""Fase 2 · Entrenamiento de los modelos de previsión y del bundle del agente.

Modelos (LightGBM directo multi-horizonte por bandas, `h` como variable, nunca recursivo):
    load@0 / load@1 / load@3   consumo, h = 0 | 1–2 | 3–23, objetivo = load[τ] − L_base
    pv@0 / pv@1 / pv@3         índice de claridad k = pv/cs (×1), ponderado por cs
    price@3 / price@8          h = 3–7 | 8–26, objetivo = precio[τ] − p_2 (último conocido)
    co2@3 / co2@8              ídem con c_2
    sigma_hard / sigma_std     |error| esperado de la carga neta (S = 12 | 1), h = 0–23

Errores fuera de muestra con GroupKFold(3) por semana → modelos sigma y cuantiles z normalizados para los
escenarios del LP. Aumentos: 20 % de decisiones con arranque en frío (sensores anteriores a un inicio de
bloque sorteado) y 30 % de filas de consumo sin meteo (Modo Estándar).

Salida: `bundle.npz` (memoria, calendario, bosque fundido, tablas) + `params.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import agent_module, build_rows, calendar, memory, targets
from .export import flatten
from .paths import MODES, N_IDX, idx_of

LGB_BASE = dict(objective="regression", learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, seed=7, deterministic=True, force_row_wise=True,
                num_threads=2, verbose=-1)
ROUNDS = 300
SIG_ROUNDS = 150
Q_LEVELS = (2 * np.arange(1, 10) - 1) / 18.0          # 9 escenarios equiprobables
COLD_FRAC, NOMETEO_FRAC = 0.20, 0.30


def model_specs(ag) -> list[dict]:
    cal_l = ["h", "hour_tau", "hour_t", "dow_tau", "daytype_tau", "daytype_t", "act_tau", "night_tau",
             "night_anchor"]
    load_cols = cal_l + list(ag.LOAD_FEATS)
    pv_cols = ["h", "hour_tau", "hour_t", "hour_utc_tau"] + list(ag.PV_FEATS)

    def mkt_cols(px, py):
        c = ["h", "hour_tau", "hour_t", "dow_tau", "daytype_tau", "period_tau", "period_a"]
        c += [f"{px}_{s}" for s in ag.MKT_STEMS] + (["p24_adj"] if px == "p" else [])
        return c + [f"{py}_0", f"{py}_2", f"{py}_lvl_d1", "k_day", "k_yday"]

    sig_cols = ["h", "hour_tau", "daytype_tau", "act_tau", "night_tau", "night_anchor", "cs_tau", "k1",
                "dk", "k_day", "L1", "dL"]
    S = []
    for name, lo, hi, nl, md in (("load@0", 0, 1, 15, 40), ("load@1", 1, 3, 31, 60), ("load@3", 3, 24, 31, 100)):
        S.append(dict(name=name, target="load", h_lo=lo, h_hi=hi, cols=load_cols, base="L_base",
                      lgb=dict(num_leaves=nl, min_data_in_leaf=md), rounds=ROUNDS))
    for name, lo, hi, nl, md in (("pv@0", 0, 1, 15, 40), ("pv@1", 1, 3, 31, 60), ("pv@3", 3, 24, 31, 100)):
        S.append(dict(name=name, target="k", h_lo=lo, h_hi=hi, cols=pv_cols, base=None,
                      lgb=dict(num_leaves=nl, min_data_in_leaf=md), rounds=ROUNDS))
    for var, px, py in (("price", "p", "c"), ("co2", "c", "p")):
        for name, lo, hi in ((f"{var}@3", 3, 8), (f"{var}@8", 8, 27)):
            S.append(dict(name=name, target=var, h_lo=lo, h_hi=hi, cols=mkt_cols(px, py), base=f"{px}_2",
                          lgb=dict(num_leaves=31, min_data_in_leaf=100), rounds=ROUNDS))
    for mode in ("hard", "std"):
        S.append(dict(name=f"sigma_{mode}", target=f"abs_r_{mode}", h_lo=0, h_hi=24, cols=sig_cols, base=None,
                      lgb=dict(num_leaves=31, min_data_in_leaf=200), rounds=SIG_ROUNDS))
    return S


def default_blocks() -> list[tuple[int, int]]:
    """Bloques de test esperados: últimos 7 días de cada mes, de sep-2025 a ago-2026."""
    out = []
    for m in pd.period_range("2025-09", "2026-08", freq="M"):
        last = m.to_timestamp(how="end").normalize()
        out.append((idx_of(last - pd.Timedelta(days=6)), idx_of(last + pd.Timedelta(hours=23))))
    return out


class Rows:
    """Filas de entrenamiento: base (todas las decisiones, h = 0..26) + aumentos."""

    def __init__(self, ag, full: pd.DataFrame, log):
        t0 = time.time()
        self.ag = ag
        self.M, self.C, self.Y = memory(full), calendar(full), targets(full)
        FI = ag.FI
        ts = np.arange(N_IDX)
        self.X, self.T, self.H = build_rows(ag, self.M, self.C, ts, 3)
        rng = np.random.default_rng(7)
        cand = ts[np.isfinite(self.Y["load"][ts])]
        cold_t = np.sort(rng.choice(cand, size=int(COLD_FRAC * len(cand)), replace=False))
        floors = cold_t - rng.integers(0, 168, len(cold_t))
        self.Xc, self.Tc, self.Hc = build_rows(ag, self.M, self.C, cold_t, 3, np.arange(24), floors)
        self.cs = self.X[:, FI["cs_tau"]]
        self.group = (self.T // 168) % 3
        self.groupc = (self.Tc // 168) % 3
        self.nometeo = rng.random(len(self.X)) < NOMETEO_FRAC
        log(f"  filas: base {len(self.X):,} + arranque en frío {len(self.Xc):,} "
            f"({len(cold_t)} decisiones) en {time.time() - t0:.0f} s")

    def target(self, name: str, T, H, X):
        tau = T + H
        if name == "load":
            return self.Y["load"][tau]
        if name == "k":
            cs = X[:, self.ag.FI["cs_tau"]]
            return np.where(cs > 5, self.Y["pv"][tau] / np.where(cs > 5, cs, 1.0), np.nan)
        return self.Y[name][tau]

    def design(self, spec: dict, groups=None, extra=None):
        """(X de las columnas del modelo, y, peso) con los aumentos, opcionalmente solo ciertos grupos."""
        ag = self.ag
        cols = [ag.FI[c] for c in spec["cols"]]
        parts = []
        for X, T, H, G, aug in ((self.X, self.T, self.H, self.group, "base"),
                                (self.Xc, self.Tc, self.Hc, self.groupc, "cold")):
            if aug == "cold" and spec["target"] not in ("load", "k"):
                continue                      # el arranque en frío solo afecta a los sensores
            sel = (H >= spec["h_lo"]) & (H < spec["h_hi"])
            if groups is not None:
                sel &= np.isin(G, groups)
            if spec["target"].startswith("abs_r"):
                y = extra[spec["target"]]
            else:
                y = self.target(spec["target"], T, H, X)
            if spec["base"]:
                y = y - X[:, ag.FI[spec["base"]]]
            sel &= np.isfinite(y)
            if spec["target"] in ("price", "co2"):
                # el mercado conocido (h ≤ 2) tiene que existir para que la fila sea realista
                sel &= np.isfinite(X[:, ag.FI["p_0"]]) & np.isfinite(X[:, ag.FI["p_2"]])
            idx = np.flatnonzero(sel)
            Xm = X[np.ix_(idx, cols)]
            w = np.maximum(X[idx, ag.FI["cs_tau"]], 0.0) if spec["target"] == "k" else np.ones(len(idx))
            parts.append((Xm, y[idx], w))
            if spec["target"] == "load" and aug == "base":
                nm = idx[self.nometeo[idx]]
                Xn = X[np.ix_(nm, cols)].copy()
                for f in ag.METEO_FEATS:
                    if f in spec["cols"]:
                        Xn[:, spec["cols"].index(f)] = np.nan
                parts.append((Xn, y[nm], np.ones(len(nm))))
        Xm = np.vstack([p[0] for p in parts])
        return Xm, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])


def fit(spec: dict, Xm, y, w, rounds: int | None = None) -> lgb.Booster:
    params = dict(LGB_BASE, **spec["lgb"])
    ds = lgb.Dataset(Xm, label=y, weight=w, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=rounds or spec["rounds"])


def predict(booster, spec, X, ag, no_meteo: bool = False):
    Xm = X[:, [ag.FI[c] for c in spec["cols"]]]
    if no_meteo:
        Xm = Xm.copy()
        for f in ag.METEO_FEATS:
            if f in spec["cols"]:
                Xm[:, spec["cols"].index(f)] = np.nan
    y = booster.predict(Xm, num_threads=2)
    if spec["base"]:
        y = y + X[:, ag.FI[spec["base"]]]
    return y


def _combine(preds: dict, names, H, n):
    y = np.full(n, np.nan)
    for nm, (lo, hi) in names.items():
        sel = (H >= lo) & (H < hi)
        y[sel] = preds[nm][sel]
    return y


def net_residuals(R: Rows, specs, rounds, log) -> dict:
    """Residuos fuera de muestra (GroupKFold(3) por semana) de la carga neta, para S = 12 y S = 1."""
    ag = R.ag
    sens = [s for s in specs if s["target"] in ("load", "k")]
    n = len(R.X)
    oof = {(s["name"], nm): np.full(n, np.nan) for s in sens for nm in (False, True)}
    for g in range(3):
        held = np.flatnonzero((R.group == g) & (R.H < 24))
        for s in sens:
            Xm, y, w = R.design(s, groups=[x for x in range(3) if x != g])
            b = fit(s, Xm, y, w, rounds)
            del Xm
            band = held[(R.H[held] >= s["h_lo"]) & (R.H[held] < s["h_hi"])]
            for nm in (False, True):
                if s["target"] == "k" and nm:
                    continue
                oof[(s["name"], nm)][band] = predict(b, s, R.X[band], ag, no_meteo=nm)
        log(f"  OOF grupo {g} listo")
    bands_l = {"load@0": (0, 1), "load@1": (1, 3), "load@3": (3, 24)}
    bands_k = {"pv@0": (0, 1), "pv@1": (1, 3), "pv@3": (3, 24)}
    k_hat = _combine({k: oof[(k, False)] for k in bands_k}, bands_k, R.H, n)
    pv_hat = ag.pv_from_k(k_hat, R.X)
    out = {"pv_hat": pv_hat}
    y_load, y_pv = R.Y["load"][R.T + R.H], R.Y["pv"][R.T + R.H]
    for mode, nm in (("hard", False), ("std", True)):
        lh = _combine({k: oof[(k, nm)] for k in bands_l}, bands_l, R.H, n)
        exact = np.zeros(n, bool)
        HF = ag.H_FEAT
        for t in range(N_IDX):
            r0 = t * HF
            lh[r0:r0 + HF], exact[r0:r0 + HF] = ag.night_rule(lh[r0:r0 + HF], R.X[r0:r0 + HF], R.C, t, R.M)
        S = MODES[mode]["S"]
        r = (y_load - lh) - S * (y_pv - pv_hat)
        valid = (R.H < 24) & np.isfinite(r) & ~exact
        out[f"r_{mode}"] = np.where(valid, r, np.nan)
        out[f"load_hat_{mode}"] = lh
        out["exact"] = exact
    return out


def sigma_tables(R: Rows, specs, res: dict, rounds, log) -> tuple[dict, dict, dict]:
    """Modelos sigma (fuera de muestra para los cuantiles z) y tabla sigma(h = 0) por hora y régimen."""
    ag = R.ag
    extra = {f"abs_r_{m}": np.abs(res[f"r_{m}"]) for m in ("hard", "std")}
    z, s0, metrics = {}, {}, {}
    for mode in ("hard", "std"):
        spec = next(s for s in specs if s["name"] == f"sigma_{mode}")
        r = res[f"r_{mode}"]
        sig = np.full(len(r), np.nan)
        for g in range(3):
            Xm, y, w = R.design(spec, groups=[x for x in range(3) if x != g], extra=extra)
            b = fit(spec, Xm, y, w, rounds)
            held = np.flatnonzero((R.group == g) & np.isfinite(r))
            sig[held] = predict(b, spec, R.X[held], ag)
        sig = np.maximum(sig, 5.0)
        ok = np.isfinite(r)
        day = R.cs > 0
        zz = np.zeros((2, 9))
        for reg, m in ((0, ok & ~day), (1, ok & day)):
            zz[reg] = np.quantile(r[m] / sig[m], Q_LEVELS)
        z[mode] = zz
        tab = np.zeros((24, 2))
        hour = R.X[:, ag.FI["hour_tau"]].astype(int)
        for reg, m in ((0, ok & ~day & (R.H == 0)), (1, ok & day & (R.H == 0))):
            fill = float(np.mean(sig[m])) if m.any() else 5.0
            for hh in range(24):
                mm = m & (hour == hh)
                tab[hh, reg] = float(np.mean(sig[mm])) if mm.sum() >= 10 else fill
        s0[mode] = tab
        cover = [float(np.mean(r[ok] <= sig[ok] * np.where(day[ok], zz[1, j], zz[0, j]))) for j in range(9)]
        metrics[mode] = {"mae_h0": float(np.nanmean(np.abs(r[ok & (R.H == 0)]))),
                         "mae_h1_2": float(np.nanmean(np.abs(r[ok & (R.H >= 1) & (R.H <= 2)]))),
                         "mae_h3_23": float(np.nanmean(np.abs(r[ok & (R.H >= 3)]))),
                         "coverage_in_sample": cover}
        log(f"  sigma_{mode}: MAE neta OOF h0 {metrics[mode]['mae_h0']:.1f} kW, h1–2 "
            f"{metrics[mode]['mae_h1_2']:.1f}, h3–23 {metrics[mode]['mae_h3_23']:.1f}")
    return z, s0, {"extra": extra, "metrics": metrics}


def profiles(full: pd.DataFrame) -> dict:
    """Perfiles de reserva de precio y CO2: mediana por (laborable/fin de semana, hora)."""
    out = {}
    wk = (full.index.day_of_week >= 5).astype(int)
    for var in ("price", "co2"):
        g = full[var].groupby([wk, full.index.hour]).median()
        out[var] = [[float(g.get((w, h), np.nan)) for h in range(24)] for w in (0, 1)]
    return out


def train_bundle(full: pd.DataFrame, out_dir: Path, fold: str, rounds: int | None = None,
                 log=print) -> dict:
    """Entrena todo con los datos visibles de `full` y escribe bundle.npz + params.json en out_dir."""
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ag = agent_module()
    specs = model_specs(ag)
    R = Rows(ag, full, log)
    res = net_residuals(R, specs, rounds, log)
    z, s0, sg = sigma_tables(R, specs, res, rounds, log)

    boosters = []
    for s in specs:
        t0 = time.time()
        Xm, y, w = R.design(s, extra=sg["extra"])
        b = fit(s, Xm, y, w, rounds)
        boosters.append(b)
        log(f"  {s['name']:10s} {len(y):7,} filas, {b.num_trees()} árboles ({time.time() - t0:.1f} s)")
        del Xm

    arrs = flatten([(b, [ag.FI[c] for c in s["cols"]]) for b, s in zip(boosters, specs)])
    models = [{"name": s["name"], "target": s["target"], "h_lo": s["h_lo"], "h_hi": s["h_hi"],
               "cols": s["cols"], "base": s["base"], "n_trees": b.num_trees()} for s, b in zip(specs, boosters)]
    # paridad bosque numpy vs LightGBM con NaN aleatorios
    forest = ag.Forest(arrs, models)
    rng = np.random.default_rng(1)
    worst = 0.0
    for s, b in zip(specs, boosters):
        rows = rng.choice(np.flatnonzero((R.H >= s["h_lo"]) & (R.H < s["h_hi"])), 2000, replace=False)
        Xp = R.X[rows].copy()
        Xp[rng.random(Xp.shape) < 0.10] = np.nan
        Xp[:, ag.FI["h"]] = R.X[rows, ag.FI["h"]]
        mine = forest.predict(Xp, [s["name"]], rows_of={s["name"]: np.arange(len(rows))})[s["name"]]
        ref = predict(b, s, Xp, ag)
        ok = np.isfinite(ref)
        worst = max(worst, float(np.max(np.abs(mine[ok] - ref[ok]))))
    log(f"  paridad bosque numpy vs LightGBM: máx |Δ| = {worst:.2e}")
    if worst > 1e-6:
        raise AssertionError(f"paridad del bosque fallida: {worst}")

    bundle = {f"M_{k}": v for k, v in R.M.items()}
    bundle.update({f"C_{k}": v for k, v in R.C.items()})
    bundle.update(arrs)
    bundle.update({"Z_hard": z["hard"], "Z_std": z["std"], "S0_hard": s0["hard"], "S0_std": s0["std"]})
    np.savez_compressed(out_dir / "bundle.npz", **bundle)
    params = {
        "version": 1, "fold": fold, "feats": list(ag.FEATS), "models": models,
        "lam": {m: MODES[m]["lam"] for m in MODES}, "blocks": default_blocks(),
        "profiles": profiles(full), "sigma_floor": 5.0, "q_levels": list(Q_LEVELS),
        "train_metrics": sg["metrics"], "parity_max_abs": worst, "train_seconds": time.time() - t_start,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    size = (out_dir / "bundle.npz").stat().st_size / 2 ** 20
    log(f"  bundle {size:.2f} MB, {len(arrs['F_roots'])} árboles, {len(arrs['F_feat']):,} nodos; "
        f"total {time.time() - t_start:.0f} s")
    return params
