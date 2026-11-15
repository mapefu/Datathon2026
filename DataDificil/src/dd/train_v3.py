"""Variante experimental de `train.py` para `experimentos/agentes/agent_v3_wdirect.py` (no toca train.py).

Añade al bundle los modelos w@3 / w@8, que prevén directamente w = co2 + λ·precio (λ del Modo Difícil),
la única combinación de mercado que usa el LP. Objetivo = w[τ] − (c_2 + λ·p_2); el agente suma esa base.
Con el mismo bundle, `W_DIRECT=False` en el agente reproduce el agente sin estos modelos, así que la
comparación queda emparejada sin ruido de semilla en el resto de modelos.

    DD_AGENT=experimentos/agentes/agent_v3_wdirect.py .venv/bin/python -m dd.train_v3 --fold A --tag v3w
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import train as T
from .paths import MODES

LAM_W = MODES["hard"]["lam"]
W_SAMPLE_WEIGHT = os.environ.get("DD_W_WEIGHT", "")    # "" | "spread": peso ∝ dispersión diaria de w

_specs0 = T.model_specs
_target0 = T.Rows.target
_design0 = T.Rows.design


def model_specs(ag):
    S = _specs0(ag)
    cols = []
    for s in S:
        if s["name"] in ("price@3", "co2@3"):
            cols += [c for c in s["cols"] if c not in cols]
    # DD_W_OBJ='{"objective": "huber", "alpha": 0.01}': objetivo de los modelos w (como DD_MKT_OBJ en train.py)
    obj = json.loads(os.environ.get("DD_W_OBJ", "{}"))
    for name, lo, hi in (("w@3", 3, 8), ("w@8", 8, 27)):
        S.append(dict(name=name, target="w", h_lo=lo, h_hi=hi, cols=cols, base=None,
                      lgb=dict(num_leaves=31, min_data_in_leaf=100, **obj), rounds=T.ROUNDS))
    return S


def target(self, name, Tt, H, X):
    if name != "w":
        return _target0(self, name, Tt, H, X)
    tau = Tt + H
    FI = self.ag.FI
    return (self.Y["co2"][tau] + LAM_W * self.Y["price"][tau]) - (X[:, FI["c_2"]] + LAM_W * X[:, FI["p_2"]])


def design(self, spec, groups=None, extra=None):
    Xm, y, w = _design0(self, spec, groups, extra)
    if spec["target"] == "w" and W_SAMPLE_WEIGHT == "spread":
        # más peso a las decisiones con mucho ciclo diario de w en juego (rango max24 − min24 de CO2)
        i = spec["cols"].index("c_max24"), spec["cols"].index("c_min24")
        rng = np.nan_to_num(Xm[:, i[0]] - Xm[:, i[1]], nan=np.nanmedian(Xm[:, i[0]] - Xm[:, i[1]]))
        w = w * np.clip(rng / np.median(rng), 0.3, 3.0)
    return Xm, y, w


T.model_specs = model_specs
T.Rows.target = target
T.Rows.design = design


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from .cv import fold_dir, masked_full
    from .data import load_full, load_sim

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", nargs="*", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="directorio de salida de --fold full")
    args = ap.parse_args()
    if args.seed is not None:
        T.LGB_BASE["seed"] = args.seed
    full, sim = load_full(), load_sim()
    for fold in args.fold:
        out = Path(args.out).resolve() if (fold == "full" and args.out) else fold_dir(fold, args.tag)
        print(f"== entrenando pliegue {fold} → {out}", flush=True)
        T.train_bundle(masked_full(full, sim, fold), out, fold, None, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
