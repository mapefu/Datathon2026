"""Validación cruzada local (P1/P2, semanas A y B) de un agente, comparada con el oráculo y una referencia.

    .venv/bin/python scripts/03_cv.py --agent submission/agent.py --mode hard --name base
    .venv/bin/python scripts/03_cv.py --agent submission/agent.py --mode hard --name x --set SPREAD_MODE=decision --ref base
    .venv/bin/python scripts/03_cv.py --agent ../Datathon2026-hardcv/agente_anterior/agent.py --mode std --name prev --folds A --protocols P2

`--set NOMBRE=valor` fija constantes de módulo del agente (ablaciones). El agente recibe `BUNDLE_DIR` (modelos
del pliegue, si existen en data/processed/folds/<tag>/<pliegue>) y `BLOCKS` (bloques evaluados).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dd.cv import load_result, report, run_cv  # noqa: E402
from dd.data import load_sim  # noqa: E402
from dd.paths import REPORTS  # noqa: E402


def parse_set(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--mode", choices=("std", "hard"), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--folds", nargs="*", default=["A", "B"])
    ap.add_argument("--protocols", nargs="*", default=["P1", "P2"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-time", action="store_true",
                    help="no desactivar el gobernador de tiempo del agente (por defecto se mide el algoritmo, "
                         "no la carga de la máquina; el tiempo se mide aparte en la Fase 4)")
    args = ap.parse_args()
    sim = load_sim()
    overrides = {} if args.keep_time else {"TIME_SOFT_S": 1e12, "TIME_MID_S": 1e12, "TIME_HARD_S": 1e12}
    overrides.update(parse_set(args.set))
    res = run_cv(args.name, Path(args.agent).resolve(), args.mode, sim, args.tag, args.folds, args.protocols,
                 overrides, args.workers)
    orc_path = REPORTS / "cv" / f"{args.mode}_oracle.json"
    orc = json.loads(orc_path.read_text()) if orc_path.exists() else None
    ref = load_result(args.mode, args.ref) if args.ref else None
    print(report(res, ref, orc))
    print(f"({res['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
