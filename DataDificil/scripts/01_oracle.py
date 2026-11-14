"""Fase 1 · Techo con información perfecta (oráculo LP) por modo y protocolo.

    .venv/bin/python scripts/01_oracle.py

Criterios: Difícil 21,45 ± 0,10 (P2) y 22,84 ± 0,10 (P1) en las 22 semanas; Estándar ≈ 6,12 en las 11
semanas A seguidas (la validación del repo previo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dd.cv import cv_weeks  # noqa: E402
from dd.data import load_sim  # noqa: E402
from dd.paths import REPORTS  # noqa: E402
from dd.score import aggregate, oracle  # noqa: E402


def main() -> None:
    sim = load_sim()
    wk = cv_weeks(sim)
    print(f"semanas: A = {len(wk['A'])}, B = {len(wk['B'])}")
    out = REPORTS / "cv"
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for mode in ("std", "hard"):
        orc = oracle(wk, mode)
        (out / f"{mode}_oracle.json").write_text(json.dumps(orc, indent=1), encoding="utf-8")
        for p in ("P1", "P2"):
            a = aggregate(orc[p])
            print(f"{mode:4s} {p} (22 sem): score {a['score']:.3f}  ΔCO2 {a['dco2']:.2f} %  ΔCoste {a['dcost']:.2f} %")
        orcA = oracle({"A": wk["A"]}, mode)
        aA = aggregate(orcA["P2"])
        print(f"{mode:4s} P2 (semanas A): score {aA['score']:.3f}")
        if mode == "hard":
            s1, s2 = aggregate(orc["P1"])["score"], aggregate(orc["P2"])["score"]
            ok &= abs(s1 - 22.84) <= 0.10 and abs(s2 - 21.45) <= 0.10
        else:
            ok &= abs(aA["score"] - 6.12) <= 0.03
    print("ACEPTACIÓN:", "OK" if ok else "FALLA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
