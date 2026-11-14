"""Fase 2 · Las variables no leen el futuro.

Para 400 horas de decisión aleatorias: alterar los sensores con índice ≥ t, el mercado con índice ≥ t+3
y los sensores anteriores a `floor` no cambia ninguna variable. También comprueba la regla de la noche.

    .venv/bin/python tests/features_causal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dd.data import load_full  # noqa: E402
from dd.dataset import agent_module, calendar, memory  # noqa: E402
from dd.paths import N_IDX  # noqa: E402


def same(a, b) -> bool:
    return np.array_equal(np.nan_to_num(a, nan=-7.7), np.nan_to_num(b, nan=-7.7))


def main() -> None:
    ag = agent_module()
    full = load_full()
    M, C = memory(full), calendar(full)
    rng = np.random.default_rng(3)
    bad = 0
    for t in rng.integers(200, N_IDX - 30, 400):
        floor = int(t - rng.integers(0, 100)) if rng.random() < 0.5 else None
        X = ag.build_features(M, C, int(t), 3, floor=floor)
        M2 = {k: v.copy() for k, v in M.items()}
        for k in ("load", "pv", "irr", "tamb"):
            M2[k][t:] = rng.normal(500, 300, N_IDX - t)
            if floor is not None:
                M2[k][:floor] = rng.normal(500, 300, floor)
        for k in ("price", "co2"):
            M2[k][t + 3:] = rng.random(N_IDX - t - 3)
        if not same(X, ag.build_features(M2, C, int(t), 3, floor=floor)):
            bad += 1
    print(f"decisiones con fuga: {bad} de 400")
    ok = bad == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
