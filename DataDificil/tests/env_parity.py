"""Fase 1 · Paridad del simulador y del arnés.

1. Estándar: `SimEnv("std")` da exactamente las mismas obs, red y score que `CampusBatteryEnv` oficial.
2. Difícil: las obs coinciden con el `HardEnv` del repo previo (validado contra la web: 19,88 vs 19,86).
3. Arnés: el agente Estándar previo en las semanas A seguidas (P2) da 6,10 como en su README.

    .venv/bin/python tests/env_parity.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dd.cv import cv_weeks  # noqa: E402
from dd.data import load_sim  # noqa: E402
from dd.env import SimEnv, run_p2, sim_frame, spec_of  # noqa: E402
from dd.score import aggregate  # noqa: E402
from environment import CampusBatteryEnv  # noqa: E402

PREV = Path("/home/marc/Projects/Datathon2026-hardcv")
FAILED = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALLA ") + msg)
    if not cond:
        FAILED.append(msg)


def same_obs(a, b) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, dict):
            if not same_obs(x, y):
                return False
        elif isinstance(x, list):
            if len(x) != len(y) or any(not math.isclose(u, v, rel_tol=0, abs_tol=1e-12) for u, v in zip(x, y)):
                return False
        elif x != y:
            return False
    return True


def rollout(env, actions):
    obs_all = [env.reset()]
    for a in actions:
        o, done = env.step(a)
        if done:
            break
        obs_all.append(o)
    return obs_all


def main():
    sim = load_sim()
    wk = cv_weeks(sim)
    g = sim_frame(wk["A"][3], "std")
    rng = np.random.default_rng(0)
    acts = rng.uniform(-700, 700, len(g))          # incluye peticiones fuera de límites → infracciones
    off = CampusBatteryEnv(g[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]], spec=spec_of("std"))
    mine = SimEnv(g, "std")
    o1, o2 = rollout(off, acts), rollout(mine, acts)
    check(len(o1) == len(o2) and all(same_obs(a, b) for a, b in zip(o1, o2)), "Estándar: obs idénticas a las oficiales")
    check(np.array_equal(off.grid, mine.grid) and off.evaluate_results() == mine.evaluate_results(),
          f"Estándar: misma red y score ({mine.evaluate_results()['score']}, {mine.evaluate_results()['infraccions']} infr.)")

    if (PREV / "src" / "datathon" / "hard.py").exists():
        sys.path.insert(0, str(PREV / "src"))
        from datathon.hard import HardEnv, hard_frame  # type: ignore

        gw = wk["A"][7]
        prev_env, my_env = HardEnv(hard_frame(gw)), SimEnv(sim_frame(gw, "hard"), "hard")
        acts = rng.uniform(-1600, 1600, len(gw))
        p1, p2 = rollout(prev_env, acts), rollout(my_env, acts)
        check(all(same_obs(a, b) for a, b in zip(p1, p2)) and np.array_equal(prev_env.grid, my_env.grid),
              "Difícil: obs y red idénticas al HardEnv previo")
        o = p2[5]
        check(len(o["preus_properes_24h"]) == 3 and o["cap_max_kwh"] == 6000.0
              and "irradiancia_wm2" in o["noves_dades_5min"], "Difícil: 3 h de mercado, cap y meteo en obs")

    # agente_anterior/ es la versión del README (6,10). El submission/agent.py posterior (commit 5d5a489)
    # da 6,035 en este mismo arnés: es una regresión, no un fallo del simulador.
    std_prev = PREV / "agente_anterior" / "agent.py"
    if std_prev.exists():
        r = run_p2(std_prev, wk["A"], "std")
        a = aggregate(r)
        check(abs(a["score"] - 6.10) < 0.02 and a["infr"] == 0,
              f"Arnés: agente Estándar previo en semanas A (P2) = {a['score']:.3f} (README: 6,10), "
              f"{a['infr']} infr., {a['ms_h']:.2f} ms/h")
    print("RESULTADO:", "OK" if not FAILED else f"{len(FAILED)} fallos")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
