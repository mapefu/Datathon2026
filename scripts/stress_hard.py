"""Mode Difícil: robustez del agente frente a observaciones degradadas (validación).

Uso: python scripts/stress_hard.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCENARIOS = [
    ("Limpio", {}, {}),
    ("Sin meteo en obs", {"drop_meteo": True}, {}),
    ("Sin cap_max_kwh / power_max_kw", {"drop_spec": True}, {}),
    ("NaN 5 % + timestamp string", {"nan_prob": 0.05, "str_timestamp": True}, {}),
    ("NaN 20 %", {"nan_prob": 0.20}, {}),
    ("Vectores como np.array", {"as_array": True}, {}),
    ("Obs malformada (hora 50)", {"bad_obs_at": 50}, {}),
    ("Sin soc_kwh", {"drop_soc": True}, {}),
    ("Mismo agente todas las semanas", {}, {"fresh_agent": False}),
    ("Sin aviso de fin de bloque", {}, {"month_end": True}),
    ("Simulación continua (FAQ oficial)", {}, {"continuous": True}),
    ("Continua + NaN 5 % + timestamp string", {"nan_prob": 0.05, "str_timestamp": True}, {"continuous": True}),
]


def run(args):
    name, env_kw, kw = args
    from datathon.data import load_hourly, split
    from datathon.hard import PerturbedHardEnv, evaluate_hard
    from datathon.hard_train import load_agent_module
    from hard import end_hook, set_block_ends
    _, val = split(load_hourly())
    mod = load_agent_module()
    kw = dict(kw)
    month_end = kw.pop("month_end", False)
    if kw.get("continuous"):
        hook = None
        if not month_end:
            set_block_ends(mod, val)
    else:
        hook = None if month_end else end_hook(mod)
    r = evaluate_hard(mod.BatteryAgent, val, env_cls=PerturbedHardEnv, week_hook=hook, **kw, **env_kw)
    return name, r["score_oficial"], r["infraccions"], r["segundos"]


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(run, SCENARIOS))
    print(f"{'escenario':34s} {'score':>7s} {'infr':>5s} {'seg':>5s}")
    for name, sc, inf, sec in rows:
        print(f"{name:34s} {sc:7.3f} {inf:5d} {sec:5.1f}")


if __name__ == "__main__":
    main()
