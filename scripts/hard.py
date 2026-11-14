"""Mode Difícil: entrenar los modelos de previsión y evaluar el agente en local.

Uso:
    python scripts/hard.py train            # modelos con train sin validación -> submission_dificil/
    python scripts/hard.py train --all      # modelos con todos los datos (para entregar)
    python scripts/hard.py eval [--oracle]  # score agregado en validación, semana a semana
    python scripts/hard.py eval --continuo  # una sola simulación seguida (como dice el FAQ oficial)
    python scripts/hard.py errors           # MAE de previsión por horizonte en validación
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datathon.data import load_hourly, split  # noqa: E402
from datathon.hard import evaluate_hard, oracle_hard  # noqa: E402
from datathon.hard_train import AGENT_DIR, forecast_errors, load_agent_module, train  # noqa: E402


def end_hook(mod):
    """En local, los bloques de validación no acaban a fin de mes: le decimos al agente dónde
    acaba cada bloque (en el servidor, el último día del mes)."""
    def hook(g):
        last = int(g.index[-1].value // 3_600_000_000_000)
        mod.episode_end_key = lambda t, k=last: k
    return hook


def set_block_ends(mod, df):
    """Simulación continua: el agente recibe el final del bloque (semana) en el que está."""
    ends = {}
    for per, g in df.groupby(df.index.to_period("M")):
        ends[(per.year, per.month)] = int(g.index[-1].value // 3_600_000_000_000)
    mod.episode_end_key = lambda t, e=ends: e.get((t.year, t.month))


def run_eval(df, oracle=False, agent_path=AGENT_DIR / "agent.py", month_end=False, continuous=False, **kw):
    mod = load_agent_module(Path(agent_path))
    if continuous:
        if not month_end:
            set_block_ends(mod, df)
        r = evaluate_hard(mod.BatteryAgent, df, continuous=True, **kw)
    else:
        r = evaluate_hard(mod.BatteryAgent, df, week_hook=None if month_end else end_hook(mod), **kw)
    print(f"agente   {r['score_oficial']:7.3f}  ΔCO2 {r['co2_estalviat_pct']:6.2f}%  "
          f"ΔCoste {r['cost_estalviat_pct']:6.2f}%  infr {r['infraccions']}  {r['segundos']:.1f}s")
    print("  " + "  ".join(f"{m}:{s:.2f}" for m, s in r["por_semana"].items()))
    if oracle:
        o = oracle_hard(df, continuous=continuous)
        print(f"oráculo  {o['score_oficial']:7.3f}  ΔCO2 {o['co2_estalviat_pct']:6.2f}%  ΔCoste {o['cost_estalviat_pct']:6.2f}%")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "eval", "errors"])
    ap.add_argument("--all", action="store_true", help="train: usar todos los datos")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--agent", default=str(AGENT_DIR / "agent.py"))
    ap.add_argument("--month-end", action="store_true",
                    help="eval: no avisar del final de bloque (el agente asume fin de mes)")
    ap.add_argument("--continuo", action="store_true",
                    help="eval: una sola simulación con todas las semanas seguidas (FAQ oficial)")
    args = ap.parse_args()
    df = load_hourly()
    tr, val = split(df)
    if args.cmd == "train":
        train(df if args.all else tr)
    elif args.cmd == "eval":
        run_eval(val, args.oracle, args.agent, args.month_end, args.continuo)
    else:
        print(forecast_errors(val).round(4).to_string())


if __name__ == "__main__":
    main()
