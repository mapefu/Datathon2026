"""Evalúa uno o varios agentes en local con el simulador oficial.

Uso:
    python scripts/evaluate.py                          # submission/agent.py en validación
    python scripts/evaluate.py submission/agent.py starter_kit/agent.py --oracle
    python scripts/evaluate.py submission/agent.py --split all
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datathon.data import load_hourly, split  # noqa: E402
from datathon.evaluate import evaluate, oracle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="*", default=[str(ROOT / "submission" / "agent.py")])
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="val = últimos 7 días de cada mes (imita el test oculto)")
    ap.add_argument("--oracle", action="store_true", help="incluye la cota superior con información perfecta")
    ap.add_argument("--rebuild", action="store_true", help="regenera data/processed/hourly.csv")
    args = ap.parse_args()

    df = load_hourly(rebuild=args.rebuild)
    train, val = split(df)
    df = {"val": val, "train": train, "all": df}[args.split]
    print(f"Datos: split={args.split}, {len(df)} horas ({df.index.min()} -> {df.index.max()})\n")

    rows = {}
    for path in args.agents:
        r = evaluate(path, df)
        rows[Path(path).parent.name + "/" + Path(path).name] = r
    if args.oracle:
        rows["ORÁCULO (info perfecta)"] = oracle(df)

    header = f"{'agente':34s} {'score':>7s} {'ΔCO2%':>7s} {'ΔCoste%':>8s} {'infr':>5s} {'score_env':>9s}"
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        print(f"{name:34s} {r['score_oficial']:7.2f} {r['co2_estalviat_pct']:7.2f} "
              f"{r['cost_estalviat_pct']:8.2f} {r.get('infraccions', 0):5d} {r.get('score', float('nan')):9.2f}")

    for name, r in rows.items():
        if "por_mes" in r:
            print(f"\nPor mes — {name}:")
            print("  " + "  ".join(f"{m}:{s:.2f}" for m, s in r["por_mes"].items()))


if __name__ == "__main__":
    main()
