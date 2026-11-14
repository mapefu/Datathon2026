"""¿Ayudaría un modelo de predicción mejor? Cota superior con predicciones perfectas.

Compara el agente con versiones "tramposas" que conocen el futuro real (λ de la semana,
precio/CO2 más allá de 24 h, carga neta) y con el oráculo de la semana entera.

Uso:
    python scripts/headroom.py                  # 11 semanas de validación (~15 s)
    python scripts/headroom.py --step 3         # ventanas de 7 días cada 3 días de todo el año (~2 min)
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from datathon.data import load_hourly, split  # noqa: E402
from datathon.headroom import headroom_week, weeks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", nargs="?", default=str(ROOT / "submission" / "agent.py"))
    ap.add_argument("--step", type=int, default=None, help="ventanas deslizantes sobre todo el año")
    args = ap.parse_args()

    df = load_hourly()
    data = split(df)[1] if args.step is None else df
    ws = weeks(data, args.step)
    with Pool() as p:
        rows = p.map(headroom_week, [(args.agent, label, g) for label, g in ws])
    r = pd.DataFrame(rows).set_index("semana")

    pd.set_option("display.width", 250)
    if args.step is None:
        print(r.round(2).to_string(), "\n")
    print(f"{len(r)} semanas, cada una evaluada por separado (SOC inicial 50 %). Media y ganancia sobre el agente:")
    for c in r.columns:
        d = r[c] - r["agente"]
        print(f"  {c:30s} {r[c].mean():7.3f}   {d.mean():+.4f} (máx. {d.max():+.3f})")


if __name__ == "__main__":
    main()
