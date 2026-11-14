"""Tabla de escenarios de estrés para uno o varios agentes.

Uso:
    python scripts/stress_test.py                                   # submission/agent.py
    python scripts/stress_test.py submission/agent.py otro/agent.py # comparar versiones
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datathon.data import load_hourly, split  # noqa: E402
from datathon.stress import load_agent_class, oracle_reference, scenarios, timing  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="*", default=[str(ROOT / "submission" / "agent.py")])
    args = ap.parse_args()

    full = load_hourly()
    _, val = split(full)
    agents = {str(Path(p).resolve().relative_to(ROOT)) if Path(p).resolve().is_relative_to(ROOT) else p:
              load_agent_class(p) for p in args.agents}
    ref = oracle_reference(val, full)

    names = list(agents)
    w = 26
    print(f"{'escenario':26s}" + "".join(f"{n[-w:]:>{w}s}" for n in names) + f"{'óptimo':>9s}")
    print("-" * (26 + w * len(names) + 9))
    for label, fn in scenarios(val, full):
        cells = []
        for A in agents.values():
            try:
                score, infr = fn(A)
                cells.append(f"{score:.2f}" + (f" ({infr} infr)" if infr else ""))
            except Exception as e:  # un fallo en el servidor = entrega fallida
                cells.append(f"FALLA: {type(e).__name__}")
        opt = f"{ref[label]:.2f}" if label in ref else "—"
        print(f"{label:26s}" + "".join(f"{c:>{w}s}" for c in cells) + f"{opt:>9s}")
    print(f"{'tiempo 2.016 h (s)':26s}" + "".join(f"{timing(A, val):>{w}.1f}" for A in agents.values()))


if __name__ == "__main__":
    main()
