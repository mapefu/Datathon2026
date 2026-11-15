"""Fase 2 · Entrena los modelos de un pliegue (o los finales) y evalúa la previsión fuera de muestra.

    .venv/bin/python scripts/02_train.py --fold A            # modelos sin las semanas A → data/processed/folds/default/A
    .venv/bin/python scripts/02_train.py --fold B
    .venv/bin/python scripts/02_train.py --fold full         # modelos finales → submission/
    .venv/bin/python scripts/02_train.py --eval A B          # MAE fuera de muestra de los bundles A y B
Opciones: --tag (variantes de ablación), --rounds (pruebas rápidas).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dd.cv import fold_dir, masked_full  # noqa: E402
from dd.data import load_full, load_sim  # noqa: E402
from dd.paths import REPORTS, SUBMISSION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", nargs="*", default=[])
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", default=None, help="directorio de salida de --fold full (por defecto submission/)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--lgb", nargs="*", default=[], help="parámetros LightGBM comunes, p. ej. max_depth=8")
    args = ap.parse_args()
    full, sim = load_full(), load_sim()

    for fold in args.fold:
        from dd import train as T
        from dd.train import train_bundle

        for kv in args.lgb:
            k, v = kv.split("=", 1)
            T.LGB_BASE[k] = json.loads(v)

        out = (Path(args.out).resolve() if args.out else SUBMISSION) if fold == "full" else fold_dir(fold, args.tag)
        print(f"== entrenando pliegue {fold} → {out}", flush=True)
        train_bundle(masked_full(full, sim, fold), out, fold, args.rounds,
                     log=lambda m: print(m, flush=True))

    if args.eval:
        from dd.forecast_eval import evaluate

        res = {}
        for fold in args.eval:
            for mode in ("hard", "std"):
                r = evaluate(fold, args.tag, mode)
                res[f"{fold}_{mode}"] = r
                print(f"== pliegue {fold}, {mode}: {r['rows']} filas")
                for b in ("h0", "h1-2", "h3", "h3-7", "h8-23"):
                    e = r[b]
                    line = (f"  {b:6s} neta {e['e_net']:7.1f} kW (ref {e['e_net_base']:6.1f})  consumo "
                            f"{e['e_load']:6.1f}  solar {e['e_pv']:6.1f}")
                    if b not in ("h0", "h1-2") and mode == "hard":
                        line += (f" | precio {1e3 * e['e_price']:5.2f}e-3 (ref {1e3 * e['e_price_base']:5.2f})"
                                 f"  CO2 {1e3 * e['e_co2']:5.2f}e-3 (ref {1e3 * e['e_co2_base']:5.2f})")
                    print(line)
                print("  cobertura de los 9 cuantiles (nominal 0,056 … 0,944):",
                      [round(c, 3) for c in r["coverage"]])
                print(f"  horas de noche exactas: {r['exact_rows']} (MAE {r['exact_mae']})")
        out = REPORTS / "cv"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"forecast_{args.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
