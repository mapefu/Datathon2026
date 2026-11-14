"""Empaqueta submission/ en submissions/submission_<fecha>.zip y lo valida.

Uso:
    python scripts/make_submission.py                       # Mode Estàndard
    python scripts/make_submission.py --dificil             # Mode Difícil (submission_dificil/)
"""

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission"
OUT_DIR = ROOT / "submissions"
MAX_MB = 25
SKIP = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dificil", action="store_true", help="empaqueta submission_dificil/")
    args = ap.parse_args()
    src = ROOT / "submission_dificil" if args.dificil else SRC
    prefix = "dificil" if args.dificil else "submission"
    files = [f for f in src.rglob("*") if f.is_file() and not SKIP & set(f.parts)]
    if not (src / "agent.py").exists():
        sys.exit(f"Falta {src.name}/agent.py")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(src))  # agent.py en la raíz del zip

    size_mb = out.stat().st_size / 1e6
    names = [str(f.relative_to(src)) for f in files]
    print(f"Creado {out.relative_to(ROOT)} ({size_mb:.2f} MB) con: {names}", flush=True)
    if size_mb > MAX_MB:
        sys.exit(f"ERROR: supera el límite de {MAX_MB} MB")

    # Validador oficial (escáner AST + 24 pasos de prueba)
    r = subprocess.run([sys.executable, "check_submission.py", str(out)], cwd=ROOT / "starter_kit")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
