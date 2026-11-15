"""Fase 4 · Empaqueta submission.zip (plano) y verifica todo lo exigible antes de subirlo.

    .venv/bin/python scripts/04_package_verify.py

1. Zip plano con agent.py + bundle.npz + params.json (< 20 MB).
2. Validador oficial `starter_kit/check_submission.py` sobre el zip.
3. Escáner AST propio: solo imports permitidos y nada de las listas prohibidas del validador.
4. Desde el zip descomprimido en otro directorio: simulación continua de ≈2.016 h en Difícil y Estándar
   (tiempo total con __init__, ms por paso, 0 infracciones).
5. Estrés: obs sucias, sin SOC, sin cap, sin meteo, listas de 24 en Difícil, obs malformada, fechas fuera
   de calendario, sin bundle. 0 infracciones y ninguna excepción.
6. Fuga a nivel de agente: alterar el futuro del simulador no cambia las acciones pasadas.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dd.cv import blocks_of, cv_weeks, fold_dir  # noqa: E402
from dd.data import load_sim  # noqa: E402
from dd.env import load_agent_module, sim_frame, week_id  # noqa: E402
from dd.score import aggregate, week_stats  # noqa: E402
from dd.stress import StressEnv  # noqa: E402

FILES = ("agent.py", "bundle.npz", "params.json")
# DD_SUB_DIR / DD_ZIP_NAME: empaquetar una variante (p. ej. submission_v2 → dist/submission_v2_<fecha>.zip)
SUB_DIR = ROOT / os.environ.get("DD_SUB_DIR", "submission")
ZIP_NAME = os.environ.get("DD_ZIP_NAME", "submission")
FOLD_TAG = os.environ.get("DD_FOLD_TAG", "default")
ALLOWED_IMPORTS = {"__future__", "datetime", "json", "math", "pathlib", "numpy", "scipy"}
FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  OK    " if cond else "  FALLA ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def build_zip() -> Path:
    out = ROOT / "dist" / f"{ZIP_NAME}_{datetime.now():%Y%m%d_%H%M}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in FILES:
            z.write(SUB_DIR / f, arcname=f)
    return out


def ast_scan(path: Path) -> list[str]:
    sys.path.insert(0, str(ROOT / "starter_kit"))
    import check_submission as cs  # type: ignore

    tree = ast.parse(path.read_text(encoding="utf-8"))
    errs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            roots = []
        errs += [f"import no permitido: {r}" for r in roots if r not in ALLOWED_IMPORTS]
        if isinstance(node, ast.Attribute) and node.attr in cs.FORBIDDEN_ATTRS:
            errs.append(f"atributo prohibido: {node.attr}")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in cs.FORBIDDEN_CALLS:
                errs.append(f"llamada prohibida: {f.attr}")
            if isinstance(f, ast.Name) and (f.id in cs.FORBIDDEN_BUILTINS or f.id == "open"):
                errs.append(f"builtin prohibido: {f.id}")
    official = cs.check_ast(path)
    if official:
        errs.append("escáner oficial: " + official)
    return errs


def simulate(mod_path: Path, weeks, mode: str, env_kw=None, overrides=None) -> dict:
    t0 = time.perf_counter()
    mod = load_agent_module(mod_path, overrides)
    agent = mod.BatteryAgent()
    df = sim_frame(pd.concat(weeks), mode)
    env = StressEnv(df, mode, **(env_kw or {}))
    obs, done, times, actions = env.reset(), False, [], []
    while not done:
        t1 = time.perf_counter()
        a = agent.act(obs)
        times.append(time.perf_counter() - t1)
        actions.append(a)
        obs, done = env.step(a)
    stats, i = {}, 0
    for g in weeks:
        n = len(g)
        stats[week_id(g)] = week_stats(np.asarray(env.grid[i:i + n]), df.iloc[i:i + n],
                                       int(sum(env.infraccions_per_step[i:i + n])), 0.0)
        i += n
    return {"score": aggregate(stats)["score"], "infr": int(sum(env.infraccions_per_step)),
            "total_s": time.perf_counter() - t0, "ms": 1000 * np.asarray(times), "actions": np.asarray(actions),
            "types_ok": all(isinstance(a, float) and np.isfinite(a) for a in actions)}


def main() -> None:
    sim = load_sim()
    wk = cv_weeks(sim)
    weeks12 = sorted(wk["A"] + wk["B"][-1:], key=lambda g: g.index[0])     # ≈ 2.016 h, 12 bloques

    print("1. Paquete")
    z = build_zip()
    size = z.stat().st_size / 2 ** 20
    names = zipfile.ZipFile(z).namelist()
    check(sorted(names) == sorted(FILES), f"zip plano con {names}")
    check(size < 20, f"tamaño {size:.2f} MB (< 20 MB)")

    print("2. Validador oficial")
    r = subprocess.run([sys.executable, str(ROOT / "starter_kit" / "check_submission.py"), str(z)],
                       capture_output=True, text=True, cwd=str(ROOT / "dist"))
    tail = [ln for ln in r.stdout.splitlines() if "Temps mitjà" in ln or "RESULTAT" in ln or "ERROR" in ln]
    check(r.returncode == 0 and "VÀLIDA" in r.stdout, "check_submission.py: " + " | ".join(s.strip() for s in tail))

    print("3. Escáner AST")
    errs = ast_scan(SUB_DIR / "agent.py")
    check(not errs, "imports y llamadas permitidas" + (f": {errs}" if errs else ""))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zipfile.ZipFile(z).extractall(tmp / "pkg")
        agent_zip = tmp / "pkg" / "agent.py"
        bare = tmp / "bare"
        bare.mkdir()
        shutil.copy(agent_zip, bare / "agent.py")

        print("4. Tiempo y física en simulación continua (desde el zip; modelos finales, el score aquí es "
              "optimista porque esas semanas están en train)")
        for mode, limit in (("hard", 30.0), ("std", 20.0)):
            res = simulate(agent_zip, weeks12, mode, overrides={"BLOCKS": blocks_of(weeks12)})
            ms = res["ms"]
            check(res["infr"] == 0 and res["types_ok"] and res["total_s"] < limit,
                  f"{mode}: {len(ms)} pasos en {res['total_s']:.1f} s (límite propio {limit:.0f} s; servidor 60 s), "
                  f"media {ms.mean():.1f} ms, p95 {np.percentile(ms, 95):.1f} ms, {res['infr']} infr., "
                  f"score {res['score']:.2f}")

        print("5. Estrés (Difícil, 3 semanas, modelos del pliegue A)")
        w3 = wk["A"][5:8]
        ov = {"BUNDLE_DIR": str(fold_dir("A", FOLD_TAG)), "BLOCKS": blocks_of(w3),
              "TIME_SOFT_S": 1e12, "TIME_MID_S": 1e12, "TIME_HARD_S": 1e12}
        base = simulate(agent_zip, w3, "hard", overrides=ov)["score"]
        print(f"        referencia limpia: {base:.2f}")
        cases = {
            "NaN 5 % + timestamp texto": dict(nan_prob=0.05, str_timestamp=True),
            "NaN 20 %": dict(nan_prob=0.20),
            "listas como np.array": dict(as_array=True),
            "sin soc_kwh": dict(drop_soc=True),
            "sin cap_max_kwh / power_max_kw": dict(drop_spec=True),
            "sin meteo": dict(drop_meteo=True),
            "listas de 24 h en Difícil": dict(full_lists=True),
            "obs malformada en el paso 50": dict(bad_obs_at=50),
            "fechas fuera de calendario (+3 años)": dict(shift_years=3),
        }
        for name, kw in cases.items():
            try:
                res = simulate(agent_zip, w3, "hard", kw, ov)
                check(res["infr"] == 0 and res["types_ok"], f"{name}: score {res['score']:.2f}, {res['infr']} infr.")
            except Exception as e:  # noqa: BLE001
                check(False, f"{name}: excepción {type(e).__name__}: {e}")
        for mode in ("hard", "std"):
            try:
                res = simulate(bare / "agent.py", w3, mode, overrides={"BLOCKS": blocks_of(w3)})
                check(res["infr"] == 0 and res["types_ok"],
                      f"{mode} sin bundle (regla de reserva): score {res['score']:.2f}, {res['infr']} infr.")
            except Exception as e:  # noqa: BLE001
                check(False, f"{mode} sin bundle: excepción {e}")

        print("6. Fuga a nivel de agente")
        g = wk["A"][6]
        ovl = {"BUNDLE_DIR": str(fold_dir("A", FOLD_TAG)), "BLOCKS": blocks_of([g]),
               "TIME_SOFT_S": 1e12, "TIME_MID_S": 1e12, "TIME_HARD_S": 1e12}
        ref = simulate(agent_zip, [g], "hard", overrides=ovl)["actions"]
        ok = True
        for s in (30, 90, 150):
            g2 = g.copy()
            rng = np.random.default_rng(s)
            cols = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]
            g2.iloc[s + 3:, [g2.columns.get_loc(c) for c in cols]] *= rng.uniform(0.5, 1.5, (len(g2) - s - 3, 4))
            act = simulate(agent_zip, [g2], "hard", overrides=ovl)["actions"]
            same = np.array_equal(ref[:s + 1], act[:s + 1])
            ok &= same
            print(f"        alterando desde la fila {s + 3}: acciones 0..{s} idénticas = {same}")
        check(ok, "el agente no ve el futuro")

    print(f"\nZIP: {z}")
    print("RESULTADO:", "OK" if not FAILED else f"{len(FAILED)} fallos: {FAILED}")
    (ROOT / "reports" / "verify.json").write_text(json.dumps({"zip": str(z), "failed": FAILED}, indent=1))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
