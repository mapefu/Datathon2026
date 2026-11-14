"""Fase 4 · Figuras y tablas para el jurado (reports/figures/*.png, reports/RESULTADOS.md).

    .venv/bin/python scripts/05_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dd.cv import blocks_of, cv_weeks, fold_dir  # noqa: E402
from dd.data import RAW, fit_pv_irr, load_full, load_sim  # noqa: E402
from dd.env import run_p2, sim_frame  # noqa: E402
from dd.paths import ETA, MODES, REPORTS  # noqa: E402
from dd.score import aggregate, contributions, lp_plan, paired  # noqa: E402

FIG = REPORTS / "figures"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categóricos 1–3 (paleta de referencia, orden fijo)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print("  ", name)


def f1_frozen_meter():
    c = pd.read_csv(RAW / "contador.csv", usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"],
                    parse_dates=["Fecha"]).set_index("Fecha").iloc[:, 0]
    x = (-c).clip(0, 5000)["2025-10-07 12:00":"2025-10-10 12:00"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for d in pd.date_range("2025-10-07", "2025-10-09"):
        ax.axvspan(d + pd.Timedelta(hours=22, minutes=5), d + pd.Timedelta(hours=30), color=GRID, alpha=0.6, lw=0)
    ax.plot(x.index, x.values, color=S1, lw=1.5)
    ax.set_title("El contador se congela cada noche de 22:05 a 06:00")
    ax.set_ylabel("Consumo del campus (kW, cada 5 min)")
    ax.text(pd.Timestamp("2025-10-08 02:00"), x.min() - 60, "valor congelado", color=INK2, ha="center", fontsize=9)
    save(fig, "F1_contador_congelado.png")


def f2_persistence(full):
    h = full["load"]
    d = (h - h.shift(1)).abs().groupby(h.index.hour).mean()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(d.index, d.values, color=S1, width=0.7)
    ax.set_title("Error de repetir el consumo de la hora anterior, por hora del día")
    ax.set_xlabel("Hora")
    ax.set_ylabel("MAE (kW)")
    ax.set_xticks(range(24))
    ax.text(2.5, 8, "0 kW: noche congelada", ha="center", color=INK2, fontsize=9)
    save(fig, "F2_persistencia_por_hora.png")


def f3_pv_irr(full):
    d = full[["pv", "irr", "tpan"]].dropna()
    d = d[d.irr > 20]
    coef = fit_pv_irr(full)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(d.irr, 12 * d.pv, s=4, color=S1, alpha=0.25, lw=0)
    g = np.linspace(0, d.irr.max(), 100)
    ax.plot(g, 12 * (coef[0] * g + coef[2] * g ** 2), color=S2, label="ajuste físico (R² 0,99)")
    ax.axhline(12 * 150, color=INK2, lw=1, ls="--")
    ax.text(30, 12 * 150 + 30, "recorte de los 3 inversores (3 × 50 kW × 12)", color=INK2, fontsize=9)
    ax.set_title("La solar es casi una función de la irradiancia")
    ax.set_xlabel("Irradiancia (W/m², media horaria)")
    ax.set_ylabel("Solar Modo Difícil (kW, ×12)")
    ax.legend(loc="lower right")
    save(fig, "F3_solar_vs_irradiancia.png")


def f4_market(full):
    wk = full.index.day_of_week < 5
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, var, title, unit in ((axes[0], "price", "Precio PVPC", "€/kWh"),
                                 (axes[1], "co2", "Intensidad de CO2 de la red", "kg/kWh")):
        for s, (lab, col) in ((wk, ("laborable", S1)), (~wk, ("fin de semana", S2))):
            p = full[var][s].groupby(full.index[s].hour).mean()
            ax.plot(p.index, p.values, color=col, label=lab, marker="o", ms=3)
        if var == "price":
            for a, b in ((10, 14), (18, 22)):
                ax.axvspan(a - 0.5, b - 0.5, color=GRID, alpha=0.6, lw=0)
            ax.text(11.5, ax.get_ylim()[1] * 0.97, "P1 punta", ha="center", color=INK2, fontsize=9, va="top")
        ax.set_title(title)
        ax.set_xlabel("Hora")
        ax.set_ylabel(unit)
        ax.set_xticks(range(0, 24, 3))
        ax.legend()
    save(fig, "F4_perfiles_mercado.png")


def f5_net(sim):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.arange(-1200, 2600, 50)
    for mode, col in (("std", S1), ("hard", S2)):
        net = sim.load_kw - MODES[mode]["S"] * sim.pv_kw
        lab = f"{'Estándar' if mode == 'std' else 'Difícil'}: excedente {100 * (net <= 0).mean():.1f} % de las horas"
        ax.hist(net, bins=bins, histtype="step", lw=2, color=col, label=lab)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_title("Carga neta horaria (consumo − solar)")
    ax.set_xlabel("kW")
    ax.set_ylabel("Horas")
    ax.legend(loc="upper left")
    save(fig, "F5_carga_neta.png")


def f6_daily(full, sim):
    dl = full["load"].resample("1D").mean()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for m in pd.period_range("2025-09", "2026-08", freq="M"):
        last = m.to_timestamp(how="end").normalize()
        ax.axvspan(last - pd.Timedelta(days=6), last + pd.Timedelta(days=1), color=GRID, alpha=0.7, lw=0)
    ax.plot(dl.index, dl.values, color=S1, lw=1.5)
    ax.set_title("Consumo medio diario: fines de semana, cierres y semanas de test (gris)")
    ax.set_ylabel("kW")
    save(fig, "F6_consumo_diario.png")


def f7_week(sim):
    wk = cv_weeks(sim)
    weeks = wk["A"]
    res = run_p2(ROOT / "submission" / "agent.py", weeks, "hard",
                 {"BUNDLE_DIR": str(fold_dir("A")), "BLOCKS": blocks_of(weeks),
                  "TIME_SOFT_S": 1e12, "TIME_MID_S": 1e12, "TIME_HARD_S": 1e12}, keep=True)
    tr = res["_trace"]
    df = sim_frame(pd.concat(weeks), "hard")
    base = np.maximum(df.load_kw - df.pv_kw, 0)
    w = 70 * df.co2_kg_per_kwh / float(base @ df.co2_kg_per_kwh) + 30 * df.precio_eur_per_kwh / float(base @ df.precio_eur_per_kwh)
    a_or, _ = lp_plan((df.load_kw - df.pv_kw).to_numpy(), w.to_numpy(), 6000, 1500, 3000)
    soc_or = np.cumsum(np.where(a_or > 0, a_or * ETA, a_or / ETA)) + 3000
    starts = np.cumsum([0] + [len(g) for g in weeks])
    k = next(i for i, g in enumerate(weeks) if g.index[0].month == 4)    # semana de abril: hay excedente
    sel = slice(starts[k], starts[k + 1])
    idx = df.index[sel]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(idx, 1e3 * w.values[sel], color=S1)
    axes[0].set_title("Peso de cada hora: w = 70·CO2/CO2_base + 30·precio/Coste_base")
    axes[0].set_ylabel("puntos por MWh")
    axes[1].plot(idx, (df.load_kw - df.pv_kw).values[sel], color=INK2, lw=1.5, label="carga neta real")
    axes[1].plot(idx, tr["grid"][sel], color=S2, label="importación con el agente")
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].set_title("Carga neta e importación de red")
    axes[1].set_ylabel("kW")
    axes[1].legend(loc="upper right", ncols=2)
    axes[2].plot(idx, tr["soc"][sel], color=S2, label="agente")
    axes[2].plot(idx, soc_or[sel], color=S1, lw=1.5, ls="--", label="oráculo (información perfecta)")
    axes[2].set_title("Estado de carga")
    axes[2].set_ylabel("kWh")
    axes[2].legend(loc="upper right", ncols=2)
    save(fig, "F7_semana_operacion.png")


def f8_f9_forecast():
    fe = json.loads((REPORTS / "cv" / "forecast_default.json").read_text())
    buckets = ["h0", "h1-2", "h3", "h3-7", "h8-23"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    specs = ((axes[0], "e_net", "e_net_base", "Carga neta (kW)", buckets, 1),
             (axes[1], "e_price", "e_price_base", "Precio (×10⁻³ €/kWh)", buckets[2:], 1e3),
             (axes[2], "e_co2", "e_co2_base", "CO2 (×10⁻³ kg/kWh)", buckets[2:], 1e3))
    for ax, k, kb, title, bs, sc in specs:
        m = [sc * np.mean([fe[f"{f}_hard"][b][k] for f in "AB"]) for b in bs]
        r = [sc * np.mean([fe[f"{f}_hard"][b][kb] for f in "AB"]) for b in bs]
        x = np.arange(len(bs))
        ax.bar(x - 0.2, r, 0.38, color=S2, label="referencia ingenua")
        ax.bar(x + 0.2, m, 0.38, color=S1, label="modelo")
        ax.set_xticks(x, bs)
        ax.set_title(title)
        ax.set_ylabel("MAE fuera de muestra")
    axes[0].legend(loc="upper left")
    save(fig, "F8_error_prevision.png")

    q = (2 * np.arange(1, 10) - 1) / 18
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1)
    for f, col in (("A", S1), ("B", S2)):
        ax.plot(q, fe[f"{f}_hard"]["coverage"], color=col, marker="o", ms=6, label=f"pliegue {f}")
    ax.set_title("Calibración de los 9 escenarios")
    ax.set_xlabel("Cuantil nominal")
    ax.set_ylabel("Frecuencia observada")
    ax.legend(loc="upper left")
    save(fig, "F9_calibracion_escenarios.png")


def f10_weeks():
    v1 = json.loads((REPORTS / "cv" / "hard_v1.json").read_text())["P2"]
    orc = json.loads((REPORTS / "cv" / "hard_oracle.json").read_text())["P2"]
    keys = sorted(v1)
    ca, co = contributions({k: v1[k] for k in keys}), contributions({k: orc[k] for k in keys})
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(x - 0.2, [co[k] for k in keys], 0.38, color=S1, label="oráculo")
    ax.bar(x + 0.2, [ca[k] for k in keys], 0.38, color=S2, label="agente")
    ax.set_xticks(x, [k[2:] for k in keys], rotation=60, fontsize=8)
    ax.set_title("Aportación de cada semana al score (Difícil, P2, 22 semanas)")
    ax.set_ylabel("puntos de score")
    ax.legend(loc="upper right")
    save(fig, "F10_aportacion_semanal.png")


def tables():
    def agg(mode, name, p):
        f = REPORTS / "cv" / f"{mode}_{name}.json"
        return aggregate(json.loads(f.read_text())[p]) if f.exists() else None

    lines = ["# Resultados de la validación cruzada (22 semanas, modelos sin las semanas evaluadas)", "",
             "P1 = cada semana por separado con SOC al 50 %; P2 = semanas seguidas con SOC continuo (protocolo del "
             "FAQ y del servidor). Score oficial agregado con base común. 0 infracciones en todo.", "",
             "## T1 · Escalera de resultados", "",
             "| Modo | Agente | P1 | P2 | ΔCO2 P2 | ΔCoste P2 |", "| --- | --- | --- | --- | --- | --- |"]
    lines.append("| Difícil | Agente previo del equipo (README previo) | 21,73 | 19,89 | — | — |")
    for mode, rows in (("hard", (("v1", "**Este agente**"), ("oracle", "Oráculo"))),
                       ("std", (("prev", "Agente previo (agente_anterior)"), ("v1", "**Este agente**"), ("oracle", "Oráculo")))):
        for name, lab in rows:
            a1, a2 = agg(mode, name, "P1"), agg(mode, name, "P2")
            if a1:
                lines.append(f"| {'Difícil' if mode == 'hard' else 'Estándar'} | {lab} | {a1['score']:.2f} | "
                             f"{a2['score']:.2f} | {a2['dco2']:.2f} % | {a2['dcost']:.2f} % |")
    lines += ["", "## T2 · Ablaciones (Difícil, diferencia emparejada frente al agente final)", "",
              "| Cambio | Δ P1 | Δ P2 (± SE) | semanas ganadas P2 | Decisión |", "| --- | --- | --- | --- | --- |"]
    v1 = json.loads((REPORTS / "cv" / "hard_v1.json").read_text())
    for name, lab in (("a1_nonight", "Sin noche exacta (σ normal de noche)"),
                      ("a2_decision", "σ de decisión en vez de σ por horizonte"),
                      ("a3_nocross", "Sin planificar el arranque del bloque siguiente"),
                      ("a6_k5", "5 escenarios en vez de 9"),
                      ("r150", "150 árboles por modelo en vez de 300 (2× más rápido)")):
        f = REPORTS / "cv" / f"hard_{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        d1, d2 = paired(r["P1"], v1["P1"]), paired(r["P2"], v1["P2"])
        lines.append(f"| {lab} | {d1['agg']:+.3f} | {d2['agg']:+.3f} ± {d2['se_sum']:.3f} | "
                     f"{100 * d2['wins']:.0f} % | se queda el agente final |")
    fe = json.loads((REPORTS / "cv" / "forecast_default.json").read_text())
    lines += ["", "## T3 · Error de previsión fuera de muestra (Difícil, media de los pliegues A y B)", "",
              "| Horizonte | Carga neta (kW) | ref. | Precio (×10⁻³) | ref. | CO2 (×10⁻³) | ref. |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for b in ("h0", "h1-2", "h3", "h3-7", "h8-23"):
        g = lambda k, s=1: s * np.mean([fe[f"{f}_hard"][b][k] for f in "AB"])  # noqa: E731
        mk = (f"{g('e_price', 1e3):.2f} | {g('e_price_base', 1e3):.2f} | {g('e_co2', 1e3):.2f} | "
              f"{g('e_co2_base', 1e3):.2f}") if b not in ("h0", "h1-2") else "— | — | — | —"
        lines.append(f"| {b} | {g('e_net'):.1f} | {g('e_net_base'):.1f} | {mk} |")
    (REPORTS / "RESULTADOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("   RESULTADOS.md")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    full, sim = load_full(), load_sim()
    f1_frozen_meter()
    f2_persistence(full)
    f3_pv_irr(full)
    f4_market(full)
    f5_net(sim)
    f6_daily(full, sim)
    f7_week(sim)
    f8_f9_forecast()
    f10_weeks()
    tables()


if __name__ == "__main__":
    main()
