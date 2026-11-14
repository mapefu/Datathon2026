"""Fase 4 · Genera notebooks/analisis_jurado.ipynb (narrativa + figuras + tablas de reports/).

    .venv/bin/python scripts/06_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def img(name: str, caption: str):
    return md(f"![{caption}](../reports/figures/{name})\n\n*{caption}*")


cells = [
    md("# Batería del Campus Nord: previsión + MPC con programación lineal\n\n"
       "Datathon UPC Sostenible 2026 · Modo Estándar y Modo Difícil. Un mismo `agent.py` detecta el modo en la "
       "primera llamada. Todo lo de este cuaderno se reproduce con los scripts de `scripts/` (ver README)."),
    md("## 1. El score es lineal en la energía importada\n\n"
       "Con la base sin batería fija, minimizar Σ w_t·red_t con "
       "**w = CO2 + λ·precio** y **λ = (0,3/0,7)·CO2_base/Coste_base** (0,2617 en Difícil, 0,2517 en Estándar) "
       "es exactamente maximizar el Score oficial. No hay pesos ajustados a mano. El rendimiento del 90 % "
       "exige una diferencia del 11,1 % entre horas; la mediana del cociente diario máx/mín de w es 2,66, así "
       "que siempre hay un ciclo diario que aprovechar. La batería no puede vender: en el LP la importación es "
       "una variable ≥ 0, y verter no premia."),
    md("## 2. Hallazgos en los datos"),
    img("F1_contador_congelado.png", "El contador repite el mismo valor de 22:05 a 06:00 en 256 de 259 noches."),
    img("F2_persistencia_por_hora.png", "Consecuencia: de 23 h a 5 h la carga se conoce sin error; el agente lo "
        "explota (σ = 0) y gana +0,015 de forma consistente."),
    img("F3_solar_vs_irradiancia.png", "Solar ≈ 0,131·irradiancia (R² 0,99), con los 3 inversores recortando a 50 kW."),
    img("F4_perfiles_mercado.png", "PVPC con escalones de la tarifa 2.0TD (P1 +0,091 €/kWh, P2 +0,023) y CO2 "
        "mínimo a mediodía: la hora más limpia coincide con el excedente solar."),
    img("F5_carga_neta.png", "En Estándar casi nunca hay excedente (la previsión apenas importa); en Difícil hay "
        "excedente el 9,3 % de las horas y descargar de más se pierde."),
    img("F6_consumo_diario.png", "Fines de semana, festivos y cierres (Navidad, agosto) cambian el nivel de consumo; "
        "en gris, las semanas de test (sin datos)."),
    md("## 3. Previsión\n\n"
       "LightGBM directo multi-horizonte por bandas (consumo y solar h0 | h1–2 | h3–23; precio y CO2 h3–7 | "
       "h8–26), cada modelo predice la corrección sobre una base conocida. La solar se modela como índice de "
       "claridad sobre un cielo despejado empírico calculado en hora UTC. Los árboles se exportan a arrays y se "
       "evalúan con numpy (paridad exacta con LightGBM, sin pickles). Aumentos de datos para arranque en frío "
       "(noviembre no tiene sensores en train) y para el Modo Estándar sin meteo."),
    img("F8_error_prevision.png", "Error fuera de muestra (modelos sin las semanas evaluadas) frente a referencias "
        "ingenuas: la carga neta de la hora actual baja de 118 a 59 kW; el precio a +8–23 h, de 41 a 17 ×10⁻³ €/kWh."),
    img("F9_calibracion_escenarios.png", "Los 9 escenarios de carga neta (σ̂ condicional × cuantiles del error "
        "normalizado) están bien calibrados en las colas, que es donde se decide cuánto descargar."),
    md("## 4. Control\n\n"
       "Cada hora: LP de 24 h con `scipy.optimize.linprog` (HiGHS) y 9 escenarios equiprobables de carga neta con "
       "una sola acción por hora. En el óptimo, la descarga llega hasta un cuantil alto de la carga neta cuando la "
       "hora es cara y hasta uno bajo cuando es marginal. Valor terminal de la energía = cuantil 30 % de w. Al final "
       "de cada semana se planifica con la previsión del arranque de la siguiente (el SOC continúa, P2). Se ejecuta "
       "solo la primera hora y pasa por el Safety Clamp oficial."),
    img("F7_semana_operacion.png", "Una semana de abril: carga gratis con el excedente del mediodía, descarga en la "
        "punta de la tarde; el SOC del agente casi coincide con el del oráculo."),
    md("## 5. Validación\n\n"
       "22 semanas pseudo-test (las dos últimas disponibles de cada mes) retiradas de los datos como un hueco real; "
       "dos protocolos de SOC (P1 por semana, P2 continuo como el servidor); decisiones por diferencias emparejadas "
       "por semana (> 2 SE y ≥ 60 % de semanas). Prueba de fuga automática, 0 infracciones en todos los escenarios "
       "de estrés, 27 s para 2.009 h en Difícil (límite 60 s), paquete de 1,8 MB."),
    code("from pathlib import Path\nfrom IPython.display import Markdown\n"
         "Markdown(Path('../reports/RESULTADOS.md').read_text(encoding='utf-8'))"),
    img("F10_aportacion_semanal.png", "Aportación de cada semana al score frente al oráculo: el hueco restante es "
        "error de previsión, concentrado en días nublados y cambios de régimen."),
    md("## 6. Límites\n\n"
       "- No se puede verificar el protocolo exacto del servidor ni el reparto del 50 % privado; el agente detecta "
       "solo si el SOC se reinicia entre semanas y si las listas traen 3 o 24 horas.\n"
       "- El calendario de festivos de las semanas de test (24/09, Navidad, Semana Santa, 24/06) no se puede "
       "contrastar con datos.\n"
       "- Noviembre no tiene sensores en train: esa semana arranca en frío (modelos entrenados con ese caso).\n"
       "- No se usa ningún dato externo: precio, CO2 y meteo de las semanas de test solo llegan por `obs`."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "analisis_jurado.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(out)
