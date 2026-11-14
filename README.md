# Datathon 2026 — Repte UPC Sostenible: la batería del campus

Plataforma oficial: <https://repteupc.datathon.cat/>

## El reto en 30 segundos

Programar un `BatteryAgent` que cada hora decide la potencia de una batería de **2.000 kWh / 500 kW**
(η ida y vuelta 90 %, SOC inicial 50 %) conectada al campus de la UPC.

- `act(obs) -> float` en kW: `> 0` carga, `< 0` descarga.
- `obs` incluye la hora, el SOC, el precio y el CO₂ actuales, **los precios y el CO₂ de las próximas 24 h**,
  y el consumo y la solar de la hora anterior.
- **Score** = `0,70·ΔCO2% + 0,30·ΔCoste% − 1,00·infracciones`, comparado con el campus sin batería.
- **Test oculto**: los últimos 7 días de cada mes (2.016 h). Límites: 10 entregas al día, 60 s de ejecución y 25 MB.
- Las bibliotecas permitidas son numpy, pandas, scipy, sklearn, joblib, xgboost, lightgbm y la biblioteca estándar.
  No se pueden usar `os`, `sys`, red ni `eval`. Ver [docs/ENUNCIAT.md](docs/ENUNCIAT.md).

## Lo que ya sabemos

| | Score en validación (`--split val`) |
|---|---|
| Plantilla oficial (`starter_kit/agent.py`) | **0,00**: 9 infracciones por un bug de redondeo de η |
| Plantilla corregida (`submission/agent.py`) | 1,38 |
| Oráculo con información perfecta (LP) | **6,12**, el techo práctico |

- **La carga neta del campus casi siempre es > 500 kW** (en 5.962 de 5.993 h), así que la batería nunca
  "sobra" y el score es casi **lineal** en la potencia: descargar 1 kWh en la hora *t* vale
  `w_t = 70·co2_t/CO2_base + 30·precio_t/Coste_base`. Predecir el consumo o la solar aporta poco. Lo que cuenta
  es decidir cuándo cargar y cuándo descargar con las 24 h de precio y CO₂ que ya vienen en `obs`.
- El CO₂ es mínimo a mediodía (≈0,05) y alto de noche (≈0,10). La heurística oficial carga de noche, que es justo
  la peor hora para el 70 % del score.
- La penalización real es de 1 punto por infracción, mientras que `environment.py` aplica 0,2. Hay que tener
  **0 infracciones** siempre. `score_oficial` usa 1,0.
- Noviembre de 2025 no está en los datos de train.

## Estructura

```
.
├── submission/          # ← LO QUE SE ENTREGA: agent.py + .pkl/.json auxiliares
├── src/datathon/
│   ├── data.py          # CSV de 5 min → tabla horaria (caché en data/processed/hourly.csv) + split train/val
│   └── evaluate.py      # simulador oficial, score oficial, desglose por mes y oráculo LP
├── scripts/
│   ├── evaluate.py      # CLI para comparar agentes
│   └── make_submission.py  # submission/ → submissions/submission_<fecha>.zip + validador oficial
├── notebooks/01_exploracion.ipynb
├── starter_kit/         # kit oficial (sin modificar): environment.py, check_submission.py, tutorial
├── docs/                # enunciado oficial y guía de variables
└── data/raw/train/      # CSV oficiales (no versionados)
```

## Puesta en marcha

```bash
git clone https://github.com/mapefu/Datathon2026.git && cd Datathon2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Copiar los CSV de data/train del paquete oficial a data/raw/train/
```

## Flujo de trabajo

```bash
# 1. Editar submission/agent.py
# 2. Evaluar en local (validación = últimos 7 días disponibles de cada mes, imita el test)
python scripts/evaluate.py --oracle
python scripts/evaluate.py submission/agent.py starter_kit/agent.py   # comparar agentes
python scripts/evaluate.py --split all                                # todo el año
# 3. Empaquetar y validar
python scripts/make_submission.py
# 4. Subir submissions/submission_<fecha>.zip en https://repteupc.datathon.cat/
```

Si el agente carga ficheros, hay que usar `Path(__file__).parent / "modelo.pkl"` y nunca rutas absolutas.

## Equipo

- TODO
