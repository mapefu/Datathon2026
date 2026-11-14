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

| | Validación | Año completo |
|---|---|---|
| Plantilla oficial (`starter_kit/agent.py`) | 0,00 (9 infracciones por un bug de redondeo de η) | — |
| Plantilla con η corregida | 1,38 | 1,27 |
| **Nuestro agente MPC (`submission/agent.py`)** | **6,10** | **5,67** |
| Oráculo con información perfecta (LP), el techo | 6,12 | 5,67 |

### El agente: control predictivo (MPC) con programación lineal

Cada hora, `act(obs)`:
1. Calcula el peso de cada hora de la ventana conocida: `w_k = co2_k + λ·precio_k`, con
   `λ = (0,30/0,70)·(CO2_base/Coste_base) ≈ 0,253`. El cociente 0,59 kg/€ sale de train y es estable (0,57 en val).
2. Resuelve un LP de 24 h con la dinámica exacta del simulador: maximiza `Σ w_k·(d_k − c_k)`.
3. Ejecuta solo la primera hora (horizonte deslizante) y la recorta a los límites físicos con un margen
   de 1e-6 kW, así que nunca hay infracciones.

Detalles:
- Si la ventana trae menos de 24 h, la evaluación termina dentro del horizonte y la batería se vacía.
- La descarga de la hora actual se limita a la última carga neta observada. Así no se tira energía
  durante cortes de datos ni los domingos a mediodía.
- Controla el tiempo: si el LP acumula 40 s, pasa a una regla barata por ranking, que en validación saca 4,78.
  En condiciones normales tarda ≈6 s en 2.016 h.

Iguala al oráculo en train (5,55 = 5,55) y en el año completo. Si cada semana se evalúa por separado, saca
6,47 frente a 6,49. Con la información de `obs` no queda margen apreciable: el resto de la diferencia se debe
a que el oráculo conoce el futuro más allá de 24 h.

- **La carga neta del campus casi siempre es > 500 kW** (en 5.962 de 5.993 h), así que la batería nunca
  "sobra" y el score es casi **lineal** en la potencia: descargar 1 kWh en la hora *t* vale
  `w_t = 70·co2_t/CO2_base + 30·precio_t/Coste_base`. Predecir el consumo o la solar aporta poco. Lo que cuenta
  es decidir cuándo cargar y cuándo descargar con las 24 h de precio y CO₂ que ya vienen en `obs`.
- El CO₂ es mínimo a mediodía (≈0,05) y alto de noche (≈0,10). La heurística oficial carga de noche, que es justo
  la peor hora para el 70 % del score.
- La penalización real es de 1 punto por infracción, mientras que `environment.py` aplica 0,2. Hay que tener
  **0 infracciones** siempre. `score_oficial` usa 1,0.
- Noviembre de 2025 no está en los datos de train.

### Robustez frente al servidor (`python scripts/stress_test.py`)

No sabemos si el servidor da previsiones exactas, qué cociente CO₂/coste tendrá el test ni si llegarán
datos sucios. El agente se adapta midiendo lo que ve y no cambia nada si los datos son exactos:
- **λ online**: el cociente CO₂/coste se actualiza con las horas ya vividas (prior de train con peso de una semana).
- **Previsiones con error**: compara lo que se previó con lo que luego pasa y, para cada horizonte, usa la
  previsión más reciente o la combinación de todas las previsiones de esa hora, según cuál haya acertado más.
- **Datos sucios**: NaN, claves que faltan o cualquier excepción terminan en la regla de reserva o en 0 kW,
  siempre con recorte físico.

| Escenario | Versión anterior | Actual | Óptimo |
|---|---|---|---|
| Validación exacta | 6,09 | **6,10** | 6,12 |
| Año completo / semanas separadas | 5,67 / 6,47 | 5,67 / 6,47 | 5,67 / — |
| Ruido 10 % / 20 % / 30 % (independiente) | 6,07 / 6,02 / 5,93 | **6,09 / 6,08 / 6,05** | — |
| Ruido 20 % / 30 % (correlado entre previsiones) | 6,02 / 5,95 | 6,02 / 5,94 | — |
| NaN 5 % + `timestamp` como string | **FALLA** (excepción) | 6,08 | — |
| Tiempo por 2.016 h | 5,1 s | 5,8 s | límite 60 s |

En validación exacta, todo el hueco con el óptimo se debe al corte de sensores del 17–18 de junio, que es
imposible de prever. Sin esas horas, agente = óptimo = 6,12.

## Estructura

```
.
├── submission/          # ← LO QUE SE ENTREGA: agent.py + .pkl/.json auxiliares
├── src/datathon/
│   ├── data.py          # CSV de 5 min → tabla horaria (caché en data/processed/hourly.csv) + split train/val
│   ├── evaluate.py      # simulador oficial, score oficial, desglose por mes y oráculo LP
│   └── stress.py        # escenarios de estrés: ruido, NaN, semanas separadas, tiempo
├── scripts/
│   ├── evaluate.py      # CLI para comparar agentes
│   ├── stress_test.py   # CLI con la tabla de robustez
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
