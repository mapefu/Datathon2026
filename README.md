# Datathon 2026 — Repte UPC Sostenible: la batería del campus

Plataforma oficial: <https://repteupc.datathon.cat/>

## El reto en 30 segundos

Programar un `BatteryAgent` que cada hora decide la potencia de una batería de **2.000 kWh / 500 kW**
(η ida y vuelta 90 %, SOC inicial 50 %) conectada al campus de la UPC.

- `act(obs) -> float` en kW: `> 0` carga, `< 0` descarga.
- `obs` incluye la hora, el SOC, el precio y el CO₂ actuales, **los precios y el CO₂ de las próximas 24 h**,
  y el consumo y la solar de la hora anterior.
- **Score** = `0,70·ΔCO2% + 0,30·ΔCoste% − penalización·infracciones`, comparado con el campus sin batería. La
  fórmula de la web dice 0,20 por infracción y el FAQ 1,00; aquí se usa 1,00 (lo prudente) y siempre 0 infracciones.
- **Test oculto**: los últimos 7 días de cada mes (2.016 h; el ranking público usa el 50 %, 1.008 h). Límites: 10 entregas
  por franja (sábado hasta las 21:00, noche, domingo; 30 en total por modo), 60 s de ejecución y 25 MB.
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
- Controla el tiempo: si pasan 40 s de tiempo real desde la primera llamada, pasa a una regla barata por ranking,
  que en validación saca 4,78.
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
- La penalización por infracción es de 0,2 puntos según la fórmula de la web y `environment.py`, pero el FAQ dice
  1 punto. Hay que tener **0 infracciones** siempre. `score_oficial` usa 1,0 por prudencia.
- Noviembre de 2025 no está en los datos de train.

### Robustez frente al servidor (`python scripts/stress_test.py`)

No sabemos si el servidor da previsiones exactas, qué cociente CO₂/coste tendrá el test ni si llegarán
datos sucios. El agente se adapta midiendo lo que ve y no cambia nada si los datos son exactos:
- **λ online**: el cociente CO₂/coste se actualiza con las horas ya vividas (prior de train con peso de una semana).
- **Previsiones con error**: compara lo que se previó con lo que luego pasa y, para cada horizonte, usa la
  previsión más reciente o la combinación de todas las previsiones de esa hora, según cuál haya acertado más.
- **Datos sucios**: NaN, claves que faltan, tipos inesperados (arrays, Series) o cualquier excepción terminan
  en la regla de reserva o en 0 kW, siempre con recorte físico, y el paso siguiente vuelve a funcionar con
  normalidad. Un NaN en una previsión se rellena con la previsión anterior para esa misma hora.
- **SOC ausente**: se usa `soc_pct`, `soc_percent` o un modelo interno de la física; si no hay ninguno, 0 kW.

| Escenario | v1 (sin robustez) | v2 | **Actual** | Óptimo |
|---|---|---|---|---|
| Validación exacta | 6,09 | 6,10 | **6,10** | 6,12 |
| Año completo / semanas separadas | 5,67 / 6,47 | 5,67 / 6,47 | 5,67 / 6,47 | 5,67 / — |
| Ruido 10 % / 20 % / 30 % (independiente) | 6,07 / 6,02 / 5,93 | 6,09 / 6,08 / 6,05 | **6,09 / 6,08 / 6,05** | — |
| Ruido 20 % / 30 % (correlado entre previsiones) | 6,02 / 5,95 | 6,02 / 5,94 | 6,02 / 5,94 | — |
| NaN 5 % + `timestamp` como string | **FALLA** (excepción) | 6,08 | **6,10** | — |
| NaN 20 % | — | 5,97 | **6,10** | — |
| Previsiones como `np.array` | — | **0,00** (excepción en cada hora) | **6,10** | — |
| Una sola observación malformada (hora 50) | — | **0,29** (se queda en 0 kW para siempre) | **6,10** | — |
| Sin `soc_kwh` (quedan `soc_pct` y `soc_percent`) | — | **0,00** (896 infracciones) | **6,10** | — |
| Tiempo por 2.016 h | 5,1 s | 5,8 s | 5,9 s | límite 60 s |

En validación exacta, todo el hueco con el óptimo se debe al corte de sensores del 17–18 de junio, que es
imposible de prever. Sin esas horas, agente = óptimo = 6,12.

### ¿Mejoraría un modelo de predicción? (`python scripts/headroom.py`)

Para acotar lo que puede aportar cualquier modelo predictivo, le damos al mismo MPC el futuro **real**
(predicción perfecta) y medimos cuánto sube el score. Cada semana se evalúa por separado:

| Información extra (perfecta) | Validación (11 semanas) | Todo el año (184 ventanas de 7 días) |
|---|---|---|
| λ real de la semana (cociente CO₂/coste) | +0,003 | +0,004 |
| … + precio y CO₂ a 48 h o 72 h | +0,003 (= nada más) | +0,004 (= nada más) |
| … + carga neta real de las próximas 24 h | +0,026 | +0,018 |
| Oráculo de la semana entera | +0,027 | +0,018 |

- **Predecir precio/CO₂ más allá de 24 h no aporta nada.** La batería hace un ciclo diario (≈4 h de carga y
  ≈4 h de descarga) y las 24 h de `obs` ya contienen ese ciclo.
- **Predecir consumo o solar solo aporta en los cortes de sensores.** Fuera de la semana del 17 de junio la
  ganancia es 0: la carga neta no baja de ≈500 kW ni en Navidad ni en agosto. El corte empieza tras un hueco de
  datos y termina sin aviso: en la primera hora de vuelta, el agente aún ve 0 kW de la hora anterior y no
  descarga. Esa hora explica ≈0,20 de los 0,25 perdidos en junio. Las reglas de detección probadas (sin tope,
  tope binario) empatan o pierden hasta 0,5.
- **λ es una apuesta sobre cómo puntúa el servidor.** Un λ por semana gana +0,003 si se promedian puntuaciones
  semanales, pero pierde entre −0,003 y −0,006 si se puntúa el agregado de todas las semanas (lo que sugieren
  los kg/€ del leaderboard). El λ online actual queda a ≤0,003 del mejor en los dos casos.

Conclusión: con la información de `obs`, el agente está en el óptimo alcanzable. El hueco restante son
cortes de sensores imprevisibles.

## Mode Difícil (`submission_dificil/`)

Batería de 6.000 kWh / ±1.500 kW, solar ×12 (máximo en train 1.729 kW; la web dice ~1.650) y precio/CO₂ conocidos solo 3 h
(la actual y las 2 siguientes). Ahora sí hay horas con excedente solar (≈9 %) y descargar más
que la carga neta vierte energía a 0 € / 0 kg, así que **predecir importa**.

**Cómo se evalúa (FAQ oficial).** El evaluador crea UNA instancia del agente y la llama hora a hora con las 12
semanas seguidas: el SOC empieza al 50 % solo en la primera hora y **no se reinicia entre semanas**, y
`obs["timestamp"]` salta de un mes al siguiente. Hasta ahora la evaluación local simulaba cada semana por separado
con el SOC al 50 % al empezar cada una. Eso regala 3.000 kWh por semana e infla el score unos 2 puntos. Es la
explicación más probable de que la primera entrega sacara 19,88 en la web frente a 22,06 en local: el agente de las
18:23 (22,13 por semanas) da 19,86 en simulación continua (no se ha medido la versión exacta de la primera entrega).
`python scripts/hard.py eval --continuo` reproduce el modo del FAQ.

| Validación (11 semanas, score agregado) | Semanas separadas (SOC 50 % cada una) | **Continua (FAQ)** |
|---|---|---|
| Agente que vacía la batería al final de cada semana (versión anterior) | 22,13 | 19,86 |
| **Agente actual: conserva la energía entre semanas** | 21,32 | **20,31** |
| Mismo agente, SOC reiniciado entre semanas (lo detecta y vacía) | 22,04 | — |
| Oráculo con información perfecta | 23,70 | 22,45 |

Historial de mejoras (semanas separadas): perfiles estacionales 19,86 → + LightGBM 20,48 → + escenarios de carga
neta 22,06 → + escenarios escalados por el error previsto 22,13.

Cómo funciona:
1. **Previsiones a 24 h** con 4 modelos LightGBM (horizonte como variable). Las variables se
   calculan con el mismo código (`History.features` del agente) en entrenamiento e inferencia,
   reproduciendo semanas de 168 h con memoria vacía. Consumo, precio y CO₂ predicen el cambio
   respecto al último valor conocido. La solar usa geometría solar, cielo despejado (Haurwitz) e
   índice de claridad de la hora anterior. Sin día del año ni mes: con un solo año, los árboles
   memorizaban fechas (MAE de consumo 23 kW en train frente a 65 en validación).
2. **LP con escenarios**: la acción de cada hora es única, pero la importación se evalúa con
   10 escenarios equiprobables de carga neta: previsión + σ̂·z, donde σ̂ es el error esperado
   (un 5.º modelo: cielo variable frente a despejado, horizonte…) y z son los cuantiles del error
   normalizado fuera de muestra. Conocer la carga neta de la hora actual valía +1,75 de los +2,0
   que da conocerla entera; los escenarios recuperan +1,0 y el escalado por σ̂ +0,12 más.
3. **Sin LightGBM en el servidor**: los árboles se exportan a arrays y el agente los evalúa con
   numpy (los 5 bosques en un solo recorrido, ≈4 ms por hora). El entrenamiento comprueba que la
   predicción coincide con LightGBM.
4. Cada semana de test acaba a las 23:00 del último día del mes y el plan no pasa de ahí. Como el SOC
   continúa en la semana siguiente, la energía sobrante **no** se vacía (valor terminal normal), salvo en
   la última semana del periodo (agosto de 2026, `FINAL_MONTH`) o si el agente ve que el SOC vuelve al
   50 % tras un salto de semana. `KEEP_AT_WEEK_END = False` recupera el comportamiento anterior.
   La telemetría que llega tras un salto de `timestamp` se guarda en su hora real (la anterior al salto).
5. Safety Clamp con `cap_max_kwh` / `power_max_kw` de `obs` (o `soc_kwh / soc_pct`), presupuesto
   de tiempo acumulado (40 s → menos escenarios, 52 s → sin escenarios). ≈12 ms por hora.

Qué falta para el oráculo (agente con el futuro real sustituido): precio/CO₂ perfectos +0,9 (casi
todo en las horas 3–7), carga neta perfecta +0,9.

```bash
python scripts/hard.py train          # modelos con train sin validación (para evaluar)
python scripts/hard.py eval --oracle  # validación, semanas separadas
python scripts/hard.py eval --continuo --oracle  # validación como dice el FAQ (una simulación seguida)
python scripts/stress_hard.py         # robustez: sin meteo, NaN, obs malformada, etc. (0 infracciones)
python scripts/hard.py train --all    # modelos finales con todos los datos (¡la validación deja de ser limpia!)
python scripts/make_submission.py --dificil
```

## Estructura

```
.
├── submission/          # ← LO QUE SE ENTREGA: agent.py + .pkl/.json auxiliares
├── src/datathon/
│   ├── data.py          # CSV de 5 min → tabla horaria (caché en data/processed/hourly.csv) + split train/val
│   ├── evaluate.py      # simulador oficial, score oficial, desglose por mes y oráculo LP
│   ├── stress.py        # escenarios de estrés: ruido, NaN, semanas separadas, tiempo
│   └── headroom.py      # cota de lo que aportaría una predicción perfecta (agentes "tramposos")
├── scripts/
│   ├── evaluate.py      # CLI para comparar agentes
│   ├── stress_test.py   # CLI con la tabla de robustez
│   ├── headroom.py      # CLI: ¿ayudaría un modelo de predicción mejor?
│   └── make_submission.py  # submission/ → submissions/submission_<fecha>.zip + validador oficial
├── notebooks/01_exploracion.ipynb
├── starter_kit/         # kit oficial (sin modificar): environment.py, check_submission.py, tutorial
├── docs/                # enunciado oficial, guía de variables y RETO_Y_MODO_DIFICIL.md (resumen de la web)
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
