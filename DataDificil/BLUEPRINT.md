# BLUEPRINT — Datathon UPC Sostenible 2026 (Modo Estándar + Modo Difícil)

Especificación técnica para el modelo ejecutor (Opus 5.5). Redactada el 14/11/2026 tras inspeccionar los datos
reales y el trabajo previo del equipo. Todo lo marcado **[MEDIDO]** se ha comprobado sobre `data/train/`; lo
marcado **[PREVIO]** viene de los repos anteriores del equipo; lo marcado **[SUPUESTO]** no se ha podido verificar.

## 0. Punto de partida: qué se sabe ya y qué se espera

### 0.1 Recursos en esta máquina

| Recurso | Ruta |
| --- | --- |
| CSV oficiales | `/home/marc/Projects/Datathon2026/data/raw/train/` (6 ficheros) |
| Kit oficial (sin tocar) | `/home/marc/Projects/Datathon2026/starter_kit/` (`environment.py`, `check_submission.py`, `battery_spec.json`) |
| Python | `/home/marc/Projects/Datathon2026/.venv/bin/python` (3.14; numpy 2.5, pandas 3.1, scipy 1.18, sklearn 1.9, lightgbm 4.7, matplotlib, nbformat). **No hay pytest ni pyarrow; prohibido `pip install`.** |
| Agente previo del Modo Difícil (21,6 en la web) | `/home/marc/Projects/Datathon2026-hardcv/submission_dificil/agent.py` + `src/datathon/{hard,hard_cv,hard_train}.py` |
| Agente previo del Modo Estándar (6,10 en validación) | `/home/marc/Projects/Datathon2026-hardcv/submission/agent.py` |

Máquina: 8 CPU, **6 GB de RAM compartidos con otros procesos**. LightGBM con `num_threads=2`, un solo
entrenamiento a la vez, y `inversor.csv` (65 MB, 176 columnas) siempre con `usecols`.

**Se parte del agente previo, no de cero.** Sus componentes ya validados en el servidor real se portan tal cual
(evaluador de árboles en numpy, LP con escenarios, política de fin de semana, tolerancia a `obs` sucio). Este
documento es autosuficiente, pero donde diga «portar» significa copiar y adaptar ese código.

### 0.2 Hechos medidos que condicionan el diseño

1. **[MEDIDO] El contador se congela cada noche de 22:05 a 06:00.** En 256 de 259 noches la lectura de potencia
   es idéntica durante esas 8 horas. En la tabla horaria, la carga de las horas 0–5 es exactamente igual a la de
   la hora anterior (MAE 0,0 kW) y la de la hora 23 difiere 0,5 kW de media. Es un artefacto del sensor, pero es
   la verdad con la que se puntúa. Consecuencia: 7 horas al día la carga neta se conoce sin error.
2. **[MEDIDO] La persistencia de la carga falla donde más valor hay:** MAE por hora de `load[t] − load[t−1]`:
   150 kW a las 6 h (descongelación), 70–88 kW de 7 a 10 h, 30–46 kW de 11 a 19 h, 74–83 kW a las 20–21 h,
   33 kW a las 22 h.
3. **[MEDIDO] Solar:** 3 inversores que recortan a 50 kW cada uno (máximo horario 144,1 kW; ×12 = 1.729 kW).
   `pv ≈ 0,1311·irr − 2,30e-4·irr·(tpan−25) + 1,38e-5·irr²` con R² = 0,990 y MAE 2,7 kW (32 kW a escala ×12).
   La irradiancia de la hora anterior no añade casi nada a la solar de la hora anterior. `VELOCIDAD VIENTO`
   vale 0 siempre: se descarta.
4. **[MEDIDO] Las marcas de tiempo son hora local con cambio de hora** (el centroide solar pasa de 12,8 h en
   invierno a 13,4 h en verano). Los dos cambios de hora caen dentro de semanas de test (26/10/2025 y 29/03/2026).
5. **[MEDIDO] El PVPC sigue la tarifa 2.0TD.** En laborables hay escalones de ≈0,065 €/kWh a las 10, 14, 18 y
   22 h; en fines de semana no. Los festivos nacionales de fecha fija (8/12, 6/1, 1/5) no tienen escalón; los
   autonómicos y el Viernes Santo sí lo tienen. El mínimo diario de `w` cae a las 14 h en 118 días.
6. **[MEDIDO] Cobertura por fichero.** `precio_pvpc.csv` y `co2_red.csv` cubren **noviembre (1–23)** y
   **agosto 1–24**. `contador.csv` y `meteo.csv` cubren agosto 1–24 pero no noviembre. `inversor.csv` acaba el
   16/08 a las 22:00. El `hourly.csv` previo (unión interna) tiraba todo eso: el agente previo arranca en frío en
   noviembre y en agosto. El 24/08 el campus ya está reabierto (media 1.223 kW frente a ≈880 kW del cierre).
7. **[MEDIDO] Totales del baseline sin batería en las 5.993 h con las 4 variables:**

   | Modo | CO2 base (kg) | Coste base (€) | λ = (0,3/0,7)·CO2/Coste |
   | --- | --- | --- | --- |
   | Estándar (pv×1) | 551.823 | 939.634 | **0,2517** |
   | Difícil (pv×12) | 468.035 | 766.495 | **0,2617** |

   En Difícil hay excedente en el 9,3 % de las horas y la carga neta es < 1.500 kW en el 98,4 %: la descarga
   casi siempre está limitada por la carga neta, no por el inversor. El cociente máx/mín diario de `w` es ≥ 1,5
   en el 95 % de los días (mediana 2,66): el ciclo diario siempre compensa.
8. **[PREVIO] Modo Estándar resuelto:** LP de 24 h con los precios de `obs` da 6,10 frente a 6,12 del oráculo.
   La carga neta es > 500 kW en el 99,5 % de las horas, así que predecir consumo o solar no aporta.
9. **[PREVIO] Modo Difícil, validación cruzada de 22 semanas:** agente previo 19,89 (SOC continuo, P2) y 21,73
   (SOC reiniciado, P1); oráculo 21,45 / 22,84. Con el futuro real el mismo controlador iguala al oráculo, así
   que todo el hueco es de previsión: carga neta de la hora actual +0,65, carga neta a 24 h +0,83,
   precio/CO2 +0,77. Rechazados con medida: MPC de dos etapas (−0,14), trayectorias conjuntas (−0,10),
   más rondas de boosting, margen fijo sobre la carga neta.
10. **[PREVIO] El servidor parece usar SOC continuo** (19,88 en la web frente a 19,86 en simulación continua
    local). `scipy.optimize.linprog(method="highs")` y la evaluación de árboles en numpy funcionan en el servidor.

### 0.3 Expectativa realista

El techo local con información perfecta está ≈1,5 puntos por encima del agente previo. Este diseño ataca la
parte recuperable: se espera **+0,2 a +0,6** sobre el agente previo en CV-P2. No hay ningún camino honesto
conocido hasta los ≈24 de los líderes con el simulador local; esa diferencia puede venir del reparto
público/privado, del protocolo del servidor o de datos externos. **Línea roja:** no se incrustan precios, CO2 ni
meteorología reales de las fechas de test obtenidos fuera de `data/train/`; es fuga de datos y el jurado audita
el código.

---

## 1. Arquitectura general y flujo de datos

### 1.1 Estructura de desarrollo

```
DataDificil/
├── BLUEPRINT.md
├── data/raw/train -> /home/marc/Projects/Datathon2026/data/raw/train   (enlace simbólico)
├── data/processed/            hourly_full.csv, hourly_sim.csv, folds/<fold>/
├── starter_kit/               copia intacta del kit oficial
├── submission/
│   ├── agent.py               ÚNICA fuente de verdad del runtime (memoria, features, bosque, LP, agente)
│   ├── bundle.npz             generado: historia de train, calendario, bosques, tablas de error
│   └── params.json            generado: constantes, metadatos de modelos, bloques de test
├── src/dd/
│   ├── data.py                CSV → tablas horarias
│   ├── calendar_upc.py        festivos, cierres, periodos tarifarios, cielo despejado empírico
│   ├── env.py                 StdEnv / HardEnv fieles al oficial + ejecutores P1 y P2
│   ├── score.py               score oficial agregado, contribución por semana, oráculo LP
│   ├── cv.py                  pliegues, comparación emparejada
│   ├── dataset.py             filas de entrenamiento llamando a agent.build_features
│   ├── train.py               entrena, calcula residuos OOF, exporta bundle + params
│   ├── export.py              LightGBM → arrays planos + prueba de paridad
│   └── stress.py              obs sucio, tiempo, tamaño
├── scripts/                   00_build_data, 01_oracle, 02_train, 03_cv, 04_ablate, 05_stress,
│                              06_make_submission, 07_figures
├── tests/                     scripts con assert ejecutables con `python tests/x.py` (no hay pytest)
├── notebooks/analisis_jurado.ipynb
├── reports/                   cv/*.json, figures/*.png, tablas .md
└── dist/                      submission_std_<fecha>.zip, submission_hard_<fecha>.zip
```

`src/dd/dataset.py` y `train.py` **importan `submission/agent.py`** para construir las variables: el mismo código
en entrenamiento e inferencia, sin duplicados.

### 1.2 Estructura plana de `submission.zip`

```
agent.py        (≈60 KB)
bundle.npz      (≤ 8 MB, np.savez_compressed, solo arrays numéricos, se carga con allow_pickle=False)
params.json     (≤ 50 KB)
```

Los dos modos usan **el mismo zip**; el agente detecta el modo en la primera llamada. Se sube una copia a cada
ranking. Carga: `Path(__file__).parent / "bundle.npz"`. Sin pickles ni joblib: no hay dependencia de versiones.

### 1.3 Flujo de datos

```
OFFLINE  6 CSV ─► data.py ─► hourly_full (8.808 h, NaN por variable) ─► calendar_upc ─► dataset.py
                                   │                                                        │
                                   └─► hourly_sim (5.993 h) ─► env/score/cv ◄── agent.py ◄──┤
                                                                                train.py ───┘─► bundle.npz + params.json

ONLINE   obs ─► detectar modo (1.ª llamada) ─► Memory.observe ─► build_features (24 filas)
             ─► Forest.predict_all ─► previsión {load, pv, price, co2} + σ ─► escenarios N[H,K]
             ─► LP (linprog/HiGHS) ─► c₀ − d₀ ─► Safety Clamp ─► float
```

---

## 2. Ingeniería de datos, limpieza y validación local

### 2.1 Tablas horarias (`src/dd/data.py`)

Índice común: todas las horas de `2025-09-01 00:00` a `2026-09-01 23:00` (8.808 filas), hora local sin zona.
`idx(ts) = int((ts − BASE) / 1h)` con `BASE = 2025-09-01 00:00`. Remuestreo idéntico al oficial:
`.resample("1h").mean()` (etiqueta y cierre por la izquierda).

| Columna | Fórmula exacta |
| --- | --- |
| `load_kw` | `(-contador["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(0, 5000)` → media horaria |
| `pv_kw` (×1) | suma de `Inversor {1,2,3} - POTENCIA ACTIVA (kW)` → `.clip(lower=0)` → media horaria. El filtro oficial también coge las 3 columnas `SETPOINT RELATIVO POTENCIA ACTIVA (%)`, que valen 0 siempre: mismo resultado |
| `irr`, `tamb`, `tpan` | `Atersa - IRRADIANCIA (W/m2)`, `TEMPERATURA AMBIENTE (SENSOR 1)`, `TEMPERATURA PANEL (SENSOR 1)` → media horaria |
| `price`, `co2` | `precio_eur_per_kwh`, `co2_kg_per_kwh` |

- `hourly_full`: unión **externa**, NaN por variable. Es la memoria que viaja en el bundle.
- `hourly_sim`: filas con `load_kw, pv_kw, price, co2` todas finitas (como hace el organizador con `.dropna()`).
  **Criterio de aceptación: 5.993 filas, de 2025-09-01 00:00 a 2026-08-16 22:00.**
- `pv_mem`: copia de `pv_kw` para la memoria en la que los huecos con `irr` finita (16/08 23:00 – 24/08 23:00) se
  rellenan con el ajuste del punto 0.2-3 (reajustar los 3 coeficientes por mínimos cuadrados con `irr > 20`),
  recortado a `[0, 150]`.
- No se imputa nada más. Las horas con carga 0,0 exacta son cortes del sensor (17–18/06, 06/08) y se dejan: son
  la verdad con la que se puntúa.

### 2.2 Calendario (`src/dd/calendar_upc.py`), precalculado para los 8.808 índices y guardado en el bundle

| Array | Definición |
| --- | --- |
| `cal_hour`, `cal_dow` | hora local y día de la semana |
| `cal_daytype` | 0 laborable, 1 sábado, 2 domingo o festivo de campus |
| `cal_act` | nivel de actividad: 1,0 normal; 0,6 reducido; 0,3 cierre |
| `cal_period` | periodo 2.0TD: 3 de 0 a 8 h y todo el día en sábado, domingo y festivo nacional fijo; 1 de 10–14 y 18–22 h; 2 el resto |
| `cal_utc_off` | 2 entre el último domingo de marzo 02:00 y el último de octubre 03:00; 1 fuera |
| `cal_cs` | solar de cielo despejado empírica (kW, escala ×1), ver 3.3 |
| `cal_night` | 1 si la hora ∈ {23, 0, 1, 2, 3, 4, 5} (contador congelado y sin sol) |

- Festivos nacionales fijos (tarifa): 12/10, 01/11, 06/12, 08/12, 25/12 de 2025; 01/01, 06/01, 01/05, 15/08 de 2026.
- Festivos de campus (`daytype = 2`): los anteriores más 11/09/2025, 24/09/2025, 26/12/2025, 03/04/2026,
  06/04/2026, 25/05/2026, 24/06/2026.
- Cierre (`act = 0,3`): 24/12/2025–06/01/2026 y 01/08–23/08/2026. Reducido (`act = 0,6`): 22–23/12/2025 y
  30/03–02/04/2026.
- Los días 24/09, 25–31/12, 30–31/03, 25/05 y 24/06 caen en semanas de test y no se pueden comprobar con datos
  **[SUPUESTO]**. Opus debe listar además los laborables de train cuya media diaria sea < 0,8 × la mediana del
  mismo día de la semana en ±3 semanas (por ejemplo 12/02/2026) y dejarlos documentados, sin añadirlos al calendario.

Bloques de test esperados (`params.json["blocks"]`): para cada mes de 2025-09 a 2026-08, de `00:00` del día
(último − 6) a `23:00` del último día. El arnés local sobrescribe esta lista con sus pseudo-bloques
(`agent_module.BLOCKS = [...]` antes de instanciar).

### 2.3 Simulador y score (`src/dd/env.py`, `score.py`)

`HardEnv(CampusBatteryEnv)` sobrescribe solo `_get_obs`:

- `h_end = min(t + K, n)` con `K = spec["forecast_horizon"]` (3 en Difícil, 24 en Estándar). El `environment.py`
  oficial ignora `forecast_horizon` y siempre da 24; el nuestro lo respeta.
- Añade `cap_max_kwh`, `power_max_kw` y, en Difícil, `irradiancia_wm2`, `temp_ambient_c`, `temp_panel_c`,
  `vent_ms` (= 0,0) dentro de `noves_dades_5min`, todos de la fila `t−1` (fila `t` si `t = 0`), igual que
  consumo y solar.
- Especificaciones: Estándar `{2000, 500, 0.90, 0.50, K=24}`, pv×1. Difícil `{6000, 1500, 0.90, 0.50, K=3}`, pv×12.

Protocolos:

- **P2 (principal):** una sola simulación con las semanas concatenadas en orden cronológico, una sola instancia
  del agente, SOC heredado. La telemetría del primer paso de cada semana es la de la última hora de la semana
  anterior, como en el servidor.
- **P1:** una simulación y un agente nuevos por semana, SOC al 50 %.

Score agregado con la fórmula oficial (`evaluate_results`, penalización 0,2 por infracción; el objetivo es 0).
Contribución de la semana `s`, que suma el score agregado y sirve para comparar por pares:

$$\text{contrib}_s = \frac{70}{C_{\text{base}}}\sum_{t\in s} co2_t\,(g^{\text{base}}_t - g_t) + \frac{30}{E_{\text{base}}}\sum_{t\in s} precio_t\,(g^{\text{base}}_t - g_t)$$

con $g^{\text{base}}_t = \max(D_t - S\cdot G_t, 0)$, $S \in \{1, 12\}$ y $C_{\text{base}}, E_{\text{base}}$ sobre todas las
semanas evaluadas.

**Oráculo** (`score.oracle`): el LP de la sección 4 con `K = 1`, valores reales, horizonte = toda la simulación y
`v_T = 0`. **Aceptación del arnés: reproduce 21,45 ± 0,10 (P2) y 22,84 ± 0,10 (P1) en las 22 semanas del Modo
Difícil.**

### 2.4 Validación cruzada sin fuga (`src/dd/cv.py`)

- Semanas pseudo-test: por cada mes con datos en `hourly_sim` (11 meses), pliegue **A** = últimos 7 días
  disponibles y pliegue **B** = los 7 anteriores; se descartan las de menos de 120 horas. Total ≈ 22 semanas,
  la misma definición que el CV previo, así que los números son comparables.
- Para evaluar el pliegue X: `hourly_full` con las semanas de X puestas a NaN **en todas las variables**, además
  de los huecos reales de test. Con eso se recalcula `cal_cs`, se entrenan los modelos y se rellena la memoria del
  agente. Las semanas quedan como un hueco real, igual que en el servidor.
- Regla de decisión fijada de antemano: un cambio se acepta si la diferencia emparejada por semana en P2 cumple
  `media > 2·SE` y gana en ≥ 60 % de las semanas. Si no, se queda el valor por defecto de este documento.
- Pliegues de robustez, que no deciden parámetros:
  - **FRÍO:** semanas A de octubre, enero y mayo ocultando carga, solar y meteo de los 30 días anteriores
    (mercado visible). Emula noviembre.
  - **SIN-PV:** semana A de julio con `pv` de los 8 días anteriores sustituida por la imputación desde `irr`.
    Emula agosto.
- Modelos finales: entrenados con todo `hourly_full`.

---

## 3. Modelos predictivos

### 3.1 Estrategia

- **Directa multi-horizonte por bandas, con `h` como variable; nunca recursiva.** Los errores no se encadenan y
  todas las bandas se evalúan en una sola pasada del bosque. Las bandas vienen del trabajo previo, donde dieron
  +0,24 (consumo y solar) y +0,28 (mercado).
- Objetivo de cada modelo: **diferencia respecto a una base conocida**, para que el árbol solo aprenda la
  corrección.
- Sin mes ni día del año como variables (con un solo año los árboles memorizan fechas **[PREVIO]**). La
  estacionalidad entra por los retardos, `cal_cs`, `cal_act` y `lvl_load`.
- LightGBM, exportado a arrays y evaluado con numpy en el agente.
- **Modo Estándar:** usa los mismos modelos de consumo y solar con la meteo en NaN. No usa modelos de mercado
  (vienen 24 valores en `obs`).

Notación: `t` = índice de la hora de decisión, `τ = t + h` = hora objetivo, `M[v][i]` = memoria de la variable
`v`. **Límites de causalidad:** sensores (`load`, `pv`, `irr`, `tamb`) solo con índice ≤ `t−1`; mercado solo con
índice ≤ `a = t + K_obs − 1` (en Difícil `a = t+2`). Toda lectura fuera de límite devuelve NaN.

### 3.2 Consumo — 3 modelos: `load@0` (h=0), `load@1` (h=1–2), `load@3` (h=3–23)

Base: `L1` = último valor finito de `M[load]` en `[t−3, t−1]`; si no hay, `lvl_load(t−1)`.
Objetivo: `load[τ] − base`.

| Variable | Definición |
| --- | --- |
| `h`, `hour_tau`, `hour_t`, `dow_tau`, `daytype_tau`, `daytype_t`, `act_tau` | calendario |
| `L1`, `L2`, `dL` | `load[t−1]`, `load[t−2]`, `L1 − L2` |
| `L24`, `L48`, `L168` | `load[τ−24]`, `load[τ−48]`, `load[τ−168]` |
| `L_same` | `load[τ−24j]` del `j ∈ 1..7` más pequeño con el mismo `daytype` que `τ` |
| `ramp_yd`, `ramp_wk` | `load[τ−24] − load[t−1−24]`, `load[τ−168] − load[t−1−168]` |
| `ramp_md` | media de `load[τ−24j] − load[t−1−24j]` en los `j ∈ 1..7` disponibles |
| `L_m24` | media de `load[t−24..t−1]` (mínimo 6 valores) |
| `F_prev` | carga de la hora más reciente ≤ `t−1` con `cal_night = 1` (nivel congelado de la última noche) |
| `night_tau`, `night_anchor` | `cal_night[τ]`, `cal_night[t−1]` |
| `tamb1`, `tamb_m24` | temperatura en `t−1` y media de 24 h (NaN en Estándar) |
| `lvl_load` | mediana de la carga por (`daytype`, hora) en los días de train a ±35 días de `τ`, excluyendo el propio día |

**Regla de la noche congelada, aplicada después del modelo (no es opcional):**

- Si `cal_night[t−1] = 1` o `hour_t = 23`, y `L1` es un dato observado: `load_hat[k] = L1` para todos los pasos
  consecutivos `k = 0, 1, …` con `cal_night[τ_k] = 1`. Esos pasos se marcan `exact[k] = True`.
- Para noches futuras dentro del horizonte (decisión antes de las 23 h): todas las horas con `cal_night = 1` de
  una misma noche toman el valor previsto para las 23 h de esa noche (perfil plano), sin marca de exactas.
- Corte de sensor: si `L1 < 1,0` kW, todos los escenarios del paso 0 valen 0 (el corte persiste).

### 3.3 Solar — 3 modelos: `pv@0`, `pv@1`, `pv@3`, en escala ×1

**Cielo despejado empírico `cal_cs[i]`**, calculado en hora UTC entera (`u = hora_local − cal_utc_off`) para
absorber el cambio de hora:

1. Para cada `u`, serie diaria `x_d = pv(d, u)` con los datos de train del pliegue.
2. Envolvente: máximo móvil centrado de ±10 días → mediana móvil de 15 días → media móvil de 15 días.
3. Interpolación lineal en los días sin datos (semanas de test, noviembre, final de agosto), tratando el año como
   circular (agosto de 2026 enlaza con septiembre de 2025).

Objetivo: índice de claridad `k[τ] = pv[τ] / cs[τ]`, solo filas con `cs[τ] > 5` kW, **ponderadas por `cs[τ]`**.
Predicción: `pv_hat = clip(k_hat, 0, 1.25)·cs[τ]`. Si `cs[τ] ≤ 5`: `pv_hat = k_day·cs[τ]` (0,6 si `k_day` es NaN).
Si `cs[τ] = 0`: `pv_hat = 0` exacto. En el agente la telemetría solar se divide por `S` al guardarla y la
previsión se multiplica por `S` al salir (`S = 12` si la capacidad es ≥ 4.000 kWh, 1 si no).

| Variable | Definición |
| --- | --- |
| `h`, `hour_utc_tau`, `cs_tau`, `cs_rel` | `cs_rel = cs[τ] / max(cs del día de τ)` |
| `k1`, `k2`, `dk` | `pv[t−1]/cs[t−1]` y `pv[t−2]/cs[t−2]` si `cs > 5`, si no NaN; `k1 − k2` |
| `k_day` | `Σpv / Σcs` en las horas de hoy con `cs > 5` hasta `t−1` (mínimo 2 horas) |
| `k_yday`, `k_24` | lo mismo para el día anterior completo y para las últimas 24 h |
| `k24` | `pv[τ−24] / cs[τ−24]` |
| `cs1` | `cs[t−1]` |

### 3.4 Precio y CO2 (solo Modo Difícil) — 4 modelos: `price@3`, `price@8`, `co2@3`, `co2@8`

Bandas: `@3` = h ∈ 3–7, `@8` = h ∈ 8–26. Se llega a 26 para prever las primeras 24 h del bloque siguiente desde
tres horas antes (ver 5.4). Base: `x2` = último valor conocido (`M[x][a]`). Objetivo: `x[τ] − x2`.

| Variable (para `x` ∈ {price, co2}) | Definición |
| --- | --- |
| `h`, `hour_tau`, `hour_t`, `dow_tau`, `period_tau`, `period_a` | calendario y periodo tarifario del objetivo y del ancla |
| `x0`, `x1`, `x2`, `slope` | las 3 horas conocidas y `x2 − x0` |
| `x24`, `x48`, `x168` | `x[τ−24]`, `x[τ−48]`, `x[τ−168]` (límite `a`) |
| `lvl_d1` | media de `x[t+i] − x[t+i−24]` para `i = 0, 1, 2` |
| `shape_yd`, `shape_wk` | `x[τ−24] − x[a−24]`, `x[τ−168] − x[a−168]` |
| `shape_md` | media de `x[τ−24j] − x[a−24j]` en `j ∈ 1..7` |
| `xfer_yd` | `x2 + shape_yd` |
| `x_m24`, `x_min24`, `x_max24` | sobre `[t−21, t+2]` |
| `y0`, `y2`, `y_lvl_d1` | las mismas tres de la otra variable de mercado |
| `k_day`, `k_yday` | claridad solar local (hueco solar de mediodía) |
| solo precio: `x24_adj` | `x[τ−24] − toll[period(τ−24)] + toll[period(τ)]` |

`toll`: escalón tarifario por periodo y año natural, estimado por diferencias en diferencias con el pliegue de
train: media en laborables de `p[h] − p[h−1]` en las fronteras 10, 14, 18 y 22 h menos la misma media en fines de
semana. `toll[3] = 0`; `toll[1] − toll[2]` ≈ 0,065.

Los modelos de mercado se entrenan con las **6.768 horas** de precio y CO2 (incluye noviembre y agosto 17–24),
no con las 5.993 de `hourly_sim`.

### 3.5 Incertidumbre de la carga neta — modelo `sigma` + tablas `z`

1. Residuos fuera de muestra `r = net_real − net_hat` (`net = load − 12·pv`) con `GroupKFold(3)` por semana ISO
   sobre los modelos puntuales.
2. `sigma`: LightGBM de regresión sobre `|r|` con las variables `h, hour_tau, daytype_tau, cs_tau, k1, dk, k_day,
   L1, dL, night_tau`, `min_data_in_leaf = 200`, 150 rondas. Suelo de 5 kW.
3. `z_table[régimen][j]`: cuantiles `(2j−1)/18`, `j = 1..9`, de `r / sigma_hat` fuera de muestra, por régimen
   `día` (`cs[τ] > 0`) o `noche`.
4. `sigma_h0_table[hora][régimen]`: media de `sigma_hat` de las filas con `h = 0`, por hora del día objetivo.
   Es la incertidumbre que quedará cuando llegue esa hora.
5. Para el Modo Estándar se guarda una segunda `z_table` y `sigma_h0_table` con `net = load − pv`.

### 3.6 Hiperparámetros, tamaño y latencia

| Modelos | Filas aprox. | `num_leaves` | `min_data_in_leaf` | Rondas | `learning_rate` |
| --- | --- | --- | --- | --- | --- |
| `load@0`, `pv@0` | 6.000 | 15 | 40 | 300 | 0,05 |
| `load@1`, `pv@1` | 12.000 | 31 | 60 | 300 | 0,05 |
| `load@3`, `pv@3` | 125.000 | 31 | 100 | 300 | 0,05 |
| `price@3`, `co2@3` | 34.000 | 31 | 100 | 300 | 0,05 |
| `price@8`, `co2@8` | 128.000 | 31 | 100 | 300 | 0,05 |
| `sigma` | 144.000 | 31 | 200 | 150 | 0,05 |

Comunes: `objective="regression"`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`,
`lambda_l2=1.0`, `seed=7`, `deterministic=True`, `force_row_wise=True`, `num_threads=2`, `verbose=-1`. Sin parada
temprana (rondas fijas, reproducible). Solo variables numéricas.

**Aumentos de datos obligatorios** (filas duplicadas con las variables indicadas en NaN):

- Arranque en frío, 20 % de las filas de consumo y solar: se anulan los retardos de sensores con índice anterior
  a un inicio de bloque simulado, sorteado entre 0 y 167 h antes de `t`.
- Sin meteo, 30 % de las filas de consumo: `tamb1` y `tamb_m24` en NaN (Modo Estándar).

Presupuesto: 11 modelos ≈ 190.000 nodos × 30 bytes ≈ **6 MB sin comprimir, ≤ 4 MB en el npz**; historia y
calendario < 0,5 MB. Total < 8 MB (límite 25 MB). Latencia objetivo por paso en local: variables ≤ 2 ms, bosque
≤ 4 ms, LP ≤ 7 ms, resto ≤ 1 ms → **≤ 14 ms de media, ≤ 30 s por simulación de 2.016 h** (límite 60 s).

### 3.7 Exportación del bosque y evaluación en numpy (portar del agente previo)

- `export.py`: de `booster.dump_model()` se sacan, por árbol, arrays de nodos internos (`feat`, `thr` float64,
  `left`, `right`, `default_left`, `missing_type`) y de hojas (`value`). Se renumera con las hojas a continuación
  de los nodos internos y cada hoja apuntándose a sí misma (`thr = +inf`), para recorrer sin máscaras.
- Todos los modelos se funden en un bosque único con columnas globales, `tree_model[árbol]` y la banda
  `[h_lo, h_hi)` de cada árbol. Cada árbol solo se evalúa en las filas de su banda.
- Regla de valor ausente idéntica a LightGBM: `missing_type = NaN` y `x` NaN → rama por defecto;
  `missing_type = None` y `x` NaN → se trata como 0,0; `missing_type = Zero` → rama por defecto si `x` es NaN o
  `|x| ≤ 1e-35`.
- Recorrido vectorizado nivel a nivel sobre todos los pares (árbol, fila) activos; suma por modelo con
  `np.bincount`; se suma la base de cada modelo.
- **Prueba de paridad obligatoria:** `max |forest − booster.predict| < 1e-6` sobre 2.000 filas con NaN
  aleatorios, para cada modelo.

---

## 4. Controlador MPC: formulación exacta para `scipy.optimize.linprog`

### 4.1 Peso de cada hora

$$w_k = co2_k + \lambda \cdot precio_k,\qquad \lambda = \frac{0{,}30}{0{,}70}\cdot\frac{C_{\text{base}}}{E_{\text{base}}}$$

λ = 0,2517 (Estándar) y 0,2617 (Difícil), constantes. Minimizar $\sum_t w_t\, g_t$ equivale a maximizar el Score:
$\text{Score} = \frac{70}{C_{\text{base}}}\sum_t w_t (g^{\text{base}}_t - g_t)$. `w_k` usa el valor de `obs` si
`k < K_obs` y la previsión si no.

### 4.2 Variables y problema

Horizonte `H` pasos (24 por defecto), `K = 9` escenarios equiprobables por paso, $\eta = \sqrt{0{,}90}$.

$$x = [\,c_0..c_{H-1}\;|\;d_0..d_{H-1}\;|\;s_0..s_{H-1}\;|\;e_{0,1}..e_{0,K},\,\dots,\,e_{H-1,K}\,] \in \mathbb{R}^{3H + HK}$$

`c` carga, `d` descarga (kW), `s_k` SOC al final del paso `k` (kWh), `e_{k,j}` importación de red del paso `k` en
el escenario `j`.

$$\min\; \sum_{k=0}^{H-1}\sum_{j=1}^{K} \frac{w_k}{K}\, e_{k,j} \;+\; \varepsilon\sum_k (c_k + d_k) \;-\; v_T\, s_{H-1}$$

sujeto a

$$c_k - d_k - e_{k,j} \le -n_{k,j}\quad \forall k,j \qquad\text{(sin premio por verter: } e_{k,j} \ge n_{k,j} + c_k - d_k,\; e \ge 0)$$

$$s_k - s_{k-1} - \eta\, c_k + \tfrac{1}{\eta}\, d_k = 0\quad (s_{-1} = \text{SOC}_{\text{obs}})$$

### 4.3 Matrices (dispersas CSR, construidas una vez por par `(H, K)` y guardadas en caché)

Sea $R = I_H \otimes \mathbf{1}_K$ (de $HK \times H$: la fila $(k,j)$ tiene un 1 en la columna $k$) y
$D = I_H - L$, con $L$ la subdiagonal de unos.

$$A_{ub} = [\; R \;\; -R \;\; 0_{HK\times H} \;\; -I_{HK} \;],\qquad b_{ub} = -\text{vec}(N)\;\; (N_{k,j} = n_{k,j},\text{ por filas})$$

$$A_{eq} = [\; -\eta I_H \;\; \tfrac{1}{\eta} I_H \;\; D \;\; 0_{H\times HK} \;],\qquad b_{eq} = (\text{SOC}_{\text{obs}}, 0, \dots, 0)$$

$$c^{\top} = [\; \varepsilon\mathbf{1}_H \;\; \varepsilon\mathbf{1}_H \;\; (0,\dots,0,-v_T) \;\; \text{vec}(w_k/K) \;]$$

`bounds` (array `(n, 2)` preconstruido; solo se reescriben las dos primeras entradas en cada paso):

| Bloque | Límites |
| --- | --- |
| `c_0` | `[0, max_carga]` del Safety Clamp |
| `d_0` | `[0, max_descarga]` del Safety Clamp |
| `c_k`, `d_k`, k ≥ 1 | `[0, P_max]` |
| `s_k` | `[0, C_max]` |
| `e_{k,j}` | `[0, +inf)` |

Llamada: `linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
options={"presolve": False})`. En cada paso solo cambian `c`, `b_ub`, `b_eq[0]` y dos filas de `bounds`.
Acción: `p = x[0] − x[H]`. `ε = 1e-5` evita cargar y descargar a la vez y los ciclos sin beneficio.

**Latencia [MEDIDO] en esta máquina, con carga ajena:** H=24, K=9 → mediana 6,5 ms; H=24, K=1 → 3,3 ms; H=48,
K=9 → 12,5 ms. El objetivo de < 3 ms por LP no es alcanzable con `linprog` (solo la envoltura de scipy cuesta
≈ 2 ms); no hace falta, porque el presupuesto real es 29,8 ms por paso. Se fija **H = 24, K = 9**.

### 4.4 Escenarios de carga neta `N[k, j]`

$$n_{k,j} = \hat n_k + \sigma_k \cdot z_{r(k),\,j},\qquad \hat n_k = \widehat{load}_k - S\cdot\widehat{pv}_k$$

- `r(k)` = `día` si `cal_cs[τ_k] > 0`, `noche` si no. `z` son los 9 cuantiles de 3.5.
- **`exact[k]` → `σ_k = 0`** (noche congelada con dato observado).
- `k = 0`: `σ_0 = max(5, sigma_hat(h=0))`.
- `k ≥ 1`, parámetro `SPREAD_MODE`:
  - `"horizon"` (valor por defecto, validado en el trabajo previo): `σ_k = sigma_hat(h=k)`.
  - `"decision"`: `σ_k = sigma_h0_table[hora(τ_k)][r(k)]`. Razón: cuando llegue la hora `k` el agente volverá a
    decidir con el error de una hora, no con el de `k` horas; usar el error de `k` horas con una acción común
    infravalora la descarga futura. Se activa solo si pasa la regla de 2.4 (ablación A2).
- No hay márgenes fijos sobre la carga neta: el abanico de escenarios es el margen. En el óptimo la descarga de la
  hora actual cumple $P(n_0 > d_0) = \mu / (\eta\, w_0)$ y la carga desde excedente $P(n_0 > -c_0) = \eta\mu / w_0$,
  con $\mu$ el valor dual del SOC: en horas caras se descarga hasta un cuantil alto y en horas marginales, uno bajo.

### 4.5 Valor terminal y final de horizonte

| Situación | Horizonte | `v_T` |
| --- | --- | --- |
| Normal, Modo Difícil | 24 | $\eta \cdot Q_{0{,}30}(w_{0..H-1})$ |
| Normal, Modo Estándar | `K_obs` (24) | $Q_{0{,}40}(w_{0..H-1})$ |
| Quedan menos de 24 pasos en el bloque, SOC continuo y hay bloque siguiente | 24, cruzando al bloque siguiente (5.4) | como el normal |
| Fin de simulación o reinicio de SOC detectado | pasos que quedan | 0 (vaciar) |

Los dos cuantiles son mesetas ya medidas **[PREVIO]** (0,15–0,6 no cambia el score). No se reajustan.

### 4.6 Regla de reserva determinista (si `linprog` falla, devuelve estado ≠ 0 o se agota el tiempo)

Con `w` de las próximas 24 h (conocidas más previstas, o el perfil horario de `params.json` si no hay modelos):

```
lo, hi = quantile(w, 0.20), quantile(w, 0.70)
si w[0] <= lo:   p = +P_max
si w[0] >= hi:   p = -min(P_max, max(0, n_hat_0 - sigma_0))
si no:           p = 0
```

Siempre pasa por el Safety Clamp.

---

## 5. Contrato de `agent.py`

### 5.1 Reglas de implementación

- Imports permitidos: `json`, `math`, `datetime`, `pathlib.Path`, `numpy`, `scipy.sparse`,
  `scipy.optimize.linprog`. **Nada de `time`, `os`, `sys`, `base64`, `calendar`, `pickle`, `joblib`.** El
  cronómetro usa `datetime.datetime.now()`.
- Prohibido usar `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `__import__`, y los atributos
  `__globals__`, `__code__`, `__builtins__`, `__subclasses__`.
- `act` nunca lanza excepción ni devuelve NaN: cualquier fallo acaba en `0.0` recortado.
- `check_submission.py` llama 24 veces con `timestamp` como texto fijo, `soc_kwh = 1000`, sin `cap_max_kwh` y con
  listas de 24: tiene que funcionar (se detecta Modo Estándar; un `timestamp` que no avanza no escribe memoria).

### 5.2 Estado

```
self.mode            "std" | "hard"            (fijado en la 1.ª llamada)
self.cap, self.pmax, self.S, self.lam
self.M               dict var -> np.ndarray float64 [N_IDX]  (precargado con hourly_full del bundle)
self.cal             arrays de calendario del bundle
self.forest, self.meta, self.z_table, self.sigma_h0
self.prev_idx, self.prev_soc_pred, self.kmax_seen
self.p1_reset        False   (True si se detecta reinicio del SOC entre bloques)
self.nov_seen        False
self.next_cache      {}      (previsión del arranque del bloque siguiente, por bloque)
self.lp_cache        {}      ((H, K) -> A_ub, A_eq, bounds)
self.elapsed         0.0     (segundos acumulados dentro de act)
```

`N_IDX = 8808`. Si `idx` cae fuera de `[0, N_IDX)`, el agente trabaja en modo degradado (regla de 4.6 con lo que
traiga `obs`).

### 5.3 Pseudocódigo

```python
def act(self, obs):
    t0 = datetime.datetime.now()
    try:
        cap, pmax, soc = self._read_spec_and_soc(obs)      # ver 5.5
    except Exception:
        return 0.0
    if soc is None:
        return 0.0
    try:
        p = self._decide(obs, soc)
    except Exception:
        p = 0.0
    p = self._clamp(p, soc)                                # ver 5.6
    self.prev_soc_pred = soc + (p * ETA if p >= 0 else p / ETA)
    self.elapsed += (datetime.datetime.now() - t0).total_seconds()
    return float(p)

def _decide(self, obs, soc):
    idx = self._idx(obs["timestamp"])                      # pd.Timestamp, datetime o str
    vp, vc = vector(obs["preus_properes_24h"]), vector(obs["co2_properes_24h"])
    K_obs = min(len(vp), len(vc))
    if self.mode is None:
        self._detect_mode(obs, K_obs)                      # ver 5.5
    jump = self.prev_idx is not None and idx - self.prev_idx > 2
    new_block = self.prev_idx is None or idx - self.prev_idx > 72

    # 1) protocolo de SOC entre bloques
    if new_block and self.prev_idx is not None and self.prev_soc_pred is not None:
        if abs(soc - self.prev_soc_pred) > 1.0 and abs(soc - 0.5 * self.cap) < 1.0:
            self.p1_reset = True
    if month(idx) == 11:
        self.nov_seen = True

    # 2) memoria: O(1) por variable
    tel = obs.get("noves_dades_5min") or {}
    if self.prev_idx is not None and self.prev_idx < idx:
        k_tel = self.prev_idx                              # la telemetría es de la fila anterior de la
        M.load[k_tel] = num(tel.ultim_consum_kw)           # simulación; tras un salto NO se copia a idx-1
        M.pv[k_tel]   = num(tel.ultima_solar_kw) / self.S
        M.irr[k_tel], M.tamb[k_tel] = num(tel.irradiancia_wm2), num(tel.temp_ambient_c)
    for i in range(K_obs):                                 # mercado por PASO: dentro del bloque paso = hora
        if idx + i <= self._block_end(idx):
            M.price[idx + i], M.co2[idx + i] = vp[i], vc[i]
    self.prev_idx = idx
    self.kmax_seen = max(self.kmax_seen, K_obs)

    # 3) horizonte y final
    left = self._block_end(idx) - idx + 1                  # pasos que quedan en el bloque
    ending = (K_obs < self.kmax_seen)                      # la lista se acorta: la simulación termina
    last_block = self._is_last_block(idx)                  # ver 5.4
    if ending:
        H, vT_zero, cross = K_obs, True, False
    elif left < 24 and (self.p1_reset or last_block):
        H, vT_zero, cross = left, True, False
    else:
        H, vT_zero, cross = 24, False, left < 24

    # 4) previsión
    f = self._forecast(idx, K_obs, min(H, left))           # dict: load, pv, price, co2, sigma, exact
    if cross:
        f = concat(f, self._next_block_forecast(idx)[: H - left])     # ver 5.4
    w = f.co2 + self.lam * f.price
    for i in range(min(K_obs, H)):
        w[i] = vc[i] + self.lam * vp[i]                    # lo conocido manda, también al cruzar de bloque
    net = f.load - self.S * f.pv
    N = scenarios(net, f.sigma, f.exact, regimes, self.z_table)        # [H, K]
    if M.load[idx - 1] < 1.0:
        N[0, :] = 0.0                                      # corte de sensor

    # 5) gobernador de tiempo y LP
    if self.elapsed > 52:
        return self._rule(w, net, f.sigma)
    K = 9 if self.elapsed < 38 else (3 if self.elapsed < 46 else 1)
    vT = 0.0 if vT_zero else terminal_value(w, self.mode)
    try:
        return self._lp(w, N[:, pick(K)], soc, vT)
    except Exception:
        return self._rule(w, net, f.sigma)
```

`_forecast(idx, K_obs, H)`: `build_features` devuelve la matriz `(H, n_features)` con indexación vectorizada
sobre `self.M` (`tau = idx + arange(H)`, retardos con máscara de límite) → `forest.predict_all` → suma de bases →
regla de la noche congelada → rellenos (`price` 0,12; `co2` 0,08; `load` → `lvl_load`; `pv` → 0). Si
`K_obs ≥ H` no se evalúan los modelos de mercado.

### 5.4 Bloques, saltos y final de simulación

- **Salto > 2 h dentro de un bloque** (hueco de datos, cambio de hora): la telemetría se guarda en su hora real
  (`prev_idx`); las horas intermedias quedan en NaN y `L1` usa el último valor finito de `[t−3, t−1]`. No hay
  ventanas móviles que reiniciar: todo se calcula por índice absoluto.
- **Salto > 72 h = bloque nuevo.** Los retardos anteriores al bloque salen de la historia de train precargada
  (hasta la hora anterior al bloque en 10 bloques; en noviembre solo el mercado; en agosto carga, meteo y solar
  imputada hasta el 24/08). El SOC siempre es el de `obs["soc_kwh"]`.
- **Cruce de bloque** (`cross`): los pasos posteriores al final del bloque son las primeras horas del bloque
  siguiente de `BLOCKS`. Su previsión se calcula una vez por bloque y se guarda en `next_cache`: sensores con
  decisión en `t_b` (inicio del bloque siguiente; los retardos ≤ `t_b − 1` están en la historia precargada) y
  `h = 0..23`; mercado con decisión en `t_b − 3` (ancla `t_b − 1`) y `h = 3..26`. Es el mismo `_forecast`, con
  otro índice.
- **Último bloque** (`_is_last_block`): agosto de 2026 siempre. Julio de 2026 también si `nov_seen` es falso: si
  el servidor se ha saltado noviembre (sin sensores en train), lo más probable es que también falte agosto (sin
  inversores desde el 16/08) y la simulación acabe el 31/07 **[SUPUESTO]**.
- **Primer bloque:** se asume SOC continuo (FAQ y evidencia de la web); se cambia a vaciar por bloque solo si se
  detecta un reinicio.

### 5.5 Detección de modo (primera llamada, sin variables externas)

```
cap  = num(obs.get("cap_max_kwh"))
si no cap > 0:  cap = soc_kwh / soc_pct   si soc_pct > 0.01
si no cap > 0:  cap = 6000 si K_obs <= 3, si no 2000
pmax = num(obs.get("power_max_kw"));  si no pmax > 0: pmax = cap / 4
mode = "hard" si cap >= 4000, si no "std"
S    = 12 si mode == "hard", si no 1
lam  = 0.2617 si mode == "hard", si no 0.2517
```

`cap` y `pmax` se releen en cada llamada (si vienen en `obs`); `mode`, `S` y `lam` no cambian. El modo nunca se
deduce de la longitud de la lista después de la primera llamada, porque al final de la simulación se acorta.
SOC: `soc_kwh`; si falta, `soc_pct·cap`; si falta, `soc_percent/100·cap`; si falta, `prev_soc_pred`; si no hay
nada, 0 kW.

### 5.6 Safety Clamp (oficial, con los valores por defecto del modo detectado)

```python
ETA, EPS = 0.90 ** 0.5, 1e-4
max_carga    = max(0.0, min(pmax, (cap - soc) / ETA - EPS))
max_descarga = max(0.0, min(pmax, soc * ETA - EPS))
p = 0.0 if not math.isfinite(p) else max(-max_descarga, min(max_carga, p))
if abs(p) < 1e-3: p = 0.0
```

---

## 6. Plan de ejecución para Opus 5.5

Orden estricto. Cada fase termina cuando se cumplen sus criterios; si uno falla, se arregla antes de seguir.
Tras F3 siempre hay algo entregable.

| Fase | Trabajo | Criterios de aceptación |
| --- | --- | --- |
| **F0** Arranque | Enlace a los datos, copia de `starter_kit/`, `data.py`, `calendar_upc.py`, `00_build_data.py` | `hourly_sim` = 5.993 filas; totales de 0.2-7 con error < 0,1 %; MAE de `load[t] − load[t−1]` en horas 0–5 = 0,0 |
| **F1** Arnés | `env.py`, `score.py`, `cv.py`, `01_oracle.py`; `tests/env_parity.py` | Con K=24 y pv×1, `StdEnv` da las mismas `obs` y score que el oficial; oráculo Difícil 21,45 ± 0,10 (P2) y 22,84 ± 0,10 (P1); oráculo Estándar ≈ 6,12 en las 11 semanas A |
| **F2** Agente sin modelos | `agent.py` completo con previsiones de perfil (consumo: `lvl_load` + noche congelada + persistencia; solar: `k_day·cs`; mercado: `xfer_yd`), LP, clamp, bloques | 0 infracciones en P1 y P2; Estándar ≥ 6,09 en semanas A; Difícil P2 ≥ 17,5; `check_submission.py` pasa |
| **F3** Modelos | `dataset.py`, `train.py`, `export.py`, bundle; `tests/forest_parity.py`, `tests/leak.py` | Paridad < 1e-6; prueba de fuga con diferencia 0; MAE fuera de muestra de la carga neta en h0 ≤ 64 kW; MAE de precio en h3 ≤ 8,0e-3 €/kWh; **Difícil CV-P2 ≥ 19,9 y P1 ≥ 21,7** (paridad con el agente previo) |
| **F4** Ablaciones | `04_ablate.py`, tabla en `reports/ablaciones.md` | Cada fila con P1, P2, Δ emparejado, SE y % de semanas ganadas; decisión según la regla de 2.4 |
| **F5** Estrés y entrega | `05_stress.py`, `06_make_submission.py` | Ver lista de abajo |
| **F6** Jurado | `07_figures.py`, notebook | Las 10 figuras y las 3 tablas de la sección 7 |

**Prueba de fuga (`tests/leak.py`):** se ejecuta el agente dos veces sobre el mismo bloque; en la segunda, las
filas del simulador posteriores al paso `s` llevan ruido. Las acciones hasta `s` deben ser idénticas
(`s` = 30, 90, 150). Además `build_features` en modo depuración comprueba que ningún índice leído supera los límites.

**Ablaciones de F4, en este orden** (cada una sobre la mejor configuración aceptada hasta el momento):

| Id | Cambio | Ganancia esperada en P2 |
| --- | --- | --- |
| A1 | Noche congelada (`exact`, perfil plano) activada frente a desactivada | +0,02 a +0,08 |
| A2 | `SPREAD_MODE = "decision"` frente a `"horizon"` | −0,05 a +0,15 |
| A3 | Cruce de bloque frente a truncar con `v_T` de cuantil | +0,02 a +0,06 |
| A4 | `period_tau`, `period_a` y `x24_adj` frente a sin ellos | +0,03 a +0,10 |
| A5 | `cal_cs` empírico frente a Haurwitz (`1098·cosz·exp(−0,057/cosz)`) | +0,02 a +0,08 |
| A6 | K = 5, 9, 15 | meseta; elegir el menor sin pérdida |
| A7 | Pliegues FRÍO y SIN-PV: historia por variable frente a arranque en frío total | informativo: cuánto se pierde |
| A8 | Modo Estándar: agente unificado frente al agente previo del Estándar | debe ser ≥ −0,01 |

Si se acaba el tiempo, se entrega la última configuración aceptada. A1, A3 y A7 no dependen de reentrenar y van
primero si hay prisa.

**Lista de entrega (F5), todo obligatorio:**

1. `python starter_kit/check_submission.py dist/submission_*.zip` → válido, para los dos zips.
2. Escáner AST propio (`tests/ast_scan.py`): ningún import fuera de la lista de 5.1 y ninguna llamada o atributo
   de las listas `FORBIDDEN_*` de `check_submission.py`.
3. Zip plano: `agent.py`, `bundle.npz`, `params.json` en la raíz; tamaño < 25 MB.
4. Simulación continua de 2.016 pasos (12 pseudo-bloques) en un proceso nuevo: **≤ 30 s** en Difícil y ≤ 20 s en
   Estándar, contando el `__init__`.
5. 0 infracciones en todos los escenarios de estrés: limpio; NaN en el 5 % y el 20 % de los campos; `timestamp`
   como texto; listas como `np.array`; sin `soc_kwh`; sin `cap_max_kwh` ni `power_max_kw`; sin claves de meteo;
   una `obs` malformada en el paso 50; listas de 24 en Difícil; sin `bundle.npz` (regla de reserva); fecha fuera
   de calendario.
6. El zip se prueba descomprimido en un directorio temporal vacío, con otro directorio de trabajo.
7. Subida: primero «Validació Ràpida» de cada modo (gratis), después el envío con cuota.

Sondas opcionales del servidor (longitud real de las listas, reinicio de SOC, penalización): ya están escritas en
`/home/marc/Projects/Datathon2026-hardcv/experiments/fleet/lane5/sondas/`. El agente se adapta solo a las tres
cosas, así que no bloquean nada.

---

## 7. Guía de argumentación para el jurado

### 7.1 Narrativa (una frase por idea, en este orden)

1. **El score es lineal en la energía importada.** Con `w = CO2 + λ·precio` y `λ = (0,3/0,7)·CO2_base/Coste_base`,
   minimizar `Σ w·red` es exactamente maximizar el Score. No hay pesos ajustados a mano.
2. **La física decide la forma del problema.** El 90 % de rendimiento exige un 11,1 % de diferencia; el cociente
   máx/mín diario de `w` tiene mediana 2,66, así que siempre hay ciclo diario. La batería no puede vender: un LP
   con la importación como variable `≥ 0` lo modela de forma exacta.
3. **En Estándar no hace falta predecir; en Difícil es lo único que importa.** Estándar: carga neta > 500 kW el
   99,5 % del tiempo y 24 h de mercado conocidas → el LP iguala al oráculo. Difícil: 9,3 % de horas con excedente,
   descarga limitada por la carga neta el 98,4 % del tiempo y solo 3 h de mercado → con el futuro real el mismo
   controlador iguala al oráculo, así que cada punto que falta es error de previsión.
4. **Los datos tienen estructura que se puede usar.** El contador congelado de 22:05 a 06:00; los inversores que
   recortan a 50 kW; `pv ≈ 0,13·irradiancia` con R² 0,99; los escalones de la tarifa 2.0TD y los festivos
   nacionales sin escalón; el hueco solar del CO2 a mediodía; el cierre de agosto y la reapertura del 24/08.
5. **La incertidumbre se gestiona, no se ignora.** Nueve escenarios de carga neta por hora dentro del LP: descarga
   hasta un cuantil alto cuando la hora es cara y hasta uno bajo cuando es marginal (regla del vendedor de
   periódicos). De noche el error es cero y no se cubre nada.
6. **Validación honesta.** Semanas pseudo-test retiradas de los datos como en el servidor, dos protocolos de SOC,
   decisiones por diferencias emparejadas con 2 errores estándar, prueba de fuga automática, 0 infracciones.
7. **Robustez de producción.** Sin pickles, árboles en numpy, 30 s de 60, regla de reserva determinista, `obs`
   sucio tolerado.

### 7.2 Tablas

| Tabla | Contenido |
| --- | --- |
| T1 | Escalera de resultados por modo: sin batería, plantilla, regla de reserva, MPC con perfiles, MPC con modelos, oráculo (P1 y P2; ΔCO2 %, ΔCoste %, Score, infracciones) |
| T2 | Ablaciones A1–A8 con Δ emparejado, SE y % de semanas ganadas |
| T3 | Valor de la información: agente con carga neta real en h0, a 24 h, con mercado real y con todo |

Además: MAE fuera de muestra por variable y horizonte, cobertura empírica de los 9 cuantiles, ms por paso, MB del zip.

### 7.3 Figuras (`reports/figures/`, generadas por `scripts/07_figures.py` e incluidas en el notebook)

| Fig. | Qué muestra |
| --- | --- |
| F1 | Potencia del contador a 5 min durante 3 días: la meseta congelada de 22:05 a 06:00 |
| F2 | MAE de la persistencia de la carga por hora del día |
| F3 | Solar frente a irradiancia (nube de puntos, ajuste, recorte de los inversores) |
| F4 | Perfiles horarios de precio y CO2, laborable frente a fin de semana, con los periodos tarifarios marcados |
| F5 | Carga neta horaria en Estándar y en Difícil (cuantiles 5–50–95) y fracción de horas con excedente |
| F6 | Media diaria de carga a lo largo del año con festivos, cierres y huecos de test sombreados |
| F7 | Una semana de operación: `w`, carga neta real y prevista con su banda, consigna y SOC, agente frente a oráculo |
| F8 | Error de previsión por horizonte (consumo, solar, precio, CO2) frente a persistencia |
| F9 | Calibración de los cuantiles de carga neta (cobertura nominal frente a empírica) |
| F10 | Contribución al score por semana: agente frente a oráculo, P2 |

Notebook `notebooks/analisis_jurado.ipynb` (construido con `nbformat`): secciones Datos → Física y score →
Hallazgos → Previsión → Control → Validación → Resultados → Límites y trabajo futuro. En «Límites» se declara lo no
verificable: protocolo exacto del servidor, composición del 50 % privado y calendario de las semanas de test.
