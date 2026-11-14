# Datathon UPC Sostenible 2026 — Objetivo del reto y Modo Difícil

Resumen de la información fija de la plataforma oficial (<https://repteupc.datathon.cat/>), consultada el 14/11/2026.
Versión editable: <https://claude.ai/artifact/3reYSwfRiJmML5bgfBfufw>

## Objetivo del reto

Hay que programar un agente (`BatteryAgent`) que fije cada hora la potencia de carga o descarga de la batería del
Campus Nord durante 2.016 horas de evaluación (12 semanas del año). Lo organizan AED × UPC Sostenible (Datathon 2026).

UPC Sostenible lidera la descarbonización de los campus de la UPC. El campus tiene una instalación fotovoltaica y una
batería industrial de gran capacidad. El agente debe maximizar dos objetivos:

- **Descarbonización (70% del peso):** minimizar el CO2 evitando consumir de la red en las horas en que el mix
  peninsular usa ciclos combinados.
- **Eficiencia económica (30% del peso):** reducir el coste neto de la factura importando energía en horas valle y
  descargando la batería o consumiendo solar en horas punta.

El Modo Difícil evalúa una ampliación de la infraestructura (más batería y más solar) con un ranking independiente.
El Modo Estándar sigue abierto en paralelo.

**Guía rápida (5 minutos)**

1. Descargar el paquete ZIP oficial (13 MB) desde la pestaña «Dades del Campus i Starter Kit».
2. Abrir `starter_kit/tutorial.ipynb` para ver el flujo completo y entrenar un primer modelo.
3. En la pestaña de entrega, probar «Validació Ràpida» con la plantilla `agent.py`, que no gasta cuota.

## Balance de potencia y física de la batería

Cada hora, la potencia intercambiada con la red es la suma directa de la demanda, la solar y la consigna de la batería:

```
P_red = D_campus − G_solar + P_batería
```

La consigna es positiva al cargar y negativa al descargar hacia el campus.

- **Red > 0 kW:** el campus compra electricidad. Genera coste (PVPC) y emisiones según el mix peninsular.
- **Red < 0 kW:** hay excedente solar que se vierte a la red. El enunciado da 0,04 €/kWh de compensación; el FAQ dice
  `export_price = 0.0` y que el excedente no cuenta ningún ahorro de CO2. El Modo Difícil habla de excedentes
  «gratuitos (0 € y 0 kg CO2)». El `environment.py` oficial usa `export_price = 0.0`.

| Parámetro | Modo Estándar | Modo Difícil |
| --- | --- | --- |
| Capacidad (kWh) | 2.000 | 6.000 |
| Potencia máxima (kW) | ±500 | ±1.500 |
| Planta solar | pv_kw × 1 | pv_kw × 12 (la web dice ~1.650 kW; en train el máximo es 1.729 kW) |
| Horizonte de mercado en `obs` | 24 h | 3 h (actual + 2 siguientes) |
| Rendimiento ida y vuelta | 90% | 90% |
| SOC inicial | 50% | 50% |

**Reglas físicas comunes**

- Rendimiento del 90% ida y vuelta: √0,90 ≈ 94,9% al cargar y 94,9% al descargar. Cargar 100 kWh deja 94,9 kWh dentro
  y entrega 90 kWh útiles.
- Solo compensa hacer un ciclo si la hora cara o sucia es al menos un 11,1% (1/0,90) peor que la hora en que se cargó.
- El SOC empieza al 50% solo en la hora 0 del primer día. No se reinicia nunca: si a las 23:00 queda al 15%, a las
  00:00 sigue al 15%.
- No hay degradación (`degradation = false`): se pueden hacer tantos ciclos como se quiera.
- La batería no puede vender a la red. Si se descarga más de lo que consume el campus, el sobrante se vierte sin
  ahorro económico ni de CO2.

## Puntuación y criterio del jurado

```
Score = max(0, 0,70·ΔCO2(%) + 0,30·ΔCoste(%) − 0,20·Infracciones)
```

- **ΔCO2 (%):** reducción porcentual de emisiones respecto al campus sin batería.
- **ΔCoste (%):** ahorro porcentual en la factura respecto al campus sin batería.
- **Infracciones:** cada hora con recorte físico (clipping) o violación de límites penaliza. Ocurre al pedir más carga
  que el espacio libre o más descarga que la energía disponible.

**Leaderboard público y privado.** El ranking visible usa el 50% público de cada semana de test (1.008 horas). El
otro 50% queda oculto hasta el veredicto para evitar el sobreajuste.

**Selección para el jurado.** Cada equipo elige 2 envíos en «🎯 Selecció Oficial per al Jurat» (pestaña de entrega,
con nombre de equipo y PIN). Solo esas dos se evalúan sobre el ranking privado. Si no se elige ninguna, el sistema
toma las dos mejores del público.

**Criterio del jurado.** El leaderboard es una guía cuantitativa, pero no lo decide todo. Pesan de forma decisiva la
presentación final, la claridad del razonamiento técnico, el análisis de datos y la justificación de la estrategia.

**Desempate:** 1) mayor Score, 2) menos infracciones físicas, 3) fecha de registro más antigua.

## Modo Difícil: qué cambia

El Modo Difícil triplica la batería, multiplica por 12 la solar y recorta la previsión de mercado a 3 horas. Por eso
predecir la demanda, la solar y el mercado pasa a ser decisivo.

1. **Más placas y más batería (pv_kw × 12; 6.000 kWh; ±1.500 kW).**
    - En los mediodías soleados la solar supera el consumo y deja excedentes gratuitos (0 € y 0 kg CO2) de hasta
      1.000 kW para cargar la batería.
    - El consumo neto del campus suele estar entre 600 y 1.300 kW. Descargar a ciegas −1.500 kW sin predecir el
      consumo (`contador.csv`) y la solar (`inversor.csv` + `meteo.csv`) vierte el sobrante a la red sin ahorro.
2. **Ventana de mercado de solo 3 horas.** `obs["preus_properes_24h"]` y `obs["co2_properes_24h"]` solo traen la hora
   actual y las 2 siguientes. Para planificar el ciclo diario de 6.000 kWh hay que predecir precio y CO2 de +3 a +24 h
   con modelos entrenados en `data/train/`.
3. **Meteo en tiempo real.** `obs["noves_dades_5min"]` añade las lecturas de la última hora de `meteo.csv`, además del
   consumo y la solar (ya escalada ×12).

**Entrega y ranking propios.** Los envíos del Modo Difícil tienen cuota independiente y no gastan intentos del Modo
Estándar: 10 hasta el sábado a las 21:00, 10 de sábado 21:00 a domingo 08:00 y 10 desde el domingo a las 08:00. Se
registran en el Ranking del Modo Difícil (test público de 1.008 horas). El formulario pide nombre de equipo, PIN
(mínimo 4 caracteres), nombres de los integrantes la primera vez y el archivo `agent.py` o `submission.zip`. Hay un
botón «🧪 Validació Ràpida Mode Difícil» que no gasta cuota.

## Contrato del agente y diccionario obs

El archivo `agent.py` debe contener la clase `BatteryAgent` con `__init__()` y `act(obs) -> float`. El evaluador crea
una sola instancia y llama a `act(obs)` hora tras hora, así que se puede acumular historial en `self.historial` durante
toda la simulación. El valor devuelto es la consigna en kW: positivo carga, negativo descarga, 0 reposo. En Modo
Difícil el rango es ±1.500 kW. Se acepta `np.float64`.

| Campo | Tipo | Descripción | Modo Difícil |
| --- | --- | --- | --- |
| `timestamp` | pd.Timestamp | Fecha y hora evaluada | Igual |
| `hour` | int | Hora del día (0–23) | Igual |
| `day_of_week` | int | 0 = lunes … 6 = domingo | Igual |
| `soc_kwh` | float | Energía almacenada (kWh) | 0–6.000 kWh |
| `soc_pct` | float | Estado de carga en tanto por uno (0,0–1,0) | Igual |
| `soc_percent` | float | Estado de carga en % (0–100) | Igual |
| `preu_actual` | float | Precio PVPC de la hora (€/kWh) | Igual |
| `co2_actual` | float | Intensidad de CO2 de la red (kg CO2/kWh) | Igual |
| `preus_properes_24h` | list | Precios de las próximas horas (€/kWh) | Solo 3 valores |
| `co2_properes_24h` | list | CO2 de las próximas horas (kg/kWh) | Solo 3 valores |
| `noves_dades_5min` | dict | Telemetría de la última hora: `ultim_consum_kw`, `ultima_solar_kw` | Solar ×12, más `irradiancia_wm2` (W/m²), `temp_ambient_c` (°C), `temp_panel_c` (°C), `vent_ms` (m/s) |
| `cap_max_kwh` | float | Capacidad máxima | 6000.0 (solo Modo Difícil) |
| `power_max_kw` | float | Potencia máxima | 1500.0 (solo Modo Difícil) |

**Saltos temporales.** El test son semanas completas (la última de cada mes) y el train las primeras ~3 semanas. Entre
bloques hay saltos en `obs["timestamp"]`. La web recomienda detectar `(obs["timestamp"] - self.prev_ts) > 2 horas` y
reiniciar las ventanas móviles cortas.

**Lo que no viene en `obs`.** Ni el consumo futuro ni la solar futura. Hay que estimarlos con ML o estadística a
partir de la hora, el día, el mes y `noves_dades_5min`. En Modo Difícil también hay que predecir precio y CO2 más allá
de +3 h.

```python
ultim_consum_kw = obs["noves_dades_5min"]["ultim_consum_kw"]
ultima_solar_kw = obs["noves_dades_5min"]["ultima_solar_kw"]
demanda_neta_kw = ultim_consum_kw - ultima_solar_kw

preus = obs["preus_properes_24h"]   # €/kWh (3 valores en Modo Difícil)
co2 = obs["co2_properes_24h"]       # kg CO2/kWh (3 valores en Modo Difícil)
```

## Datos de entrenamiento

El paquete ZIP (13 MB) trae un año completo de telemetría real del campus y del mercado en 6 CSV dentro de
`data/train/`, más `starter_kit/tutorial.ipynb`, `starter_kit/agent.py`, `starter_kit/environment.py` (simulador
físico local) y `docs/`.

| Fichero | Subsistema | Frecuencia | Variable principal |
| --- | --- | --- | --- |
| `contador.csv` | Centro de transformación | 15 min | Demanda (kW) = −`Contador - POTENCIA ACTIVA EXPORTADA (kW)` |
| `inversor.csv` | 3 inversores fotovoltaicos | 5 min | Solar total (kW) = suma de la POTENCIA ACTIVA de los 3 |
| `meteo.csv` | Estación meteorológica del tejado | 5 min | IRRADIANCIA (W/m²), TEMPERATURA AMBIENTE (°C), VELOCIDAD VIENTO |
| `entradas_salidas.csv` | Autómata industrial | 5 min | Señales binarias de maquinaria (DI_00 a DI_07) |
| `precio_pvpc.csv` | Mercado eléctrico peninsular | 1 h | `precio_eur_per_kwh` (€/kWh) |
| `co2_red.csv` | Mix de generación REE | 1 h | `co2_kg_per_kwh` (kg CO2/kWh) |

- **Signo del contador.** Un valor negativo significa que el campus consume (−1.100 kW = 1.100 kW consumidos). Se
  invierte el signo y se aplica `.clip(lower=0, upper=5000)`.
- **Remuestreo horario.** Con `.resample("1h").mean()` la potencia media en kW equivale a la energía en kWh de esa hora.
- **Modo Difícil.** La solar se escala ×12 sobre `pv_kw`.

```python
import pandas as pd

df_contador = pd.read_csv("data/train/contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
demanda_raw = (-df_contador["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
load_hourly = demanda_raw.resample("1h").mean().dropna()

df_inversor = pd.read_csv("data/train/inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
inv_cols = [c for c in df_inversor.columns if "POTENCIA ACTIVA" in c and "FASE" not in c]
pv_raw = df_inversor[inv_cols].sum(axis=1).clip(lower=0)
pv_hourly = pv_raw.resample("1h").mean().dropna()

df_pvpc = pd.read_csv("data/train/precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
df_co2 = pd.read_csv("data/train/co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

df_campus_hourly = pd.DataFrame({
    "load_kw": load_hourly,
    "pv_kw": pv_hourly,
    "precio_eur_per_kwh": df_pvpc["precio_eur_per_kwh"],
    "co2_kg_per_kwh": df_co2["co2_kg_per_kwh"]
}).dropna()
```

## Código oficial: safety clamp y simulación local del Modo Difícil

```python
cap_max = float(obs.get("cap_max_kwh", 6000.0))   # 6.000 kWh en Modo Difícil
p_max = float(obs.get("power_max_kw", 1500.0))    # ±1.500 kW en Modo Difícil
soc_kwh = float(obs["soc_kwh"])
eta = 0.90 ** 0.5
eps = 1e-4

max_carga = max(0.0, min(p_max, (cap_max - soc_kwh) / eta - eps))
max_descarga = max(0.0, min(p_max, soc_kwh * eta - eps))
consigna_segura = float(max(-max_descarga, min(max_carga, consigna_deseada)))
return consigna_segura
```

```python
from starter_kit.environment import CampusBatteryEnv

df_hard = df_campus_hourly.copy()
df_hard["pv_kw"] = df_hard["pv_kw"] * 12.0

spec_hard = {
    "capacity_kwh": 6000.0, "power_kw": 1500.0, "efficiency": 0.90, "soc_init": 0.50,
    "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": 3
}

env_hard = CampusBatteryEnv(df_hard, spec=spec_hard)
```

El `submission.zip` lleva `agent.py` y los modelos en la raíz, sin subcarpetas, y se cargan con
`Path(__file__).parent / "modelo"`. Flujo: programar → `python starter_kit/check_submission.py agent.py` → empaquetar
→ validación rápida web (48 h, gratuita) → enviar.

## Normativa de envíos, entorno y seguridad

| Franja | Ventana | Envíos |
| --- | --- | --- |
| Turno 1 | Sábado hasta las 21:00 | 10 |
| Turno 2 | Sábado 21:00 → domingo 08:00 | 10 |
| Turno 3 | Domingo desde las 08:00 hasta el cierre | 10 |

- **Tiempo límite:** 60 s por ejecución, en un subproceso aislado.
- **Tamaño máximo:** 25 MB (el selector admite 200 MB, pero la norma es 25 MB). Formatos: `.py` o `.zip`.
- **PIN:** clave privada de 4 caracteres por equipo. El primer envío pide también los nombres de los integrantes.
- **Validación rápida:** gratuita e ilimitada, 48 h de prueba.
- **Auditoría:** todas las soluciones se archivan para revisión del jurado.
- **Librerías permitidas:** numpy, pandas, scipy, scikit-learn, joblib, xgboost, lightgbm, math, datetime,
  collections, itertools, random, json, pathlib, typing.
- **Escáner AST:** rechaza os, sys, subprocess, socket, urllib, requests, inspect, eval, exec, etc.

## Discrepancias en la web

- **Penalización por infracción:** la fórmula dice −0,20 por hora; el FAQ, −1,00. Con el safety clamp no hay ninguna.
- **Precio del excedente:** el balance da 0,04 €/kWh; el FAQ y el `environment.py` dicen 0,0.
- **Valores por defecto del clamp oficial:** usa 6.000 kWh y 1.500 kW si faltan `cap_max_kwh` o `power_max_kw`; en
  Modo Estándar deberían ser 2.000 y 500.
