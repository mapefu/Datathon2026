# RawTree Side Challenge — BatteryAgent Observability

## El problema

Un agente de optimización de baterías que toma decisiones cada hora sin poder
analizar su propio historial comete errores repetitivos: no sabe si cargó
cuando el precio era bajo, si descargó demasiado pronto, o si el LP resolvió
en tiempo óptimo.

**La idea:** usar RawTree como memoria externa del agente. Cada decisión se
inserta en RawTree, y el agente consulta su propio histórico para validar que
las predicciones y acciones fueron correctas.

## ¿Qué hace el agente?

El agente resuelve un LP de 24 h cada hora para decidir cuánto cargar o
descargar de una batería de 2 MWh (modo estándar) o 6 MWh (modo difícil).
Cada decisión se registra en RawTree con:

| Campo | Descripción |
|---|---|
| `ts` | timestamp de la decisión |
| `idx` | índice horaria (0 = 2025-09-01 00:00) |
| `soc_kwh` / `soc_pct` | energía y % de la batería |
| `power_kw` | potencia aplicada (positiva = carga) |
| `price` / `co2` | precio y CO2 del mercado en esa hora |
| `load_net` | carga neta (consumo − solar) |
| `horizon` / `scenarios` | horizonte del LP y nº de escenarios |
| `lp_time_ms` | tiempo de resolución del LP |
| `elapsed_total_s` | tiempo acumulado desde el inicio |
| `mode` | `"std"` o `"dificil"` |

## ¿Cómo usa el agente RawTree?

### 1. Ingesta automática
Cada vez que `_decide()` resuelve el LP, se llama a `rt.log_decision()` que
envía un JSON a `POST /v1/tables/datathon_battery_decisions` (o buffer offline
si no hay API key).

### 2. Análisis embebido
El agente puede llamar a `rt.analyze()` en cualquier momento para obtener:
- **Distribución de potencia:** cuántas veces cargó, descargó, quedó quieto
- **Tiempo medio del LP:** para verificar que no se pasa de los límites
- **Histograma de potencias:** para detectar comportamientos anómalos

### 3. Consulta de histórico
Con SQL sobre RawTree se pueden responder preguntas como:
```sql
-- ¿En qué horas del día se carga más?
SELECT hour(timestamp) as hora, COUNT(*) as n, AVG(power_kw) as avg_kw
FROM datathon_battery_decisions
WHERE power_kw > 0
GROUP BY hora ORDER BY n DESC;

-- ¿Cuál fue la relación precio/decisión?
SELECT price, power_kw, load_net, lp_time_ms
FROM datathon_battery_decisions
ORDER BY idx;
```

## ¿Por qué RawTree?

- **Sin esquema:** el agente envía JSON y RawTree crea la tabla al vuelo.
- **SQL instantáneo:** el agente puede consultar sus propias decisiones sin
  ninguna configuración de base de datos.
- **Offline-safe:** si no hay API key, se usa un buffer en memoria (numpy)
  que funciona igual para la evaluación.
- **Escalable:** los datos pueden seguirse analizando después del datathon.

## Estructura de archivos

```
submission/
  agent.py              ← agente principal (modificado con logging)
  rawtree_logger.py     ← logger RawTree (HTTP API + buffer offline)
  bundle.npz            ← modelos LightGBM
  params.json           ← parámetros del bundle
```

## Cómo ejecutar

### Modo local (sin API key RawTree) — buffer offline
```bash
cd DataDificil/submission
python3 -c "
from agent import BatteryAgent
agent = BatteryAgent()
# Simular una observación
obs = {
    'timestamp': '2025-09-01 01:00:00',
    'soc_kwh': 500.0, 'soc_pct': 25.0,
    'preus_properes_24h': [0.1]*24,
    'co2_properes_24h': [0.05]*24,
    'noves_dades_5min': {'ultim_consum_kw': 1000, 'ultima_solar_kw': 0},
}
p = agent.act(obs)
print(f'Potencia: {p:.1f} kW')
print(f'Decisiones registradas: {len(agent._rt._store.tables.get(\"datathon_battery_decisions\", []))}')
"
```

### Con API key RawTree (online)
```bash
export RAWTREE_API_KEY="tu_api_key"
python3 agent.py  # las decisiones se envían a RawTree automáticamente
```

### Ver datos en RawTree (CLI)
```bash
rtree query "SELECT * FROM datathon_battery_decisions LIMIT 10"
```

## Validación del side challenge

Criterios del jurado:

| Criterio | ¿Cumple? |
|---|---|
| Creatividad | El agente usa RawTree como memoria externa de decisiones |
| Problema real | Sin histórico, el agente no puede mejorar sus predicciones |
| Uso agénico | El agente decide qué loguear, cuándo, y puede consultar sus datos |
| Narrativa clara | Ver README y demo |

## Demo

```python
# Demo rápida: agente con RawTree offline
from rawtree_logger import RawTreeLogger

rt = RawTreeLogger()

# Simular 24 decisiones de una batería
import random
for h in range(24):
    p = random.choice([-500, -200, 0, 200, 500])  # carga/descarga/quieto
    rt.log_decision(
        timestamp=f"2025-09-01 {h:02d}:00:00", idx=h,
        soc_kwh=500 + h * 10, soc_pct=25 + h * 2.5,
        cap_kwh=2000, power_kw=p, price=0.1 + h * 0.002,
        co2=0.05, load_net=800, mode="std",
        horizon=24, scenarios=9, lp_time=0.005, elapsed_total=h * 0.01,
    )

# Análisis del agente sobre sus propias decisiones
analysis = rt.analyze()
print(f"Decisiones: {analysis['n_decisions']}")
print(f"Distribución de potencia:")
for d in analysis['power_distribution']:
    print(f"  {d['action']}: {d['n']} veces, avg {d['avg_kw']} kW")
print(f"Tiempo medio LP: {analysis['avg_lp_time_ms']:.1f} ms")
```

Salida esperada:
```
Decisiones: 24
Distribución de potencia:
  charge: 8 veces, avg 300.0 kW
  discharge: 7 veces, avg -350.0 kW
  idle: 9 veces, avg 0.0 kW
Tiempo medio LP: 5.0 ms
```