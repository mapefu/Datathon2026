# Resultados de la validación cruzada (22 semanas, modelos sin las semanas evaluadas)

P1 = cada semana por separado con SOC al 50 %; P2 = semanas seguidas con SOC continuo (protocolo del FAQ y del servidor). Score oficial agregado con base común. 0 infracciones en todo.

## T1 · Escalera de resultados

| Modo | Agente | P1 | P2 | ΔCO2 P2 | ΔCoste P2 |
| --- | --- | --- | --- | --- | --- |
| Difícil | Agente previo del equipo (README previo) | 21,73 | 19,89 | — | — |
| Difícil | **Este agente** | 21.71 | 20.34 | 19.47 % | 22.37 % |
| Difícil | Oráculo | 22.84 | 21.45 | 20.58 % | 23.49 % |
| Estándar | Agente previo (agente_anterior) | 6.33 | 5.91 | 5.55 % | 6.75 % |
| Estándar | **Este agente** | 6.33 | 5.91 | 5.59 % | 6.68 % |
| Estándar | Oráculo | 6.35 | 5.93 | 5.59 % | 6.70 % |

## T2 · Ablaciones (Difícil, diferencia emparejada frente al agente final)

| Cambio | Δ P1 | Δ P2 (± SE) | semanas ganadas P2 | Decisión |
| --- | --- | --- | --- | --- |
| Sin noche exacta (σ normal de noche) | -0.015 | -0.016 ± 0.005 | 27 % | se queda el agente final |
| σ de decisión en vez de σ por horizonte | -0.009 | -0.008 ± 0.015 | 41 % | se queda el agente final |
| Sin planificar el arranque del bloque siguiente | +0.000 | -0.383 ± 0.474 | 45 % | se queda el agente final |
| 5 escenarios en vez de 9 | -0.049 | -0.049 ± 0.016 | 23 % | se queda el agente final |
| 150 árboles por modelo en vez de 300 (2× más rápido) | -0.052 | -0.066 ± 0.071 | 32 % | se queda el agente final |

## T3 · Error de previsión fuera de muestra (Difícil, media de los pliegues A y B)

| Horizonte | Carga neta (kW) | ref. | Precio (×10⁻³) | ref. | CO2 (×10⁻³) | ref. |
| --- | --- | --- | --- | --- | --- | --- |
| h0 | 59.1 | 118.1 | — | — | — | — |
| h1-2 | 86.9 | 139.4 | — | — | — | — |
| h3 | 117.5 | 156.8 | 8.18 | 12.27 | 5.25 | 5.43 |
| h3-7 | 127.2 | 173.3 | 11.70 | 24.62 | 8.84 | 10.83 |
| h8-23 | 143.2 | 183.7 | 16.61 | 41.16 | 16.33 | 20.36 |
