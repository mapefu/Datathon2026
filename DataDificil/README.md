# DataDificil — Datathon UPC Sostenible 2026 (Modo Estándar + Modo Difícil)

Implementación de [BLUEPRINT.md](BLUEPRINT.md): previsión multi-horizonte (LightGBM evaluado con numpy) +
MPC con programación lineal (`scipy.optimize.linprog`). Python: `.venv/bin/python` (enlace al venv del repo
`Datathon2026`). Datos: `data/raw/train` (enlace a los CSV oficiales).

## Estado

| Fase | Contenido | Estado |
| --- | --- | --- |
| 1 | Datos, simulador, score oficial, validación cruzada, oráculo | hecho y verificado |
| 2 | Variables, modelos de previsión, incertidumbre, bundle | hecho (bundle 1,9 MB, paridad exacta) |
| 3 | `BatteryAgent` (detección de modo, LP, Safety Clamp) | hecho: Difícil P2 20,34 / P1 21,71; Estándar = referencia |
| 4 | Verificación final, empaquetado, figuras, notebook | hecho: `dist/submission_*.zip` (1,8 MB) |

**Entrega final: [`submission_v3/`](submission_v3/)** (variables de tendencia del mercado + pérdida Huber
en precio y CO₂; modelos de CV en `data/processed/folds/hubcp`). CV Difícil: P1 21,77 / P2 20,39
(oráculo 22,84 / 21,45). `submission/` es la v1 (P1 21,71 / P2 20,34) y `experimentos/` guarda las
variantes descartadas. Resumen para el jurado en el [README principal](../README.md).

## Comandos

```bash
.venv/bin/python scripts/00_build_data.py --rebuild   # tablas horarias + comprobaciones
.venv/bin/python scripts/01_oracle.py                 # techo con información perfecta
.venv/bin/python tests/env_parity.py                  # simulador = oficial; arnés reproduce 6,10
.venv/bin/python scripts/02_train.py --fold A B full  # modelos por pliegue y finales (submission/)
.venv/bin/python scripts/02_train.py --eval A B       # error de previsión fuera de muestra
.venv/bin/python tests/features_causal.py             # las variables no leen el futuro
.venv/bin/python scripts/03_cv.py --agent submission/agent.py --mode hard --name v1     # validación P1/P2
.venv/bin/python scripts/04_package_verify.py        # zip + validador oficial + AST + tiempo + estrés + fuga
.venv/bin/python scripts/05_figures.py               # reports/figures + reports/RESULTADOS.md
.venv/bin/python scripts/06_notebook.py              # notebooks/analisis_jurado.ipynb
```

## Estructura

```
src/dd/        paths, data, calendar_upc, env, score, cv, dataset, export, train, forecast_eval
submission/    agent.py (runtime único: variables, bosque, agente) + bundle.npz + params.json
scripts/       00_build_data, 01_oracle, 02_train, 03_cv, 04_package_verify, 05_figures, 06_notebook
tests/         env_parity, features_causal
reports/       cv/*.json, figures/, RESULTADOS.md, train_log.txt
notebooks/     analisis_jurado.ipynb
dist/          submission_<fecha>.zip (el mismo zip vale para los dos modos)
```

## Notas

- La validación desactiva el gobernador de tiempo del agente (`TIME_*_S = 1e12`); el tiempo se mide aparte en
  `04_package_verify.py`. Con el gobernador activo y la máquina cargada, P2 cae a 17,5 porque el agente pasa a la
  regla de reserva: no es el algoritmo.
- Plan B de velocidad: modelos de 150 rondas (`02_train.py --rounds 150 --lgb max_depth=8 learning_rate=0.1`),
  2× más rápidos en el bosque y −0,07 en P2.
- El `submission/agent.py` del repo `Datathon2026` (commit 5d5a489) da 6,035 en Estándar frente al 6,098 de
  `agente_anterior/`.
