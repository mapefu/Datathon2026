# Datathon 2026 — Repte UPC

Repositorio del equipo para la datathon: <https://repteupc.datathon.cat/>

## Reto

> TODO: describir el reto, el dataset y la métrica de evaluación.

## Estructura

```
.
├── data/
│   ├── raw/          # datos originales (no versionados)
│   ├── processed/    # datos limpios / features (no versionados)
│   └── external/     # datos externos (no versionados)
├── notebooks/        # exploración y experimentos (prefijo: 01_eda_nombre.ipynb)
├── src/              # código reutilizable (carga, features, modelos)
├── submissions/      # ficheros de envío
└── reports/
    └── figures/      # gráficos para la presentación
```

## Puesta en marcha

```bash
git clone https://github.com/mapefu/Datathon2026.git
cd Datathon2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Coloca los datos del reto en `data/raw/`.

## Equipo

- TODO
