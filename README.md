# Arquitectura Bioinspirada para Localización Robótica Predictiva

Este repositorio requiere un entorno de Python con bibliotecas específicas (PyTorch y utilidades relacionadas). A continuación se describen las formas recomendadas de reproducir el entorno.

## Requisitos
- macOS, Linux o Windows
- Conda (Anaconda/Miniconda/Mamba) recomendado
- Python 3.10+ (ajustar según `environment.yml`)

## Instalación rápida con Conda (recomendada)

1. Crear el entorno desde `environment.yml`:

```bash
conda env create -f environment.yml
```

2. Activar el entorno:

```bash
conda activate torch
```

3. (Opcional) Actualizar el entorno si `environment.yml` cambia:

```bash
conda env update -f environment.yml --prune
```

## Datos y modelos

- Modelos PRM pre-entrenados en `models/PRM/saved_models/`.
- Datos para entrenar y probar el model en `data/`.

## Estructura relevante

- `metrics/`: scripts de métricas (`metrics_prm.py`, `utils_metrics.py`).
- `models/PRM/`: implementación del modelo (`prm.py`), pesos guardados.
- `utils/`: utilidades de imagen, monitoreo, y datasets.

