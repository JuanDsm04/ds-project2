"""Funciones auxiliares para guardar resultados con el formato acordado."""
from __future__ import annotations

import platform
from pathlib import Path

import pandas as pd

from .preprocesamiento import DIR_RESULTADOS

RUTA_TIEMPOS = DIR_RESULTADOS / "tiempos.csv"
DIR_PREDICCIONES = DIR_RESULTADOS / "predicciones"
DIR_TUNING = DIR_RESULTADOS / "tuning"

COLUMNAS_PREDICCION = ["row_id", "modelo", "top1", "top2", "top3", "prob1", "prob2", "prob3"]


def describir_hardware() -> str:
    """Texto corto con CPU o GPU usada."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return f"CPU ({platform.processor() or platform.machine()})"


def registrar_tiempo(modelo: str, t_entrenamiento: float, t_inferencia: float, hardware: str | None = None) -> pd.DataFrame:
    """Agrega o reemplaza la fila del modelo en results/tiempos.csv."""
    fila = pd.DataFrame([{
        "modelo": modelo,
        "tiempo_entrenamiento_s": round(float(t_entrenamiento), 2),
        "tiempo_inferencia_test_s": round(float(t_inferencia), 2),
        "hardware": hardware or describir_hardware(),
    }])
    RUTA_TIEMPOS.parent.mkdir(parents=True, exist_ok=True)
    if RUTA_TIEMPOS.exists():
        tabla = pd.read_csv(RUTA_TIEMPOS)
        tabla = pd.concat([tabla[tabla["modelo"] != modelo], fila], ignore_index=True)
    else:
        tabla = fila
    tabla.to_csv(RUTA_TIEMPOS, index=False)
    return tabla


def guardar_predicciones(pred: pd.DataFrame, modelo: str) -> Path:
    """Guarda results/predicciones/pred_<modelo>.csv con las columnas acordadas."""
    DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    pred = pred.copy()
    pred["modelo"] = modelo
    ruta = DIR_PREDICCIONES / f"pred_{modelo}.csv"
    pred[COLUMNAS_PREDICCION].to_csv(ruta, index=False)
    return ruta


def guardar_tuning(tabla: pd.DataFrame, modelo: str) -> Path:
    """Guarda results/tuning/tuning_<modelo>.csv."""
    DIR_TUNING.mkdir(parents=True, exist_ok=True)
    ruta = DIR_TUNING / f"tuning_{modelo}.csv"
    tabla.to_csv(ruta, index=False)
    return ruta
