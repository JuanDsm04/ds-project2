"""Métricas usadas para tunear y comparar modelos."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def top_k(proba: np.ndarray, clases, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (etiquetas, probabilidades) de las k clases más probables por fila."""
    clases = np.asarray(clases)
    orden = np.argsort(-proba, axis=1)[:, :k]
    return clases[orden], np.take_along_axis(proba, orden, axis=1)


def map_at_3(y_true, top3) -> float:
    """MAP@3 del reto: 1, 1/2 o 1/3 según la posición de la etiqueta correcta; 0 si no está."""
    y_true = np.asarray(y_true)
    top3 = np.asarray(top3)
    puntaje = np.zeros(len(y_true))
    for k in range(min(3, top3.shape[1])):
        acierto = (top3[:, k] == y_true) & (puntaje == 0)
        puntaje[acierto] = 1.0 / (k + 1)
    return float(puntaje.mean())


def resumen_metricas(y_true, top3) -> dict[str, float]:
    """Macro-F1, exactitud (top-1) y MAP@3."""
    top3 = np.asarray(top3)
    pred = top3[:, 0]
    return {
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "map_at_3": map_at_3(y_true, top3),
    }
