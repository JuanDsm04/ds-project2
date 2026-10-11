"""Variables de entrada de M2 (Sentence-BERT + LightGBM).

El notebook 05 y ``src/prediccion.py`` usan estas mismas funciones, así que el
entrenamiento y la aplicación construyen exactamente la misma matriz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UMBRAL_CORTA = 5  # mismo criterio que la bandera short_explanation del notebook 00

COLUMNAS_TABULARES = [
    "pregunta_cod",
    "pregunta_respuesta_cod",
    "es_correcta_num",
    "n_caracteres",
    "n_palabras",
    "short_explanation",
]
# Índices (dentro de las columnas tabulares) que LightGBM trata como categóricos
CATEGORICAS = [0, 1]


def codigos_categoricos(respuestas: pd.DataFrame) -> dict[str, dict]:
    """Códigos enteros fijos para la pregunta y el par pregunta|respuesta (60 pares)."""
    qids = sorted(respuestas["QuestionId"].astype(int).unique())
    pares = sorted(f"{int(q)}|{a}" for q, a in zip(respuestas["QuestionId"], respuestas["MC_Answer_norm"]))
    return {
        "pregunta": {str(q): i for i, q in enumerate(qids)},
        "pregunta_respuesta": {p: i for i, p in enumerate(pares)},
    }


def textos_para_embedding(dfp: pd.DataFrame, texto: str) -> list[str]:
    """``texto`` = "entrada" (pregunta + respuesta + explicación) o "explicacion"."""
    if texto == "entrada":
        return dfp["texto_entrada"].tolist()
    if texto == "explicacion":
        return dfp["StudentExplanation"].astype(str).str.strip().tolist()
    raise ValueError(f"Texto desconocido: {texto}")


def variables_tabulares(dfp: pd.DataFrame, codigos: dict[str, dict]) -> np.ndarray:
    """Matriz (n, 6) con las variables tabulares; ``dfp`` viene de ``preparar_dataframe``."""
    expl = dfp["StudentExplanation"].astype(str).str.strip()
    # Pares o preguntas desconocidas -> NaN (LightGBM los manda a la rama de faltantes)
    preg = dfp["pregunta_str"].map(codigos["pregunta"]).astype(float)
    par = dfp["pregunta_respuesta"].map(codigos["pregunta_respuesta"]).astype(float)
    corr = dfp["es_correcta"].map(lambda c: np.nan if c is None else float(bool(c))).astype(float)
    return np.column_stack([
        preg,
        par,
        corr,
        expl.str.len(),
        expl.str.split().str.len(),
        (expl.str.len() <= UMBRAL_CORTA).astype(float),
    ]).astype(np.float32)


def construir_matriz(embeddings: np.ndarray, tabulares: np.ndarray) -> np.ndarray:
    """Une variables tabulares (primero, para que los índices categóricos sean fijos) y embeddings."""
    return np.hstack([tabulares, embeddings.astype(np.float32)])


def nombres_columnas(dim_embedding: int) -> list[str]:
    return COLUMNAS_TABULARES + [f"emb_{i}" for i in range(dim_embedding)]
