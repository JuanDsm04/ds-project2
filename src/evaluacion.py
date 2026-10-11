"""Evaluación de las predicciones sobre el split de prueba (notebook 06)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .metricas import map_at_3
from .utilidades import DIR_PREDICCIONES

NIVELES = ["etiqueta", "correccion", "category", "misconception"]


def cargar_predicciones(modelo: str, verdad: pd.DataFrame) -> pd.DataFrame:
    """Une ``pred_<modelo>.csv`` con la etiqueta real (``verdad``: ``row_id``, ``target``, ...)."""
    pred = pd.read_csv(DIR_PREDICCIONES / f"pred_{modelo}.csv", keep_default_na=False)
    df = pred.merge(verdad, on="row_id", how="inner", validate="one_to_one")
    assert len(df) == len(pred) == len(verdad), f"{modelo}: las filas no coinciden con el split de prueba"
    return df


def descomponer(etiquetas) -> pd.DataFrame:
    """``True_Misconception:WNB`` -> correccion, category, misconception."""
    s = pd.Series(np.asarray(etiquetas))
    category = s.str.split(":", n=1).str[0]
    return pd.DataFrame({
        "etiqueta": s.values,
        "correccion": category.str.split("_", n=1).str[0].values,
        "category": category.values,
        "misconception": s.str.split(":", n=1).str[1].values,
    })


def pares_por_nivel(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(y_true, y_pred) por nivel usando el top 1. ``misconception`` solo considera filas con idea errónea real."""
    real, pred = descomponer(df["target"]), descomponer(df["top1"])
    pares = {n: (real[n].values, pred[n].values) for n in NIVELES}
    con_misc = real["misconception"] != "NA"
    pares["misconception"] = (real.loc[con_misc, "misconception"].values, pred.loc[con_misc, "misconception"].values)
    return pares


def metricas_resumen(df: pd.DataFrame) -> dict[str, float]:
    """Métricas globales de un modelo (top 1 para F1/precisión/recall/exactitud)."""
    y, p = df["target"].values, df["top1"].values
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    out = {
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y, p),
        "map_at_3": map_at_3(y, df[["top1", "top2", "top3"]].values),
    }
    for nivel, (yt, yp) in pares_por_nivel(df).items():
        if nivel == "etiqueta":
            continue
        out[f"f1_macro_{nivel}"] = f1_score(yt, yp, average="macro", zero_division=0)
        out[f"accuracy_{nivel}"] = accuracy_score(yt, yp)
    return {k: float(v) for k, v in out.items()}


def metricas_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    """Precisión, recall, F1 y soporte por clase en cada nivel."""
    filas = []
    for nivel, (yt, yp) in pares_por_nivel(df).items():
        clases = np.unique(yt)  # solo clases presentes en prueba
        prec, rec, f1, sop = precision_recall_fscore_support(yt, yp, labels=clases, zero_division=0)
        filas.append(pd.DataFrame({"nivel": nivel, "clase": clases, "precision": prec,
                                   "recall": rec, "f1": f1, "soporte": sop}))
    return pd.concat(filas, ignore_index=True)


def metricas_por_grupo(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Macro-F1, exactitud y MAP@3 sobre la etiqueta conjunta dentro de cada grupo (p. ej. ``QuestionId``)."""
    filas = []
    for valor, g in df.groupby(columna):
        filas.append({
            columna: valor,
            "n": len(g),
            "f1_macro": f1_score(g["target"], g["top1"], average="macro", zero_division=0),
            "accuracy": accuracy_score(g["target"], g["top1"]),
            "map_at_3": map_at_3(g["target"].values, g[["top1", "top2", "top3"]].values),
        })
    return pd.DataFrame(filas)


def matriz_confusion(y_true, y_pred, clases, normalizar: bool = True) -> pd.DataFrame:
    """Matriz real × predicha; normalizada por fila (recall) si ``normalizar``."""
    tabla = pd.crosstab(pd.Series(y_true, name="real"), pd.Series(y_pred, name="predicha"))
    tabla = tabla.reindex(index=clases, columns=clases, fill_value=0)
    if normalizar:
        tabla = tabla.div(tabla.sum(axis=1).replace(0, 1), axis=0)
    return tabla
