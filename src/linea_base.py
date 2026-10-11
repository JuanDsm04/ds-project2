"""Línea base trivial: las 3 etiquetas más frecuentes de cada par (pregunta, respuesta) en train."""
from __future__ import annotations

import pandas as pd

NOMBRE = "linea_base"


def _top(conteo: pd.Series, k: int) -> list[tuple[str, float]]:
    frec = conteo / conteo.sum()
    return list(frec.sort_values(ascending=False, kind="stable").head(k).items())


def ajustar(train_prep: pd.DataFrame, k: int = 3) -> dict:
    """Frecuencias por par, por pregunta y globales (las dos últimas como respaldo)."""
    return {
        "par": {p: _top(g["target"].value_counts(), k) for p, g in train_prep.groupby("pregunta_respuesta")},
        "pregunta": {q: _top(g["target"].value_counts(), k) for q, g in train_prep.groupby("pregunta_str")},
        "global": _top(train_prep["target"].value_counts(), k),
        "k": k,
    }


def predecir(tabla: dict, dfp: pd.DataFrame) -> pd.DataFrame:
    """Top k por fila con el formato de las predicciones; completa con la pregunta y luego el global."""
    k = tabla["k"]
    filas = []
    for par, preg in zip(dfp["pregunta_respuesta"], dfp["pregunta_str"]):
        top = list(tabla["par"].get(par, []))
        for respaldo in (tabla["pregunta"].get(preg, []), tabla["global"]):
            vistas = {e for e, _ in top}
            top += [(e, 0.0) for e, _ in respaldo if e not in vistas][: k - len(top)]
        filas.append(top[:k])
    out = pd.DataFrame({"row_id": dfp["row_id"].values})
    for i in range(k):
        out[f"top{i + 1}"] = [f[i][0] for f in filas]
    for i in range(k):
        out[f"prob{i + 1}"] = [round(f[i][1], 6) for f in filas]
    return out
