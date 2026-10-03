"""
Preprocesamiento compartido del proyecto.

Este módulo concentra todo lo que se le hace a los datos antes de entrar a un
modelo, para que el entrenamiento (notebooks) y la aplicación (Streamlit) usen
exactamente las mismas transformaciones:

- Lectura correcta de los datos (``keep_default_na=False``).
- Normalización de la notación LaTeX y limpieza del texto.
- Partición fija en entrenamiento, validación y prueba.
- Reglas derivadas del split de entrenamiento: corrección de cada respuesta,
  ideas erróneas posibles por pregunta y máscara de etiquetas válidas.
- Construcción de las columnas de entrada que usa cada modelo.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Rutas y constantes
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATA = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_RESULTADOS = RAIZ / "results"

RUTA_TRAIN_CLEAN = DIR_DATA / "train_clean.csv"
RUTA_PARTICION = DIR_DATA / "splits" / "particion.csv"
RUTA_RESPUESTAS = DIR_DATA / "respuestas_correctas.csv"
RUTA_PREGUNTAS = DIR_DATA / "preguntas.csv"
RUTA_MISC_PREGUNTA = DIR_DATA / "misconceptions_por_pregunta.json"

SEMILLA = 42
PROP_TEST = 0.20          # 20 % del total
PROP_VAL = 0.10           # 10 % del total
UMBRAL_ESTRATO = 10       # combinaciones con menos ejemplos se estratifican por Category


# Lectura de datos
def cargar_train_clean(ruta: str | Path = RUTA_TRAIN_CLEAN) -> pd.DataFrame:
    """Lee train_clean.csv sin convertir el texto "NA" en nulos y agrega ``target``."""
    df = pd.read_csv(ruta, keep_default_na=False)
    df["QuestionId"] = df["QuestionId"].astype(int)
    for col in ["QuestionText", "MC_Answer", "StudentExplanation", "Category", "Misconception"]:
        df[col] = df[col].astype(str).str.strip()
    df["target"] = df["Category"] + ":" + df["Misconception"]
    return df


# Normalización y limpieza de texto
_RE_FRAC = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_RE_TEXTO = re.compile(r"\\(?:text|mathrm|textbf|mathbf)\s*\{([^{}]*)\}")
_SIMBOLOS = {
    r"\times": " × ",
    r"\div": " ÷ ",
    r"\cdot": " · ",
    r"\pm": " ± ",
    r"\leq": " ≤ ",
    r"\geq": " ≥ ",
    r"\neq": " ≠ ",
    r"\%": "%",
    r"\left": "",
    r"\right": "",
}


def normalizar_latex(texto: str) -> str:
    r"""Convierte notación LaTeX simple a texto plano.

    Ejemplos: ``\( \frac{1}{3} \)`` -> ``1/3``; ``\( 3 \frac{1}{3} \)`` -> ``3 1/3``;
    ``\( 6 \times 2 \)`` -> ``6 × 2``.
    """
    if texto is None:
        return ""
    t = str(texto)
    # Fracciones (se repite por si hay fracciones anidadas)
    for _ in range(3):
        nuevo = _RE_FRAC.sub(r"\1/\2", t)
        if nuevo == t:
            break
        t = nuevo
    t = _RE_TEXTO.sub(r"\1", t)
    for comando, reemplazo in _SIMBOLOS.items():
        t = t.replace(comando, reemplazo)
    # Delimitadores matemáticos y espacios de LaTeX
    for delim in (r"\(", r"\)", r"\[", r"\]", "$"):
        t = t.replace(delim, " ")
    t = t.replace(r"\,", " ").replace(r"\ ", " ").replace(r"\!", "")
    # Comandos restantes: se quita la barra y se conserva el nombre
    t = re.sub(r"\\([a-zA-Z]+)", r"\1", t)
    t = t.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", t).strip()


def limpiar_texto(texto: str, minusculas: bool = False) -> str:
    """Limpieza ligera: Unicode NFKC, saltos de línea y espacios repetidos.

    No elimina números, operadores ni palabras vacías (*not*, *more*, *than*...),
    porque expresan el razonamiento del estudiante.
    """
    if texto is None:
        return ""
    t = unicodedata.normalize("NFKC", str(texto))
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if minusculas else t


def normalizar_respuesta(mc_answer: str) -> str:
    """Forma canónica de una respuesta de opción múltiple (sirve como llave)."""
    return limpiar_texto(normalizar_latex(mc_answer))


def texto_correcta(es_correcta) -> str:
    """Representación textual de la corrección: yes / no / unknown."""
    if es_correcta is None or (isinstance(es_correcta, float) and np.isnan(es_correcta)):
        return "unknown"
    return "yes" if bool(es_correcta) else "no"


def construir_entrada(question_text: str, mc_answer: str, es_correcta, explicacion: str) -> str:
    """Texto de entrada para los modelos de lenguaje (M3)."""
    return (
        f"Question: {limpiar_texto(normalizar_latex(question_text))} "
        f"Answer: {normalizar_respuesta(mc_answer)} "
        f"Correct: {texto_correcta(es_correcta)} "
        f"Explanation: {limpiar_texto(explicacion)}"
    )


# Partición
def generar_particion(df: pd.DataFrame, semilla: int = SEMILLA) -> pd.DataFrame:
    """Divide en train (70 %), val (10 %) y test (20 %), estratificando por ``target``.

    - Las combinaciones con menos de ``UMBRAL_ESTRATO`` ejemplos se estratifican
      por ``Category`` (no se puede estratificar por una clase con 1 o 2 filas).
    - Si alguna etiqueta de val/test no aparece en train, esas filas se mueven a
      train: un modelo no puede predecir una clase que nunca vio.
    """
    conteo = df["target"].map(df["target"].value_counts())
    llave = np.where(conteo >= UMBRAL_ESTRATO, df["target"], "rara__" + df["Category"])
    idx = np.arange(len(df))

    idx_resto, idx_test = train_test_split(
        idx, test_size=PROP_TEST, stratify=llave, random_state=semilla
    )
    prop_val_relativa = PROP_VAL / (1 - PROP_TEST)
    idx_train, idx_val = train_test_split(
        idx_resto, test_size=prop_val_relativa, stratify=llave[idx_resto], random_state=semilla
    )

    split = np.empty(len(df), dtype=object)
    split[idx_train], split[idx_val], split[idx_test] = "train", "val", "test"

    vistas = set(df["target"].values[split == "train"])
    sin_ver = (split != "train") & ~df["target"].isin(vistas).values
    split[sin_ver] = "train"

    return pd.DataFrame({"row_id": df["row_id"].values, "split": split})


def cargar_particion(ruta: str | Path = RUTA_PARTICION) -> pd.DataFrame:
    return pd.read_csv(ruta)


def aplicar_particion(df: pd.DataFrame, particion: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Devuelve ``{"train": ..., "val": ..., "test": ...}`` según la partición guardada."""
    if particion is None:
        particion = cargar_particion()
    df = df.merge(particion, on="row_id", how="inner")
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}


# Reglas derivadas del split de entrenamiento
def tabla_respuestas_correctas(df_train: pd.DataFrame) -> pd.DataFrame:
    """Corrección de cada par (pregunta, respuesta) por voto mayoritario en train."""
    tmp = df_train.assign(
        es_correcta=df_train["Category"].str.startswith("True"),
        MC_Answer_norm=df_train["MC_Answer"].map(normalizar_respuesta),
    )
    tabla = (
        tmp.groupby(["QuestionId", "MC_Answer", "MC_Answer_norm"])
        .agg(es_correcta=("es_correcta", "mean"), n=("es_correcta", "size"))
        .reset_index()
    )
    tabla["consistencia"] = np.maximum(tabla["es_correcta"], 1 - tabla["es_correcta"]).round(4)
    tabla["es_correcta"] = tabla["es_correcta"] >= 0.5
    return tabla.sort_values(["QuestionId", "MC_Answer_norm"]).reset_index(drop=True)


def tabla_preguntas(df: pd.DataFrame) -> pd.DataFrame:
    """Enunciado original y normalizado de cada pregunta."""
    preg = df.drop_duplicates("QuestionId")[["QuestionId", "QuestionText"]].copy()
    preg["QuestionText_norm"] = preg["QuestionText"].map(lambda t: limpiar_texto(normalizar_latex(t)))
    return preg.sort_values("QuestionId").reset_index(drop=True)


def misconceptions_por_pregunta(df_train: pd.DataFrame) -> dict[str, list[str]]:
    """Ideas erróneas observadas en train para cada pregunta."""
    con_misc = df_train[df_train["Misconception"] != "NA"]
    return {
        str(qid): sorted(grupo["Misconception"].unique().tolist())
        for qid, grupo in con_misc.groupby("QuestionId")
    }


def guardar_reglas(df_train: pd.DataFrame, df_completo: pd.DataFrame) -> None:
    """Guarda respuestas_correctas.csv, preguntas.csv y misconceptions_por_pregunta.json."""
    tabla_respuestas_correctas(df_train).to_csv(RUTA_RESPUESTAS, index=False)
    tabla_preguntas(df_completo).to_csv(RUTA_PREGUNTAS, index=False)
    with open(RUTA_MISC_PREGUNTA, "w", encoding="utf-8") as f:
        json.dump(misconceptions_por_pregunta(df_train), f, ensure_ascii=False, indent=2)


def cargar_reglas() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    respuestas = pd.read_csv(RUTA_RESPUESTAS, keep_default_na=False)
    preguntas = pd.read_csv(RUTA_PREGUNTAS, keep_default_na=False)
    with open(RUTA_MISC_PREGUNTA, encoding="utf-8") as f:
        misc = json.load(f)
    return respuestas, preguntas, misc


def mascara_etiquetas(
    clases: list[str] | np.ndarray,
    question_ids,
    es_correcta,
    misc_por_pregunta: dict[str, list[str]],
) -> np.ndarray:
    """Matriz booleana (n_filas, n_clases) con las etiquetas coherentes con cada fila.

    Una etiqueta es válida si:
    - su prefijo True/False coincide con la corrección de la respuesta, y
    - si es una idea errónea, fue observada en train para esa pregunta.
    Si la corrección es desconocida, se permiten ambos prefijos.
    """
    clases = np.asarray(clases)
    prefijo = np.array([c.split("_", 1)[0] for c in clases])
    misc = np.array([c.split(":", 1)[1] for c in clases])
    es_misc = misc != "NA"

    question_ids = np.asarray(question_ids)
    es_correcta = list(es_correcta)
    mascara = np.zeros((len(question_ids), len(clases)), dtype=bool)
    for i, (qid, corr) in enumerate(zip(question_ids, es_correcta)):
        if corr is None or (isinstance(corr, float) and np.isnan(corr)):
            ok_prefijo = np.ones(len(clases), dtype=bool)
        else:
            ok_prefijo = prefijo == ("True" if bool(corr) else "False")
        posibles = set(misc_por_pregunta.get(str(int(qid)), []))
        ok_misc = ~es_misc | np.isin(misc, list(posibles))
        fila = ok_prefijo & ok_misc
        mascara[i] = fila if fila.any() else ok_prefijo  # nunca dejar una fila sin opciones
    return mascara


# Columnas de entrada para los modelos
def preparar_dataframe(
    df: pd.DataFrame,
    respuestas: pd.DataFrame | None = None,
    preguntas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Agrega las columnas que consumen los modelos.

    Requiere ``QuestionId``, ``MC_Answer`` y ``StudentExplanation``. Si falta
    ``QuestionText``, se toma de ``preguntas.csv``.

    Columnas nuevas: ``MC_Answer_norm``, ``es_correcta`` (True/False/None),
    ``explicacion_m1`` (minúsculas), ``pregunta_str``, ``pregunta_respuesta``,
    ``correcta_str`` y ``texto_entrada`` (para M3).
    """
    if respuestas is None or preguntas is None:
        r, p, _ = cargar_reglas()
        respuestas = r if respuestas is None else respuestas
        preguntas = p if preguntas is None else preguntas

    out = df.copy()
    out["QuestionId"] = out["QuestionId"].astype(int)
    if "QuestionText" not in out.columns:
        mapa_preg = dict(zip(preguntas["QuestionId"].astype(int), preguntas["QuestionText"]))
        out["QuestionText"] = out["QuestionId"].map(mapa_preg).fillna("")

    out["MC_Answer_norm"] = out["MC_Answer"].map(normalizar_respuesta)
    mapa_corr = {
        (int(q), a): (str(c) == "True" if isinstance(c, str) else bool(c))
        for q, a, c in zip(respuestas["QuestionId"], respuestas["MC_Answer_norm"], respuestas["es_correcta"])
    }
    out["es_correcta"] = [
        mapa_corr.get((q, a)) for q, a in zip(out["QuestionId"], out["MC_Answer_norm"])
    ]
    out["explicacion_m1"] = out["StudentExplanation"].map(
        lambda t: limpiar_texto(normalizar_latex(t), minusculas=True)
    )
    out["pregunta_str"] = out["QuestionId"].astype(str)
    out["pregunta_respuesta"] = out["pregunta_str"] + "|" + out["MC_Answer_norm"]
    out["correcta_str"] = out["es_correcta"].map(texto_correcta)
    out["texto_entrada"] = [
        construir_entrada(q, a, c, e)
        for q, a, c, e in zip(out["QuestionText"], out["MC_Answer"], out["es_correcta"], out["StudentExplanation"])
    ]
    return out
