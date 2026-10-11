"""Carga de datos, modelos y estilo compartidos por las páginas de la app."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import paleta as pal  # noqa: E402
from src import preprocesamiento as pp  # noqa: E402
from src import prediccion  # noqa: E402

DIR_METRICAS = RAIZ / "results" / "metricas"

ORDEN_CATEGORIAS = list(pal.COLORES_CATEGORIA)
MODELOS = ["M1_tfidf_lr", "M2_sbert_lgbm", "M3_deberta"]
MODELO_RECOMENDADO = "M1_tfidf_lr"

DESCRIPCION_CATEGORIA = {
    "True_Correct": "Respuesta correcta con una explicación correcta.",
    "False_Correct": "Respuesta incorrecta, pero la explicación muestra un razonamiento correcto.",
    "True_Misconception": "Respuesta correcta, pero la explicación revela una idea errónea.",
    "False_Misconception": "Respuesta incorrecta causada por una idea errónea identificable.",
    "True_Neither": "Respuesta correcta con una explicación vaga o que no justifica el resultado.",
    "False_Neither": "Respuesta incorrecta sin una idea errónea identificable.",
}

DESCRIPCION_MODELO = {
    "linea_base": "Las 3 etiquetas más frecuentes de cada par pregunta y respuesta. No lee la explicación.",
    "M1_tfidf_lr": "TF-IDF de palabras y caracteres con regresión logística.",
    "M2_sbert_lgbm": "Embeddings de Sentence-BERT (MiniLM) con LightGBM.",
    "M3_deberta": "DeBERTa-v3-base con ajuste fino.",
}


def _registrar_plantilla() -> None:
    n = pal.NEUTROS
    eje = dict(gridcolor=n["rejilla"], linecolor=n["texto_suave"], zeroline=False,
               title_font=dict(color=n["texto_suave"]))
    pio.templates["proyecto"] = go.layout.Template(layout=dict(
        font=dict(color=n["texto"], size=13),
        colorway=[pal.COLORES_MODELO[m] for m in MODELOS] + [pal.COLORES_MODELO["linea_base"]],
        xaxis=eje,
        yaxis=eje,
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top", yref="container", font_size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text=""),
        margin=dict(l=10, r=10, t=90, b=10),
        hoverlabel=dict(bgcolor=n["fondo"], font_color=n["texto"]),
    ))


_registrar_plantilla()


def mostrar_figura(fig: go.Figure) -> None:
    # Streamlit cambia la plantilla por defecto de Plotly al importarse y su frontend pinta el fondo con
    # secondaryBackgroundColor si el layout no lo define, así que ambos se fijan aquí.
    fondo = pal.NEUTROS["fondo"]
    fig.update_layout(template="plotly_white+proyecto", legend_title_text="", paper_bgcolor=fondo, plot_bgcolor=fondo)
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


def leyenda_derecha(fig: go.Figure) -> go.Figure:
    """Leyenda vertical a la derecha, para gráficas angostas con muchas series."""
    return fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top"),
                             margin_t=60)


def nombre_modelo(modelo: str) -> str:
    return pal.nombre_modelo(modelo)


@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    """train_clean con la partición y columnas derivadas para explorar."""
    df = pp.aplicar_particion(pp.cargar_train_clean())
    df = pd.concat(df.values(), ignore_index=True)
    df["MC_Answer_norm"] = df["MC_Answer"].map(pp.normalizar_respuesta)
    df["correccion"] = df["Category"].str.split("_").str[0]
    df["tipo"] = df["Category"].str.split("_", n=1).str[1]
    df["n_palabras"] = df["StudentExplanation"].str.split().str.len()
    df["n_caracteres"] = df["StudentExplanation"].str.len()
    return df


@st.cache_data(show_spinner=False)
def cargar_reglas() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    return pp.cargar_reglas()


@st.cache_data(show_spinner=False)
def etiquetas_preguntas(largo: int = 70) -> dict[int, str]:
    """``{QuestionId: "31772 · What fraction of the shape..."}`` para los selectores."""
    preguntas = cargar_reglas()[1]
    return {
        int(q): f"{q} · {t[:largo]}…" if len(t) > largo else f"{q} · {t}"
        for q, t in zip(preguntas["QuestionId"], preguntas["QuestionText_norm"])
    }


@st.cache_data(show_spinner=False)
def cargar_metricas() -> dict[str, pd.DataFrame]:
    """Tablas de results/metricas/ (resumen, por_clase, por_pregunta, confusion_category)."""
    return {
        ruta.stem.removeprefix("metricas_"): pd.read_csv(ruta, keep_default_na=False)
        for ruta in sorted(DIR_METRICAS.glob("*.csv"))
    }


def modelos_disponibles() -> list[str]:
    return prediccion.modelos_disponibles()


@st.cache_resource(show_spinner=False)
def cargar_modelo(modelo: str):
    return prediccion.cargar_modelo(modelo)
