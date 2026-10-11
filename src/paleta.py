"""Paleta de colores compartida (figuras del informe y aplicación).

Basada en Okabe–Ito, segura para daltonismo. Los colores semánticos tienen
siempre el mismo significado: verde = correcto, naranja rojizo = idea errónea,
gris = ninguna / línea base.
"""
from __future__ import annotations

# Colores por modelo
COLORES_MODELO = {
    "M1_tfidf_lr": "#0072B2",
    "M2_sbert_lgbm": "#E69F00",
    "M3_deberta": "#009E73",
    "M4_llm": "#CC79A7",
    "linea_base": "#999999",
}

NOMBRES_MODELO = {
    "M1_tfidf_lr": "M1 · TF-IDF + RL",
    "M2_sbert_lgbm": "M2 · SBERT + LightGBM",
    "M3_deberta": "M3 · DeBERTa-v3",
    "M4_llm": "M4 · LLM + LoRA",
    "linea_base": "Línea base",
}

# Colores semánticos por tipo de explicación
COLORES_TIPO = {
    "Correct": "#009E73",
    "Misconception": "#D55E00",
    "Neither": "#999999",
}

# Las 6 categorías: el tono indica el tipo de explicación y la intensidad, si la respuesta es correcta
COLORES_CATEGORIA = {
    "True_Correct": "#009E73",
    "False_Correct": "#7FCEB9",
    "True_Misconception": "#EBAE80",
    "False_Misconception": "#D55E00",
    "True_Neither": "#BBBBBB",
    "False_Neither": "#777777",
}

# Neutros y acento para fondos, texto y mapas de calor
NEUTROS = {
    "fondo": "#FFFFFF",
    "fondo_suave": "#F5F5F5",
    "texto": "#222222",
    "texto_suave": "#555555",
    "rejilla": "#DDDDDD",
}
ACENTO = "#0072B2"
MAPA_SECUENCIAL = "Blues"   # matrices de confusión
MAPA_RENDIMIENTO = "viridis"  # F1 por pregunta y modelo (perceptualmente uniforme)


def color_modelo(modelo: str) -> str:
    return COLORES_MODELO.get(modelo, NEUTROS["texto_suave"])


def nombre_modelo(modelo: str) -> str:
    return NOMBRES_MODELO.get(modelo, modelo)


def aplicar_estilo() -> None:
    """Estilo base de matplotlib para las figuras estáticas."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": NEUTROS["fondo"],
        "axes.facecolor": NEUTROS["fondo"],
        "axes.edgecolor": NEUTROS["texto_suave"],
        "axes.labelcolor": NEUTROS["texto"],
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": NEUTROS["rejilla"],
        "grid.linewidth": 0.6,
        "xtick.color": NEUTROS["texto_suave"],
        "ytick.color": NEUTROS["texto_suave"],
        "text.color": NEUTROS["texto"],
        "font.size": 10,
        "legend.frameon": False,
    })
