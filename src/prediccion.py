"""
Interfaz única de predicción (Persona 1).

La aplicación y los notebooks llaman a estas funciones sin preocuparse por el
preprocesamiento: la normalización de LaTeX, la corrección de la respuesta, la
construcción de la entrada y la máscara de etiquetas ocurren aquí dentro.

Uso rápido::

    from src.prediccion import predecir, modelos_disponibles
    modelos_disponibles()                      # ['M1_tfidf_lr', 'M2_sbert_lgbm', 'M3_deberta']
    predecir("M1_tfidf_lr", 31772, r"\\( \\frac{3}{6} \\)", "3 out of 6 are white")

Para agregar un modelo nuevo (por ejemplo M2, Persona 2):
1. Crear una clase con los atributos ``nombre``, ``clases`` y ``usar_mascara`` y
   un método ``predict_proba(df_preparado) -> np.ndarray`` (filas × clases).
2. Registrarla en ``_REGISTRO`` y, si se quiere, un alias corto en ``ALIAS``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .metricas import top_k
from .preprocesamiento import DIR_MODELOS, cargar_reglas, mascara_etiquetas, preparar_dataframe

RUTA_M1 = DIR_MODELOS / "m1_tfidf_lr.joblib"
DIR_M2 = DIR_MODELOS / "m2_sbert_lgbm"
DIR_M3 = DIR_MODELOS / "m3_deberta"

COLUMNAS_M1 = ["explicacion_m1", "pregunta_str", "pregunta_respuesta", "correcta_str"]


# ---------------------------------------------------------------------------
# Adaptadores de cada modelo
# ---------------------------------------------------------------------------
class ModeloM1:
    """M1: TF-IDF + regresión logística (scikit-learn)."""

    nombre = "M1_tfidf_lr"
    ruta = RUTA_M1

    def __init__(self) -> None:
        artefacto = joblib.load(self.ruta)
        self.pipeline = artefacto["pipeline"]
        self.clases = np.asarray(self.pipeline.classes_)
        self.usar_mascara = bool(artefacto.get("usar_mascara", True))
        self.info = {k: v for k, v in artefacto.items() if k != "pipeline"}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[COLUMNAS_M1])


class ModeloM2:
    """M2: embeddings Sentence-BERT + variables tabulares con LightGBM."""

    nombre = "M2_sbert_lgbm"
    ruta = DIR_M2 / "config_proyecto.json"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        with open(self.ruta, encoding="utf-8") as f:
            cfg = json.load(f)
        self.modelo = joblib.load(DIR_M2 / "lgbm.joblib")
        self.clases = np.asarray(self.modelo.classes_)
        self.usar_mascara = bool(cfg.get("usar_mascara", True))
        self.texto = cfg["texto"]
        self.codigos = cfg["codigos"]
        self.info = cfg
        self.encoder = SentenceTransformer(str(DIR_M2 / "encoder"), device="cpu")

    def predict_proba(self, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        from .features_m2 import construir_matriz, textos_para_embedding, variables_tabulares

        emb = self.encoder.encode(
            textos_para_embedding(df, self.texto), batch_size=batch_size, normalize_embeddings=True
        )
        return self.modelo.predict_proba(construir_matriz(emb, variables_tabulares(df, self.codigos)))


class ModeloM3:
    """M3: DeBERTa-v3 con ajuste fino (Hugging Face Transformers)."""

    nombre = "M3_deberta"
    ruta = DIR_M3 / "config_proyecto.json"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        with open(self.ruta, encoding="utf-8") as f:
            cfg = json.load(f)
        self.clases = np.asarray(cfg["clases"])
        self.usar_mascara = bool(cfg.get("usar_mascara", True))
        self.max_len = int(cfg.get("max_len", 256))
        self.info = cfg
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(DIR_M3)
        self.model = AutoModelForSequenceClassification.from_pretrained(DIR_M3).to(self.device).eval()

    def predict_proba(self, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        textos = df["texto_entrada"].tolist()
        salidas = []
        with self.torch.no_grad():
            for i in range(0, len(textos), batch_size):
                lote = self.tokenizer(
                    textos[i : i + batch_size],
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**lote).logits.float()
                salidas.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(salidas)


_REGISTRO = {
    ModeloM1.nombre: ModeloM1,
    ModeloM2.nombre: ModeloM2,
    ModeloM3.nombre: ModeloM3,
}
ALIAS = {"M1": "M1_tfidf_lr", "M2": "M2_sbert_lgbm", "M3": "M3_deberta"}

_cache_modelos: dict[str, object] = {}
_cache_reglas: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def _nombre_real(modelo: str) -> str:
    nombre = ALIAS.get(modelo, modelo)
    if nombre not in _REGISTRO:
        raise ValueError(f"Modelo desconocido: {modelo}. Opciones: {list(_REGISTRO)}")
    return nombre


def modelos_disponibles() -> list[str]:
    """Modelos registrados cuyos archivos existen en ``models/``."""
    return [n for n, clase in _REGISTRO.items() if Path(clase.ruta).exists()]


def cargar_modelo(modelo: str):
    """Carga un modelo una sola vez y lo deja en memoria."""
    nombre = _nombre_real(modelo)
    if nombre not in _cache_modelos:
        clase = _REGISTRO[nombre]
        if not Path(clase.ruta).exists():
            raise FileNotFoundError(
                f"No se encontró {clase.ruta}. Descarga los pesos del modelo (ver README)."
            )
        _cache_modelos[nombre] = clase()
    return _cache_modelos[nombre]


def _reglas():
    if not _cache_reglas:
        respuestas, preguntas, misc = cargar_reglas()
        _cache_reglas.update(respuestas=respuestas, preguntas=preguntas, misc=misc)
    return _cache_reglas


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def probabilidades(modelo: str, df: pd.DataFrame, usar_mascara: bool | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Probabilidades por clase (filas × clases) ya con la máscara aplicada.

    Devuelve ``(proba, clases, df_preparado)``.
    """
    m = cargar_modelo(modelo)
    r = _reglas()
    dfp = preparar_dataframe(df, r["respuestas"], r["preguntas"])
    proba = m.predict_proba(dfp)
    if m.usar_mascara if usar_mascara is None else usar_mascara:
        mascara = mascara_etiquetas(m.clases, dfp["QuestionId"], dfp["es_correcta"], r["misc"])
        con_mascara = proba * mascara
        suma = con_mascara.sum(axis=1, keepdims=True)
        proba = np.where(suma > 0, con_mascara / np.where(suma > 0, suma, 1), proba)
    return proba, m.clases, dfp


def predecir_lote(modelo: str, df: pd.DataFrame, k: int = 3, usar_mascara: bool | None = None) -> pd.DataFrame:
    """Predice un DataFrame completo.

    ``df`` necesita ``QuestionId``, ``MC_Answer`` y ``StudentExplanation``
    (``row_id`` y ``QuestionText`` son opcionales). Devuelve ``row_id`` (si
    existe), ``top1..topk`` y ``prob1..probk``.
    """
    proba, clases, dfp = probabilidades(modelo, df, usar_mascara)
    etiquetas, probs = top_k(proba, clases, k)
    out = pd.DataFrame(index=dfp.index)
    if "row_id" in dfp.columns:
        out["row_id"] = dfp["row_id"].values
    for i in range(k):
        out[f"top{i + 1}"] = etiquetas[:, i]
    for i in range(k):
        out[f"prob{i + 1}"] = probs[:, i].round(6)
    return out.reset_index(drop=True)


def predecir(modelo: str, question_id: int, mc_answer: str, explicacion: str, k: int = 3) -> list[dict]:
    """Clasifica una sola respuesta nueva.

    ``mc_answer`` puede venir en LaTeX (como en los datos) o ya normalizada (``"1/3"``).
    Devuelve una lista ordenada de diccionarios con ``etiqueta``, ``category``,
    ``misconception`` y ``prob``.
    """
    if explicacion is None or not str(explicacion).strip():
        raise ValueError("La explicación está vacía.")
    df = pd.DataFrame([{
        "QuestionId": int(question_id),
        "MC_Answer": str(mc_answer),
        "StudentExplanation": str(explicacion),
    }])
    fila = predecir_lote(modelo, df, k=k).iloc[0]
    resultado = []
    for i in range(1, k + 1):
        category, misconception = fila[f"top{i}"].split(":", 1)
        resultado.append({
            "etiqueta": fila[f"top{i}"],
            "category": category,
            "misconception": misconception,
            "prob": float(fila[f"prob{i}"]),
        })
    return resultado


def predecir_todos(question_id: int, mc_answer: str, explicacion: str, k: int = 3) -> dict[str, list[dict]]:
    """Ejecuta ``predecir`` con todos los modelos disponibles."""
    return {m: predecir(m, question_id, mc_answer, explicacion, k) for m in modelos_disponibles()}
