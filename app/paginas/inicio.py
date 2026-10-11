import pandas as pd
import streamlit as st

from comun import (DESCRIPCION_CATEGORIA, DESCRIPCION_MODELO, MODELO_RECOMENDADO, MODELOS, ORDEN_CATEGORIAS,
                   cargar_datos, cargar_metricas, modelos_disponibles, nombre_modelo, pal)

df = cargar_datos()
resumen = cargar_metricas()["resumen"].set_index("modelo")

st.title("Ideas erróneas en matemáticas")
st.caption("Proyecto 2 de Data Science, UVG 2026. Reto MAP Charting Student Math Misunderstandings de Kaggle.")

st.markdown(
    "Estudiantes responden preguntas de diagnóstico de opción múltiple y explican por escrito su respuesta. "
    "La app clasifica cada explicación según si la respuesta es correcta, el tipo de razonamiento y la idea "
    "errónea que revela, si existe."
)

c = st.columns(4)
c[0].metric("Respuestas etiquetadas", f"{len(df):,}")
c[1].metric("Preguntas", df["QuestionId"].nunique())
c[2].metric("Etiquetas posibles", df["target"].nunique())
c[3].metric("Ideas erróneas", df.loc[df["Misconception"] != "NA", "Misconception"].nunique())

st.subheader("Cómo usar la app")
c = st.columns(3)
c[0].markdown("##### Explorar datos\nDistribución de categorías e ideas erróneas, diferencias entre preguntas y ejemplos reales.")
c[1].markdown("##### Clasificar\nElige una pregunta y una respuesta, escribe la explicación y compara lo que predice cada modelo.")
c[2].markdown("##### Rendimiento de modelos\nMétricas sobre el conjunto de prueba, errores por clase y por pregunta, y tiempos de cómputo.")

st.subheader("Qué predice cada modelo")
st.markdown(
    "Una etiqueta `Category:Misconception`. La categoría combina si la respuesta elegida es correcta "
    "(`True` o `False`) con el tipo de explicación. La idea errónea es `NA` cuando no aplica."
)
chips = "".join(
    f'<div style="display:flex;gap:.6rem;align-items:center;margin:.25rem 0">'
    f'<span style="background:{pal.COLORES_CATEGORIA[cat]};width:1rem;height:1rem;border-radius:3px;flex:none"></span>'
    f'<code>{cat}</code><span>{DESCRIPCION_CATEGORIA[cat]}</span></div>'
    for cat in ORDEN_CATEGORIAS
)
st.markdown(chips, unsafe_allow_html=True)

st.subheader("Modelos")
disponibles = set(modelos_disponibles())
tabla = pd.DataFrame([
    {
        "Modelo": nombre_modelo(m),
        "Enfoque": DESCRIPCION_MODELO[m],
        "Macro-F1": resumen.loc[m, "f1_macro"],
        "MAP@3": resumen.loc[m, "map_at_3"],
        "Estado": "Disponible" if m in disponibles else ("No aplica" if m == "linea_base" else "Faltan los pesos"),
    }
    for m in ["linea_base", *MODELOS]
])
st.dataframe(
    tabla,
    hide_index=True,
    width="stretch",
    column_config={
        "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
        "MAP@3": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.caption(
    f"Métricas sobre el conjunto de prueba (7,340 respuestas). Modelo recomendado {nombre_modelo(MODELO_RECOMENDADO)}, "
    "que empata con M3 en macro-F1 y exactitud con una fracción del costo."
)
if len(disponibles) < len(MODELOS):
    st.info("Hay modelos sin pesos en `models/`. Descárgalos siguiendo el README para usarlos en Clasificar.")

st.subheader("Hallazgos principales")
st.markdown(
    "- El conjunto está muy desbalanceado. `True_Correct` reúne el 40 % de las respuestas y algunas etiquetas "
    "tienen menos de 5 ejemplos.\n"
    "- Cada pregunta tiene solo de 2 a 4 ideas erróneas posibles, y el par pregunta y respuesta determina si la "
    "respuesta es correcta en el 99.95 % de los casos. Los modelos aprovechan esa estructura.\n"
    "- Leer la explicación casi duplica el macro-F1 frente a la línea base. La mayor dificultad está en "
    "distinguir una explicación correcta de una que no aporta (`Neither`)."
)
