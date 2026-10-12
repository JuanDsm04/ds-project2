import time

import pandas as pd
import plotly.express as px
import streamlit as st

from comun import (DESCRIPCION_CATEGORIA, MODELO_RECOMENDADO, MODELOS, cargar_descripciones, cargar_modelo,
                   cargar_reglas, etiquetas_preguntas, modelos_disponibles, mostrar_figura, nombre_modelo, pal,
                   prediccion)
from src.features_m2 import UMBRAL_CORTA

# Respuestas reales del split de prueba (row_id 35084, 19736, 30348, 9641 y 7979)
EJEMPLOS = [
    ("Bien explicada", "True_Correct", 104665, "48 hours", "because 12 divided by 3 = 4. 192 divided by 4 = 48."),
    ("Idea errónea", "False_Misconception", 32835, "6",
     "I believe this because 6 is the greatest andd if a number has decimals that means it is smaller."),
    ("Correcta con error", "True_Misconception", 89443, "-3",
     "If you do -8 - -5 it's the same as 8-5=3 but just make it negative."),
    ("Sin justificar", "True_Neither", 31777, "72", "I did the sum and got this answer"),
    ("Error al marcar", "False_Correct", 31777, "24",
     "120 divided by 5 is 24 multiplied by 3 is 72 so the answer is B."),
]

respuestas, preguntas, _ = cargar_reglas()
nombres_preg = etiquetas_preguntas()
descripciones = cargar_descripciones()
disponibles = modelos_disponibles()


def cargar_ejemplo(qid: int, respuesta: str, explicacion: str) -> None:
    st.session_state["pregunta"] = qid
    st.session_state[f"respuesta_{qid}"] = respuesta
    st.session_state["explicacion"] = explicacion


def tarjeta(modelo: str, pred: dict, ms: float, compacta: bool = False) -> None:
    """Resultado principal de un modelo. La versión compacta omite las descripciones."""
    cat, misc = pred["category"], pred["misconception"]
    suave = pal.NEUTROS["texto_suave"]
    color = pal.COLORES_CATEGORIA.get(cat, suave)
    if misc == "NA":
        bloque_misc = f'<div style="margin-top:.6rem;color:{suave}">Sin idea errónea</div>'
    else:
        detalle = "" if compacta else f"<br>{descripciones.get(misc, '')}"
        bloque_misc = (f'<div style="margin-top:.6rem"><span style="color:{suave}">Idea errónea</span><br>'
                       f'<b>{misc}</b>{detalle}</div>')
    desc_cat = "" if compacta else f"<div>{DESCRIPCION_CATEGORIA.get(cat, '')}</div>"
    st.markdown(
        f'<div style="border-left:6px solid {color};background:{pal.NEUTROS["fondo_suave"]};border-radius:6px;'
        f'padding:.8rem 1rem;margin-bottom:.8rem">'
        f'<div style="font-weight:600"><span style="display:inline-block;width:.75rem;height:.75rem;'
        f'border-radius:2px;margin-right:.4rem;background:{pal.color_modelo(modelo)}"></span>'
        f'{nombre_modelo(modelo)}</div>'
        f'<div style="font-size:1.3rem;font-weight:600;margin-top:.3rem">{cat}</div>{desc_cat}{bloque_misc}'
        f'<div style="margin-top:.6rem;color:{suave};font-size:.9rem">'
        f'Probabilidad {pred["prob"]:.0%} · {ms:,.0f} ms</div></div>',
        unsafe_allow_html=True,
    )


st.title("Clasificar una respuesta")
st.markdown(
    "Elige la pregunta y la opción que marcó el estudiante, escribe su explicación en inglés (el idioma de los "
    "datos de entrenamiento) y elige con qué modelo clasificarla."
)

if not disponibles:
    st.error("No hay modelos disponibles. Descarga los pesos en `models/` siguiendo el README.")
    st.stop()

st.caption("Ejemplos reales del conjunto de prueba")
cols = st.columns(len(EJEMPLOS))
for col, (titulo, cat, qid, resp, expl) in zip(cols, EJEMPLOS):
    col.button(titulo, on_click=cargar_ejemplo, args=(qid, resp, expl), width="stretch",
               help=f"{cat}. {DESCRIPCION_CATEGORIA[cat]}")

col_entrada, col_resultado = st.columns([2, 3], gap="large")

with col_entrada:
    qid = st.selectbox("Pregunta", list(nombres_preg), format_func=nombres_preg.get, key="pregunta")
    st.info(preguntas.set_index("QuestionId").loc[qid, "QuestionText_norm"])

    opciones = respuestas[respuestas["QuestionId"] == qid]
    respuesta = st.selectbox("Respuesta elegida", opciones["MC_Answer_norm"].tolist(), key=f"respuesta_{qid}")
    if str(opciones.set_index("MC_Answer_norm").loc[respuesta, "es_correcta"]) == "True":
        st.badge("Respuesta correcta", icon=":material/check:", color="green")
    else:
        st.badge("Respuesta incorrecta", icon=":material/close:", color="orange")

    explicacion = st.text_area("Explicación del estudiante", key="explicacion", height=120, max_chars=1000,
                               placeholder="Por ejemplo, because 6 is a whole number and decimals are smaller")

    modo = st.segmented_control("Modelo", ["Un modelo", "Todos"], default="Un modelo", key="modo")
    if modo == "Todos":
        modelos = disponibles
    else:
        recomendado = disponibles.index(MODELO_RECOMENDADO) if MODELO_RECOMENDADO in disponibles else 0
        modelos = [st.selectbox("Modelo a usar", disponibles, index=recomendado, format_func=nombre_modelo,
                                label_visibility="collapsed")]
    faltan = [nombre_modelo(m) for m in MODELOS if m not in disponibles]
    if faltan:
        st.caption(f"Sin pesos descargados: {', '.join(faltan)}.")

    clasificar = st.button("Clasificar", type="primary", icon=":material/psychology:", width="stretch")

if clasificar:
    if not explicacion.strip():
        st.session_state.pop("resultado", None)
        col_entrada.warning("Escribe la explicación del estudiante antes de clasificar.")
    else:
        resultados = {}
        with col_resultado:
            for m in modelos:
                try:
                    with st.spinner(f"Cargando {nombre_modelo(m)}. La primera vez puede tardar unos segundos."):
                        cargar_modelo(m)
                    inicio = time.perf_counter()
                    top = prediccion.predecir(m, qid, respuesta, explicacion)
                    resultados[m] = (top, 1000 * (time.perf_counter() - inicio))
                except FileNotFoundError:
                    st.error(f"Faltan los pesos de {nombre_modelo(m)}. Revisa el README.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"{nombre_modelo(m)} no pudo clasificar la respuesta ({e}).")
        st.session_state["resultado"] = {"entrada": (qid, respuesta, explicacion), "resultados": resultados}

with col_resultado:
    resultado = st.session_state.get("resultado")
    if not resultado or not resultado["resultados"]:
        st.info("Completa los datos de la izquierda y presiona Clasificar. También puedes cargar un ejemplo.",
                icon=":material/arrow_back:")
        st.stop()

    if resultado["entrada"] != (qid, respuesta, explicacion):
        st.caption(":material/info: Cambiaste los datos de entrada. Presiona Clasificar para actualizar el resultado.")
    texto = resultado["entrada"][2].strip()
    if len(texto) <= UMBRAL_CORTA or len(texto.split()) < 3:
        st.warning("La explicación es muy corta. Los modelos tienen poca información y la predicción es menos fiable.")

    res = resultado["resultados"]
    if len(res) == 1:
        modelo, (top, ms) = next(iter(res.items()))
        tarjeta(modelo, top[0], ms)
        tabla = pd.DataFrame(top)
        fig = px.bar(tabla[::-1], x="prob", y="etiqueta", orientation="h", color="category",
                     color_discrete_map=pal.COLORES_CATEGORIA, text=tabla["prob"][::-1].map("{:.0%}".format),
                     labels={"prob": "Probabilidad", "etiqueta": ""}, title="Las 3 etiquetas más probables",
                     height=280)
        fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.08], tickformat=".0%"))
        fig.update_traces(textposition="outside", cliponaxis=False)
        mostrar_figura(fig)
    else:
        top1 = {m: top[0]["etiqueta"] for m, (top, _) in res.items()}
        if len(set(top1.values())) == 1:
            st.success(f"Los {len(res)} modelos coinciden en la etiqueta más probable.", icon=":material/check_circle:")
        else:
            st.warning("Los modelos no coinciden. Revisa las probabilidades para ver qué tan seguros están.",
                       icon=":material/compare_arrows:")
        for col, (m, (top, ms)) in zip(st.columns(len(res)), res.items()):
            with col:
                tarjeta(m, top[0], ms, compacta=True)

        filas = [{"Modelo": nombre_modelo(m), "Etiqueta": p["etiqueta"], "prob": p["prob"]}
                 for m, (top, _) in res.items() for p in top]
        tabla = pd.DataFrame(filas)
        orden = tabla.groupby("Etiqueta")["prob"].max().sort_values(ascending=False).index.tolist()
        fig = px.bar(tabla, y="Etiqueta", x="prob", color="Modelo", barmode="group", orientation="h",
                     color_discrete_map={nombre_modelo(m): pal.color_modelo(m) for m in res},
                     category_orders={"Etiqueta": orden, "Modelo": [nombre_modelo(m) for m in res][::-1]},
                     labels={"prob": "Probabilidad", "Etiqueta": ""}, title="Top 3 de cada modelo",
                     height=150 + 70 * len(orden))
        # Plotly dibuja de abajo hacia arriba, así que se invierte el orden para que M1 quede arriba
        fig.update_layout(xaxis=dict(range=[0, 1.02], dtick=0.25, tickformat=".0%"), legend_traceorder="reversed")
        fig.update_traces(hovertemplate="%{y}<br>%{x:.1%}<extra></extra>")
        mostrar_figura(fig)

        with st.expander("Qué significan las etiquetas predichas", expanded=True):
            for etiqueta in dict.fromkeys(top1.values()):
                cat, misc = etiqueta.split(":", 1)
                texto = f"`{cat}` {DESCRIPCION_CATEGORIA.get(cat, '')}"
                if misc != "NA":
                    texto += f" `{misc}` {descripciones.get(misc, '')}"
                st.markdown(texto)
        st.caption("Una etiqueta sin barra para un modelo significa que no quedó en su top 3. El tiempo de cada "
                   "tarjeta es el de la predicción, sin contar la carga inicial del modelo.")
