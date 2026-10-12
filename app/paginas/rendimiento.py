import pandas as pd
import plotly.express as px
import streamlit as st

from comun import (MODELO_RECOMENDADO, MODELOS, ORDEN_CATEGORIAS, cargar_metricas, cargar_reglas, mostrar_figura,
                   nombre_modelo, pal)

TODOS = ["linea_base", *MODELOS]
METRICAS = {"f1_macro": "Macro-F1", "accuracy": "Exactitud", "map_at_3": "MAP@3"}
NIVELES = {
    "correccion": "Corrección (True/False)",
    "category": "Categoría (6 clases)",
    "misconception": "Idea errónea (34 clases)",
    "etiqueta": "Etiqueta conjunta (64 clases)",
}

tablas = cargar_metricas()
resumen = tablas["resumen"].set_index("modelo")
preguntas = cargar_reglas()[1].set_index("QuestionId")["QuestionText_norm"]

st.title("Rendimiento de modelos")
st.markdown(
    "Resultados sobre el conjunto de prueba, 7,340 respuestas que ningún modelo usó para entrenar ni para ajustar "
    "hiperparámetros. La línea base solo usa la pregunta y la respuesta elegida, sin leer la explicación."
)

col1, col2 = st.columns([1, 3], vertical_alignment="bottom")
mostrar = col1.toggle("Mostrar rendimiento", value=True)
seleccion = col2.pills("Modelos a comparar", TODOS, selection_mode="multi", default=TODOS, format_func=nombre_modelo)

if not mostrar:
    st.info("La información de rendimiento está oculta. Activa el interruptor para verla.", icon=":material/visibility_off:")
    st.stop()
if not seleccion:
    st.warning("Selecciona al menos un modelo.")
    st.stop()

modelos = [m for m in TODOS if m in seleccion]
nombres = [nombre_modelo(m) for m in modelos]
colores = {nombre_modelo(m): pal.color_modelo(m) for m in modelos}

tabs = st.tabs(["Resumen", "Por nivel", "Por clase", "Matriz de confusión", "Por pregunta", "Costo y tiempo"])

with tabs[0]:
    cols = st.columns(len(modelos))
    for col, m in zip(cols, modelos):
        delta = None if m == "linea_base" else f"{resumen.loc[m, 'f1_macro'] - resumen.loc['linea_base', 'f1_macro']:+.3f}"
        col.metric(nombre_modelo(m), f"{resumen.loc[m, 'f1_macro']:.3f}", delta, border=True,
                   help="Macro-F1 sobre la etiqueta conjunta. La variación es contra la línea base.")

    largo = resumen.loc[modelos, list(METRICAS)].rename(columns=METRICAS).reset_index().melt(
        id_vars="modelo", var_name="Métrica", value_name="valor")
    largo["Modelo"] = largo["modelo"].map(nombre_modelo)
    fig = px.bar(largo, x="Métrica", y="valor", color="Modelo", barmode="group", color_discrete_map=colores,
                 category_orders={"Modelo": nombres}, text=largo["valor"].map("{:.3f}".format),
                 labels={"valor": "Valor", "Métrica": ""}, title="Métricas globales por modelo")
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=11)
    fig.update_layout(yaxis_range=[0, 1.05])
    mostrar_figura(fig)
    st.markdown(
        "Leer la explicación casi duplica el macro-F1 de la línea base (0.30 contra 0.53 a 0.56). En MAP@3 la mejora "
        "es menor porque la línea base ya ordena bien las etiquetas frecuentes de cada pregunta. Entre modelos las "
        "diferencias son pequeñas frente a la distancia con la línea base."
    )

    with st.container(border=True):
        st.markdown(f"##### Modelo final recomendado, {nombre_modelo(MODELO_RECOMENDADO)}")
        st.markdown(
            "Empata con M3 en macro-F1 y exactitud, tiene el mejor MAP@3 y clasifica unas 500 veces más rápido en "
            "CPU. M3 conviene cuando importa más acertar la idea errónea concreta o el recall de las clases raras."
        )

    columnas = {"f1_macro": "Macro-F1", "precision_macro": "Precisión macro", "recall_macro": "Recall macro",
                "f1_weighted": "F1 ponderado", "accuracy": "Exactitud", "map_at_3": "MAP@3"}
    tabla = resumen.loc[modelos, list(columnas)].rename(columns=columnas, index=nombre_modelo)
    tabla.index.name = "Modelo"
    st.dataframe(tabla.style.format("{:.3f}").highlight_max(color="#D6E9F5"), width="stretch")
    st.caption("En azul, el mejor valor de cada métrica entre los modelos seleccionados.")

with tabs[1]:
    metrica = st.segmented_control("Métrica", ["f1_macro", "accuracy"], default="f1_macro",
                                   format_func=METRICAS.get, key="metrica_nivel") or "f1_macro"
    filas = []
    for m in modelos:
        for nivel, etiqueta in NIVELES.items():
            col = metrica if nivel == "etiqueta" else f"{metrica}_{nivel}"
            filas.append({"Modelo": nombre_modelo(m), "Nivel": etiqueta, "valor": resumen.loc[m, col]})
    tabla = pd.DataFrame(filas)
    fig = px.bar(tabla, x="Nivel", y="valor", color="Modelo", barmode="group", color_discrete_map=colores,
                 category_orders={"Modelo": nombres, "Nivel": list(NIVELES.values())},
                 text=tabla["valor"].map("{:.2f}".format), labels={"valor": METRICAS[metrica], "Nivel": ""},
                 title=f"{METRICAS[metrica]} en cada nivel de la etiqueta")
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=11)
    fig.update_layout(yaxis_range=[0, 1.08])
    mostrar_figura(fig)
    st.markdown(
        "La corrección es perfecta en todos, porque el par pregunta y respuesta la determina y todos lo usan. "
        "La dificultad está en el tipo de explicación y en la idea errónea. Cuando la respuesta tiene una idea "
        "errónea, M3 identifica cuál con 91 % de exactitud, contra 86 % de M1 y M2."
    )

with tabs[2]:
    col1, col2 = st.columns([2, 1])
    nivel = col1.segmented_control("Nivel", ["category", "misconception", "etiqueta"], default="category",
                                   format_func=NIVELES.get, key="nivel_clase") or "category"
    minimo = col2.slider("Mínimo de ejemplos en prueba", 1, 100, 1,
                         help="Oculta las clases con pocos ejemplos, donde el F1 es inestable.")
    por_clase = tablas["por_clase"]
    sub = por_clase[(por_clase["nivel"] == nivel) & por_clase["modelo"].isin(modelos) & (por_clase["soporte"] >= minimo)]
    if sub.empty:
        st.info("Ninguna clase cumple el mínimo de ejemplos.")
    else:
        sub = sub.assign(Modelo=sub["modelo"].map(nombre_modelo), Clase=sub["clase"] + " (" + sub["soporte"].astype(str) + ")")
        orden = sub.drop_duplicates("clase").sort_values("soporte", ascending=False)["Clase"].tolist()
        f1 = sub.pivot(index="Clase", columns="Modelo", values="f1").reindex(index=orden, columns=nombres)
        detalle = sub.pivot(index="Clase", columns="Modelo", values=["precision", "recall"]).reindex(index=orden)
        hover = [[f"Precisión {detalle.loc[c, ('precision', n)]:.2f}<br>Recall {detalle.loc[c, ('recall', n)]:.2f}"
                  for n in f1.columns] for c in f1.index]
        fig = px.imshow(f1, text_auto=".2f", aspect="auto", zmin=0, zmax=1,
                        color_continuous_scale=pal.MAPA_RENDIMIENTO, labels=dict(x="", y="", color="F1"),
                        title=f"F1 por clase, {NIVELES[nivel].lower()}", height=max(320, 26 * len(orden) + 140))
        fig.update_traces(customdata=hover, hovertemplate="%{y}<br>%{x}<br>F1 %{z:.2f}<br>%{customdata}<extra></extra>")
        fig.update_xaxes(side="top")
        mostrar_figura(fig)
        st.caption("Entre paréntesis, el número de ejemplos de la clase en el conjunto de prueba.")
        st.markdown(
            "En las clases grandes los modelos rinden parecido. En las minoritarias no hay un ganador único. M3 es "
            "el mejor en `True_Misconception` gracias a su pérdida ponderada y M1 en `False_Correct`. Los F1 de 0 "
            "aparecen en clases con muy pocos ejemplos, que son las que más bajan el macro-F1."
        )

with tabs[3]:
    col1, col2 = st.columns([2, 1], vertical_alignment="bottom")
    opciones = [m for m in modelos]
    modelo = col1.selectbox("Modelo", opciones, format_func=nombre_modelo,
                            index=opciones.index(MODELO_RECOMENDADO) if MODELO_RECOMENDADO in opciones else 0)
    normalizar = col2.toggle("Proporción por fila", value=True, help="Cada fila suma 100 %. La diagonal es el recall.")
    conf = tablas["confusion_category"]
    conf = conf[conf["modelo"] == modelo]
    valor = "proporcion" if normalizar else "n"
    matriz = conf.pivot(index="real", columns="predicha", values=valor).reindex(index=ORDEN_CATEGORIAS,
                                                                              columns=ORDEN_CATEGORIAS)
    fig = px.imshow(matriz, text_auto=".0%" if normalizar else ",d", color_continuous_scale=pal.MAPA_SECUENCIAL,
                    labels=dict(x="Categoría predicha", y="Categoría real", color="Proporción" if normalizar else "n"),
                    title=f"Matriz de confusión de la categoría, {nombre_modelo(modelo)}", height=520)
    fig.update_layout(coloraxis_showscale=False)
    mostrar_figura(fig)

    fuera = conf[conf["real"] != conf["predicha"]].sort_values("proporcion", ascending=False).iloc[0]
    st.markdown(
        f"Nunca se confunde una respuesta correcta con una incorrecta, eso lo garantiza la máscara. La confusión "
        f"más frecuente de este modelo es predecir `{fuera['predicha']}` cuando la real es `{fuera['real']}` "
        f"({fuera['proporcion']:.0%} de esa clase). La frontera entre una explicación correcta y una que no aporta "
        "(`Neither`) depende del criterio del anotador y limita a cualquier modelo."
    )

with tabs[4]:
    metrica = st.segmented_control("Métrica", list(METRICAS), default="f1_macro", format_func=METRICAS.get,
                                   key="metrica_pregunta") or "f1_macro"
    por_preg = tablas["por_pregunta"]
    por_preg = por_preg[por_preg["modelo"].isin(modelos)]
    tabla = por_preg.pivot(index="QuestionId", columns="modelo", values=metrica)[modelos]
    tabla = tabla.loc[tabla.mean(axis=1).sort_values(ascending=False).index]
    tabla.index = tabla.index.astype(str)
    tabla.columns = nombres
    textos = [[preguntas.get(int(q), "")[:70] for _ in tabla.columns] for q in tabla.index]
    fig = px.imshow(tabla, text_auto=".2f", aspect="auto", zmin=0, zmax=1, color_continuous_scale=pal.MAPA_RENDIMIENTO,
                    labels=dict(x="", y="Pregunta", color=METRICAS[metrica]),
                    title=f"{METRICAS[metrica]} por pregunta, de la más fácil a la más difícil", height=620)
    fig.update_traces(customdata=textos,
                      hovertemplate="Pregunta %{y}<br>%{customdata}<br>%{x}<br>%{z:.3f}<extra></extra>")
    fig.update_xaxes(side="top")
    mostrar_figura(fig)
    st.markdown(
        f"La dificultad depende más de la pregunta que del modelo. En promedio de los modelos seleccionados, la "
        f"pregunta {tabla.index[0]} es la más fácil y la {tabla.index[-1]} la más difícil. Ningún modelo gana en todas, lo que "
        "sugiere que combinar sus probabilidades podría mejorar el resultado."
    )

with tabs[5]:
    tiempos = resumen.loc[modelos].reset_index()
    tiempos["Modelo"] = tiempos["modelo"].map(nombre_modelo)
    # Tamaño mínimo para que la línea base, que no se entrena, siga visible
    tiempos["tamano"] = tiempos["tiempo_entrenamiento_s"].clip(lower=40)
    fig = px.scatter(tiempos, x="tiempo_inferencia_test_s", y="f1_macro", size="tamano",
                     color="Modelo", color_discrete_map=colores, text="Modelo", log_x=True, size_max=60,
                     category_orders={"Modelo": nombres}, custom_data=["tiempo_entrenamiento_s", "hardware"],
                     labels={"tiempo_inferencia_test_s": "Tiempo de inferencia en prueba (s, escala log)",
                             "f1_macro": "Macro-F1"},
                     title="Rendimiento contra costo de cómputo")
    fig.update_traces(textposition="top center", hovertemplate=(
        "%{text}<br>Macro-F1 %{y:.3f}<br>Inferencia %{x:,.1f} s<br>Entrenamiento %{customdata[0]:,.0f} s"
        "<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(showlegend=False, yaxis_range=[0.2, 0.7])
    mostrar_figura(fig)
    st.caption("El área de cada punto es proporcional al tiempo de entrenamiento, con un tamaño mínimo para que la "
               "línea base sea visible.")

    tabla = tiempos.set_index("Modelo")[["tiempo_entrenamiento_s", "tiempo_inferencia_test_s", "ms_por_respuesta",
                                         "hardware"]]
    st.dataframe(tabla, width="stretch", column_config={
        "tiempo_entrenamiento_s": st.column_config.NumberColumn("Entrenamiento (s)", format="%.1f"),
        "tiempo_inferencia_test_s": st.column_config.NumberColumn("Inferencia en prueba (s)", format="%.1f"),
        "ms_por_respuesta": st.column_config.NumberColumn("ms por respuesta", format="%.2f"),
        "hardware": "Hardware",
    })
    st.markdown(
        "M1 domina la relación entre costo y beneficio. Logra el mismo macro-F1 que M3 con unas 500 veces menos "
        "tiempo de inferencia y sin GPU para entrenar. M2 cuesta más que M1 y rinde menos que ambos."
    )
