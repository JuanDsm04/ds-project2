import pandas as pd
import plotly.express as px
import streamlit as st

from comun import (ORDEN_CATEGORIAS, cargar_datos, cargar_reglas, etiquetas_preguntas, leyenda_derecha,
                   mostrar_figura, pal)

COLORES_CAT = pal.COLORES_CATEGORIA
ETIQUETAS = {
    "Category": "Categoría", "Misconception": "Idea errónea", "QuestionId": "Pregunta", "n": "Respuestas",
    "pct": "% de respuestas", "n_palabras": "Palabras en la explicación", "tipo": "Tipo de explicación",
    "MC_Answer_norm": "Respuesta elegida", "split": "Partición",
}

df_total = cargar_datos()
nombres_preg = etiquetas_preguntas()

st.title("Explorar datos")
st.markdown(
    "Conjunto de entrenamiento del reto después de la limpieza. Los filtros de la barra lateral se aplican a "
    "todas las pestañas."
)

with st.sidebar:
    st.header("Filtros")
    preguntas = st.multiselect("Preguntas", list(nombres_preg), format_func=nombres_preg.get,
                               placeholder="Todas las preguntas")
    categorias = st.multiselect("Categorías", ORDEN_CATEGORIAS, placeholder="Todas las categorías")
    particion = st.segmented_control("Partición", ["train", "val", "test"], selection_mode="multi",
                                     default=["train", "val", "test"])

df = df_total
if preguntas:
    df = df[df["QuestionId"].isin(preguntas)]
if categorias:
    df = df[df["Category"].isin(categorias)]
df = df[df["split"].isin(particion or [])]

if df.empty:
    st.warning("Ningún registro cumple los filtros. Prueba con otra combinación.")
    st.stop()

st.caption(f"{len(df):,} de {len(df_total):,} respuestas seleccionadas.")

tabs = st.tabs(["Variables", "Categorías", "Ideas erróneas", "Preguntas", "Longitud", "Ejemplos"])

with tabs[0]:
    st.subheader("Variables del conjunto limpio")
    variables = pd.DataFrame([
        ("QuestionId", "Categórica (15 valores)", "Pregunta de diagnóstico", "M1, M2, máscara"),
        ("QuestionText", "Texto", "Enunciado, con notación LaTeX normalizada", "M2, M3"),
        ("MC_Answer", "Categórica (4 por pregunta)", "Opción elegida por el estudiante", "M1, M2, M3, máscara"),
        ("StudentExplanation", "Texto libre", "Explicación escrita del estudiante", "M1, M2, M3"),
        ("short_explanation", "Binaria", "Explicación de 5 caracteres o menos", "M2"),
        ("Category", "Objetivo (6 clases)", "Corrección de la respuesta y tipo de explicación", "Etiqueta"),
        ("Misconception", "Objetivo (34 clases y NA)", "Idea errónea específica", "Etiqueta"),
    ], columns=["Variable", "Tipo", "Descripción", "Uso en los modelos"])
    st.dataframe(variables, hide_index=True, width="stretch")
    st.markdown(
        "Además se derivan variables a partir de estas. La corrección de la respuesta sale del par pregunta y "
        "respuesta, la longitud de la explicación la usa M2 y la máscara descarta las etiquetas incoherentes con "
        "la pregunta antes de elegir el top 3."
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        conteo = df["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).reset_index()
        conteo.columns = ["split", "n"]
        fig = px.bar(conteo, x="split", y="n", text="n", labels=ETIQUETAS,
                     color_discrete_sequence=[pal.ACENTO], title="Partición estratificada 70/10/20")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        mostrar_figura(fig)
    with col2:
        st.markdown("##### Muestra de registros")
        st.dataframe(df[["row_id", "QuestionId", "MC_Answer", "StudentExplanation", "Category", "Misconception"]]
                     .head(8), hide_index=True, width="stretch")

with tabs[1]:
    conteo = df["Category"].value_counts().reindex(ORDEN_CATEGORIAS).dropna().reset_index()
    conteo.columns = ["Category", "n"]
    conteo["pct"] = 100 * conteo["n"] / conteo["n"].sum()
    log = st.toggle("Escala logarítmica", help="Hace visibles las categorías con muy pocas respuestas.")
    fig = px.bar(conteo, x="Category", y="n", color="Category", color_discrete_map=COLORES_CAT,
                 text=conteo["pct"].map("{:.1f} %".format), labels=ETIQUETAS, log_y=log,
                 title="Respuestas por categoría", custom_data=["pct"])
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>%{y:,} respuestas (%{customdata[0]:.1f} %)<extra></extra>")
    fig.update_layout(showlegend=False)
    mostrar_figura(fig)
    st.markdown(
        "El desbalance es fuerte. `True_Correct` reúne cerca del 40 % de las respuestas, mientras que "
        "`True_Misconception` y `False_Correct` apenas llegan al 1 %. Por eso el macro-F1, que pesa igual a todas "
        "las clases, es la métrica principal del proyecto."
    )

with tabs[2]:
    misc = df[df["Misconception"] != "NA"]
    if misc.empty:
        st.info("La selección no tiene respuestas con idea errónea.")
    else:
        conteo = misc.groupby(["Misconception", "Category"]).size().reset_index(name="n")
        orden = conteo.groupby("Misconception")["n"].sum().sort_values().index.tolist()
        fig = px.bar(conteo, y="Misconception", x="n", color="Category", orientation="h",
                     color_discrete_map=COLORES_CAT, category_orders={"Misconception": orden},
                     labels=ETIQUETAS, title="Respuestas por idea errónea",
                     height=max(350, 22 * len(orden) + 120))
        mostrar_figura(fig)
        raras = (misc["Misconception"].value_counts() < 30).sum()
        pct_false = 100 * (misc["Category"] == "False_Misconception").mean()
        st.markdown(
            f"Hay {len(orden)} ideas erróneas en la selección y {raras} tienen menos de 30 ejemplos. El "
            f"{pct_false:.0f} % aparece con respuestas incorrectas, aunque algunas como `Tacking` o `Irrelevant` "
            "también surgen con respuestas correctas. Las clases con pocos ejemplos son las que más bajan el "
            "macro-F1 de los modelos."
        )

with tabs[3]:
    tabla = df.groupby(["QuestionId", "Category"]).size().reset_index(name="n")
    tabla["pct"] = 100 * tabla["n"] / tabla.groupby("QuestionId")["n"].transform("sum")
    tabla["Pregunta"] = tabla["QuestionId"].astype(str)
    correctas = tabla[tabla["Category"] == "True_Correct"].set_index("Pregunta")["pct"]
    orden = sorted(tabla["Pregunta"].unique(), key=lambda q: correctas.get(q, 0), reverse=True)
    fig = px.bar(tabla, y="Pregunta", x="pct", color="Category", orientation="h",
                 color_discrete_map=COLORES_CAT, labels=ETIQUETAS, custom_data=["n"],
                 category_orders={"Pregunta": orden, "Category": ORDEN_CATEGORIAS},
                 title="Composición de categorías por pregunta", height=max(320, 30 * len(orden) + 140))
    fig.update_traces(hovertemplate="Pregunta %{y}<br>%{x:.1f} % (%{customdata[0]:,} respuestas)<extra></extra>")
    fig.update_layout(xaxis_ticksuffix=" %")
    mostrar_figura(fig)
    rango = (f"La proporción de `True_Correct` va de {correctas.min():.0f} % a {correctas.max():.0f} % según la "
             "pregunta, y cada una tiene su propio perfil de errores. ") if len(correctas) else ""
    st.markdown(
        f"Las preguntas no son intercambiables. {rango}Por eso todos los modelos reciben la pregunta como variable."
    )

    st.subheader("Detalle de una pregunta")
    qid = st.selectbox("Pregunta", sorted(df["QuestionId"].unique()), format_func=nombres_preg.get)
    respuestas, preguntas_tab, misc_por_preg = cargar_reglas()
    st.info(preguntas_tab.set_index("QuestionId").loc[qid, "QuestionText_norm"])

    sub = df[df["QuestionId"] == qid]
    opciones = respuestas[respuestas["QuestionId"] == qid].set_index("MC_Answer_norm")["es_correcta"]
    sub = sub.assign(Respuesta=sub["MC_Answer_norm"].map(
        lambda a: f"{a} (correcta)" if str(opciones.get(a)) == "True" else a))
    orden_resp = sorted(sub["Respuesta"].unique())
    col1, col2 = st.columns([3, 2])
    with col1:
        conteo = sub.groupby(["Respuesta", "Category"]).size().reset_index(name="n")
        fig = px.bar(conteo, y="Respuesta", x="n", color="Category", orientation="h",
                     color_discrete_map=COLORES_CAT, labels=ETIQUETAS, title="Categorías según la respuesta elegida",
                     category_orders={"Category": ORDEN_CATEGORIAS, "Respuesta": orden_resp})
        mostrar_figura(leyenda_derecha(fig))
    with col2:
        misc = sub[sub["Misconception"] != "NA"]
        if misc.empty:
            st.info("La selección no tiene ideas erróneas para esta pregunta.")
        else:
            tabla = pd.crosstab(misc["Misconception"], misc["Respuesta"]).reindex(columns=orden_resp, fill_value=0)
            fig = px.imshow(tabla, text_auto=True, aspect="auto", color_continuous_scale="Oranges",
                            labels=dict(x="Respuesta elegida", y="Idea errónea", color="Respuestas"),
                            title="Ideas erróneas según la respuesta")
            fig.update_layout(coloraxis_showscale=False, margin_t=60)
            mostrar_figura(fig)
    st.markdown(
        f"Esta pregunta solo admite {len(misc_por_preg.get(str(qid), []))} ideas erróneas y cada una se concentra "
        "en una respuesta incorrecta concreta. La respuesta elegida casi determina la categoría, y la explicación "
        "decide entre `Correct`, `Neither` y `Misconception`."
    )

with tabs[4]:
    limite = st.slider("Máximo de palabras a mostrar", 10, int(df["n_palabras"].max()), 60,
                       help="Recorta la cola larga de explicaciones muy extensas.")
    sub = df[df["n_palabras"] <= limite]
    col1, col2 = st.columns(2)
    with col1:
        dist = sub.groupby(["tipo", "n_palabras"]).size().reset_index(name="n")
        dist["pct"] = 100 * dist["n"] / dist.groupby("tipo")["n"].transform("sum")
        fig = px.line(dist, x="n_palabras", y="pct", color="tipo", color_discrete_map=pal.COLORES_TIPO,
                      category_orders={"tipo": list(pal.COLORES_TIPO)}, labels=ETIQUETAS,
                      title="Longitud según el tipo de explicación")
        fig.update_traces(line_width=2.5)
        fig.update_layout(yaxis_title="% dentro de cada tipo", hovermode="x unified")
        mostrar_figura(fig)
    with col2:
        fig = px.box(sub, y="Category", x="n_palabras", color="Category", color_discrete_map=COLORES_CAT,
                     category_orders={"Category": ORDEN_CATEGORIAS}, points=False, labels=ETIQUETAS,
                     title="Palabras por categoría")
        fig.update_layout(showlegend=False)
        mostrar_figura(fig)
    medianas = df.groupby("tipo")["n_palabras"].median().sort_values()
    st.markdown(
        f"La mediana general es de {df['n_palabras'].median():.0f} palabras. Por tipo va de "
        f"{medianas.iloc[0]:.0f} (`{medianas.index[0]}`) a {medianas.iloc[-1]:.0f} (`{medianas.index[-1]}`), pero "
        "las distribuciones se superponen mucho. La longitud sola no basta para clasificar, lo que importa es el "
        "contenido del texto."
    )

with tabs[5]:
    col1, col2 = st.columns([3, 1])
    busqueda = col1.text_input("Buscar en las explicaciones", placeholder="Por ejemplo, divide")
    n = col2.number_input("Filas", 5, 200, 20, step=5)
    sub = df
    if busqueda:
        sub = sub[sub["StudentExplanation"].str.contains(busqueda, case=False, regex=False)]
    if st.button("Otra muestra", icon=":material/shuffle:"):
        st.session_state["semilla_ejemplos"] = st.session_state.get("semilla_ejemplos", 0) + 1
    muestra = sub.sample(min(int(n), len(sub)), random_state=st.session_state.get("semilla_ejemplos", 0))
    st.caption(f"{len(sub):,} coincidencias.")
    st.dataframe(
        muestra[["QuestionId", "MC_Answer_norm", "StudentExplanation", "Category", "Misconception"]],
        hide_index=True, width="stretch",
        column_config={
            "QuestionId": st.column_config.NumberColumn("Pregunta", format="%d"),
            "MC_Answer_norm": "Respuesta",
            "StudentExplanation": st.column_config.TextColumn("Explicación", width="large"),
            "Category": "Categoría",
            "Misconception": "Idea errónea",
        },
    )
