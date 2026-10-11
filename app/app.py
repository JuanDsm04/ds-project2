"""Punto de entrada de la aplicación: streamlit run app/app.py"""
import streamlit as st

import comun  # noqa: F401  (configura sys.path y la plantilla de Plotly)

st.set_page_config(page_title="Ideas erróneas en matemáticas", page_icon=":material/calculate:", layout="wide")

paginas = [
    st.Page("paginas/inicio.py", title="Inicio", icon=":material/home:", default=True),
]

st.navigation(paginas).run()
