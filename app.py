import streamlit as st
from pathlib import Path


st.set_page_config(page_title="NBA Predictions", page_icon="🏀", layout="wide")

st.title("🏀 Predicción de Partidos NBA")
st.markdown("""
**Proyecto Integrador – Visualización e Integración (Entrega 4)**

Esta app muestra:
1) **Explora datos e interactua con graficos** (gráficos interactivos)  
2) Análisis EDA, mira nuestras hipótesis de investigación
3) Explora nuestro Dataset e información del modelo de predicción.
4) **Haz tu propia predicción**
5) ¡Conoce nuestro **Equipo**!

> La app usa datasets **ya procesados** y un **pipeline entrenado** .
""")

