import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NBA Predictions", page_icon="🏀", layout="wide")

st.title("🏀 Predicción de Partidos NBA")
st.markdown("""
**Proyecto Integrador – Visualización e Integración (Entrega 4)**

Esta app muestra:
1) **Exploración de datos** (gráficos interactivos)  
2) **Modelo y Predicción** (cargar valores y obtener la predicción del modelo entrenado)

> La app usa datasets **ya procesados** y un **pipeline entrenado** exportado desde la Entrega 3.
""")

# Estado rápido de artefactos (opcional, útil para deploy)
data_ok  = Path("data/processed/games_final_csv.csv").exists()
model_ok = Path("models/model_pipeline.pkl").exists()
meta_ok  = Path("models/metadata.json").exists()

st.subheader("Estado de artefactos")
cols = st.columns(3)
cols[0].write("**data/processed/games_final_csv.csv**")
cols[0].success("OK") if data_ok else cols[0].error("Falta")

cols[1].write("**models/model_pipeline.pkl**")
cols[1].success("OK") if model_ok else cols[1].error("Falta")

cols[2].write("**models/metadata.json**")
cols[2].success("OK") if meta_ok else cols[2].error("Falta")

st.markdown("---")
st.write("Usá el menú lateral para ir a **Modelo y Predicción**.")
