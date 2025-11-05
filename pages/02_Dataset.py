import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.title("! Exploración nuestros datos !")

DATA_PATH = Path("data/processed/games_final_csv.csv")

@st.cache_data
def load_df():
    if not DATA_PATH.exists():
        st.error(f"No se encontró el dataset: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_df()
st.caption(f"{len(df):,} filas × {len(df.columns)} columnas")
st.dataframe(df.head(20), use_container_width=True)


# ---------- MATRIZ DE CONFUSIÓN ----------
st.markdown("---")
st.subheader("📊 Rendimiento del Modelo (Validación)")

# Datos manuales de la matriz de confusión
data = pd.DataFrame({
    "Real": ["Derrota", "Derrota", "Victoria", "Victoria"],
    "Predicción": ["Derrota", "Victoria", "Derrota", "Victoria"],
    "Cantidad": [77, 36, 42, 93]
})

# --- Crear gráfico Altair ---
chart = (
    alt.Chart(data)
    .mark_rect()
    .encode(
        x=alt.X("Predicción:N", title="Predicción del modelo"),
        y=alt.Y("Real:N", title="Resultado real"),
        color=alt.Color("Cantidad:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Real", "Predicción", "Cantidad"]
    )
    .properties(
        width=400,
        height=400,
        title="Matriz de Confusión – Modelo LogReg"
    )
)

# --- Agregar texto encima de las celdas ---
text = (
    alt.Chart(data)
    .mark_text(baseline="middle", fontSize=16)
    .encode(
        x="Predicción:N",
        y="Real:N",
        text="Cantidad:Q"
    )
)

col1, col2 = st.columns([1, 1.2])
with col1:
    st.altair_chart(chart + text, use_container_width=True)
with col2:
    st.metric("Accuracy", "68.5 %")
    st.metric("ROC-AUC", "0.7366")
    st.metric("F1 (Test)", "0.7045")
    st.markdown("""
    **Interpretación rápida:**
    - La diagonal principal son aciertos (predicciones correctas).  
    - Los valores fuera de la diagonal son errores.  
    - El modelo acierta el **68 %** de los partidos, con un **F1 ≈ 0.70**.  
    """)
