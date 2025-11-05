import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# NO usar st.set_page_config aquí (solo en app.py)
st.title("🤖 Explore nuestro Dataset!")

DATA_PATH = Path("data/processed/games_final_csv.csv")
MODEL_PATH = Path("models/model_pipeline.pkl")
META_PATH  = Path("models/metadata.json")

# ---------- Loaders con manejo de errores ----------
@st.cache_data
def load_df():
    if not DATA_PATH.exists():
        st.error(f"No se encontró el dataset: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Coaccionar numéricos cuando sea posible (sin romper strings categóricos)
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el modelo: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if not META_PATH.exists():
        st.error(f"No se encontró el metadata: {META_PATH}")
        st.stop()
    return json.loads(META_PATH.read_text(encoding="utf-8"))

# ---------- UI ----------
st.subheader("Dataset de Partidos NBA")
df = load_df()
st.caption(f"{len(df):,} filas × {len(df.columns)} columnas")
st.dataframe(df.head(20), use_container_width=True)

meta = load_meta()
features = meta.get("features", [])
missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"Estas features del modelo no están en el CSV: {missing}")
    st.stop()

model = load_model()

st.subheader("Ingresá datos y obtené una predicción")
with st.form("pred"):
    cols = st.columns(3)
    row = {}
    for i, f in enumerate(features):
        col = cols[i % 3]
        if pd.api.types.is_numeric_dtype(df[f]):
            default_val = float(df[f].median()) if pd.notna(df[f].median()) else 0.0
            row[f] = col.number_input(f, value=default_val)
        else:
            opciones = sorted([x for x in df[f].dropna().unique().tolist() if str(x) != ""])
            row[f] = col.selectbox(f, opciones) if opciones else col.text_input(f, value="")
    ok = st.form_submit_button("Predecir")

if ok:
    X = pd.DataFrame([row])
    try:
        y = model.predict(X)[0]
        st.success(f"Predicción: {y}")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            try:
                clases = list(getattr(model, "classes_", []))
                st.write({str(k): float(v) for k, v in zip(clases, proba)})
            except Exception:
                st.write([float(v) for v in proba])
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.exception(e)
