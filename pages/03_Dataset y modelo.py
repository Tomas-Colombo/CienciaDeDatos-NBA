import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.title("! Exploración nuestros datos !")

DATA_PATH = Path("data/graph/df_final.csv")

@st.cache_data
def load_df():
    if not DATA_PATH.exists():
        st.error(f"No se encontró el dataset: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_df()
st.caption(f"{len(df):,} filas × {len(df.columns)} columnas")
st.dataframe(df.head(20), use_container_width=True)

# ---------- DESCRIPCIONES DE COLUMNAS ----------
column_descriptions = {
    "game_id": "Identificador único del partido (ej: YYYYMMDD + código equipo).",
    "date": "Fecha y hora del partido (UTC).",
    "home_team": "Equipo local (código abreviado).",
    "visitor_team": "Equipo visitante (código abreviado).",
    "home_game_number": "Número de partido en la temporada para el equipo local.",
    "home_result": "Resultado del equipo local en ese partido: 'W' = victoria, 'L' = derrota.",
    "home_streak": "Racha actual del equipo local (número positivo para victorias, negativo para derrotas).",
    "home_home_streak": "Racha del equipo local jugando como local.",
    "home_away_streak": "Racha del equipo local jugando como visitante.",
    "home_offensive_rating": "Rating ofensivo estimado del equipo local (puntos por 100 posesiones).",
    "home_defensive_rating": "Rating defensivo estimado del equipo local (puntos permitidos por 100 posesiones).",
    "home_ts_percent": "True Shooting % del equipo local (medida de eficiencia en el tiro).",
    "home_assist_percent": "Porcentaje de asistencias del equipo local (AST%).",
    "home_steal_percent": "Porcentaje de robos del equipo local.",
    "home_rebound_percent": "Porcentaje de rebotes del equipo local.",
    "home_turnover_percent": "Porcentaje de pérdidas del equipo local.",
    "visitor_game_number": "Número de partido en la temporada para el equipo visitante.",
    "visitor_result": "Resultado del equipo visitante en ese partido: 'W' o 'L'.",
    "visitor_streak": "Racha actual del equipo visitante.",
    "visitor_home_streak": "Racha del equipo visitante como local.",
    "visitor_away_streak": "Racha del equipo visitante como visitante.",
    "visitor_offensive_rating": "Rating ofensivo estimado del equipo visitante.",
    "visitor_defensive_rating": "Rating defensivo estimado del equipo visitante.",
    "visitor_ts_percent": "True Shooting % del equipo visitante.",
    "visitor_assist_percent": "Porcentaje de asistencias del equipo visitante.",
    "visitor_steal_percent": "Porcentaje de robos del equipo visitante.",
    "visitor_rebound_percent": "Porcentaje de rebotes del equipo visitante.",
    "visitor_turnover_percent": "Porcentaje de pérdidas del equipo visitante.",
    "home_last_10": "Registro de los últimos 10 partidos del equipo local (formato X-Y).",
    "visitor_last_10": "Registro de los últimos 10 partidos del equipo visitante (formato X-Y).",
    "home_wins_percent": "Porcentaje de victorias del equipo local en la temporada.",
    "visitor_wins_percent": "Porcentaje de victorias del equipo visitante en la temporada.",
    "wins_percent_diff": "Diferencia entre porcentaje de victorias (home - visitor).",
    "offensive_rating_diff": "Diferencia de rating ofensivo (home - visitor).",
    "defensive_rating_diff": "Diferencia de rating defensivo (home - visitor).",
    "net_rating_diff": "Diferencia neta de rating (offensive - defensive) entre equipos.",
    "home_estimated_points": "Estimación de puntos del equipo local para el partido.",
    "visitor_estimated_points": "Estimación de puntos del equipo visitante para el partido.",
    "estimated_point_diff": "Diferencia estimada de puntos (home - visitor).",
    "ts_percent_diff": "Diferencia en True Shooting % entre equipos.",
    "turnover_percent_diff": "Diferencia en porcentaje de pérdidas entre equipos.",
    "assist_percent_diff": "Diferencia en porcentaje de asistencias entre equipos.",
    "steal_percent_diff": "Diferencia en porcentaje de robos entre equipos.",
    "rebound_percent_diff": "Diferencia en porcentaje de rebotes entre equipos.",
    "home_streak_extreme": "Indicador binario (1/0) de racha extrema del equipo local (mayor/menor o igual a 6).",
    "visitor_streak_extreme": "Indicador binario (1/0) de racha extrema del equipo visitante (mayor/menor o igual a 6).",
    "streak_extreme_diff": "Diferencia entre indicadores de racha extrema.",
    "streak_diff": "Diferencia simple de rachas (home - visitor).",
    "home_quality": "Etiqueta cualitativa de la calidad del equipo local según el procentaje de victorias (ej. 'Muy Fuerte').",
    "visitor_quality": "Etiqueta cualitativa de la calidad del equipo visitante según el procentaje de victorias.",
    "home_much_better": "Indicador binario (1/0) si el local es claramente mejor.",
    "visitor_much_better": "Indicador binario (1/0) si el visitante es claramente mejor.",
    "teams_evenly_matched": "Indicador binario (1/0) si los equipos están parejos.",
    "target": "Etiqueta objetivo del modelo (1 = victoria del equipo local, 0 = derrota)."
}

with st.expander("🛈 Descripción de columnas (clic para expandir)", expanded=False):
    desc_df = pd.DataFrame.from_dict(column_descriptions, orient="index", columns=["Descripción"])
    desc_df.index.name = "Columna"
    st.dataframe(desc_df, use_container_width=True)



# ---------- MATRICES DE CONFUSIÓN ----------
st.markdown("---")
st.subheader("📊 Rendimiento de los Modelos (Validación)")

st.info("""
❓ **¿Qué predice el modelo?**  
Resultado del equipo Local en el partido.
""")

# Datos de matrices de confusión para cada modelo
model_data = {
    "Logistic Regression": {
        "data": pd.DataFrame({
            "Real": ["Derrota", "Derrota", "Victoria", "Victoria"],
            "Predicción": ["Derrota", "Victoria", "Derrota", "Victoria"],
            "Cantidad": [79, 34, 42, 93]
        }),
        "metrics": {
            "Accuracy": "69.3 %",
            "ROC-AUC": "74.0 %",
            "F1 (Test)": "71.0 %"
        }
    },
    "XGBoost": {
        "data": pd.DataFrame({
            "Real": ["Derrota", "Derrota", "Victoria", "Victoria"],
            "Predicción": ["Derrota", "Victoria", "Derrota", "Victoria"],
            "Cantidad": [45, 68, 12, 123]
        }),
        "metrics": {
            "Accuracy": "67.7 %",
            "ROC-AUC": "74.7 %",
            "F1 (Test)": "75.5 %"
        }
    },
    "LightGBM": {
        "data": pd.DataFrame({
            "Real": ["Derrota", "Derrota", "Victoria", "Victoria"],
            "Predicción": ["Derrota", "Victoria", "Derrota", "Victoria"],
            "Cantidad": [30, 83, 8, 127]
        }),
        "metrics": {
            "Accuracy": "63.3 %",
            "ROC-AUC": "72.7 %",
            "F1 (Test)": "73.6 %"
        }
    }
}

# Crear tabs para cada modelo
tab1, tab2, tab3 = st.tabs(["Logistic Regression", "XGBoost", "LightGBM"])

# Mapeo de archivos de feature importance
feature_importance_files = {
    "Logistic Regression": "data/models_feature_importance/log_reg_feature_importances.csv",
    "XGBoost": "data/models_feature_importance/xgboost_feature_importances.csv",
    "LightGBM": "data/models_feature_importance/lgbm_feature_importances.csv"
}

@st.cache_data
def load_feature_importance(filepath):
    """Carga el CSV de feature importance."""
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return None

def show_model_performance(tab, model_name, model_info):
    with tab:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            # Crear gráfico Altair
            chart = (
                alt.Chart(model_info["data"])
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
                    title=f"Matriz de Confusión – {model_name}"
                )
            )

            # Agregar texto en las celdas
            text = (
                alt.Chart(model_info["data"])
                .mark_text(baseline="middle", fontSize=16)
                .encode(
                    x="Predicción:N",
                    y="Real:N",
                    text="Cantidad:Q"
                )
            )
            
            st.altair_chart(chart + text, use_container_width=True)
        
        with col2:
            metrics = model_info["metrics"]
            st.metric("Accuracy", metrics["Accuracy"])
            st.metric("ROC-AUC", metrics["ROC-AUC"])
            st.metric("F1 (Test)", metrics["F1 (Test)"])
            st.markdown(f"""
            **Interpretación rápida:**
            - La diagonal principal son aciertos (predicciones correctas).  
            - Los valores fuera de la diagonal son errores.  
            - El modelo acierta el **{metrics["Accuracy"]}** de los partidos, con un **F1 ≈ {metrics["F1 (Test)"]}**.  
            """)
        
        # Mostrar Feature Importance debajo
        st.markdown("---")
        st.subheader(f"🎯 Feature Importance – {model_name}")
        
        fi_filepath = feature_importance_files.get(model_name)
        if fi_filepath:
            fi_df = load_feature_importance(fi_filepath)
            if fi_df is not None:
                # Mostrar las top 15 features ordenadas por valor absoluto
                fi_df_sorted = fi_df.copy()
                fi_df_sorted['abs_importance'] = fi_df_sorted.iloc[:, 1].abs()
                fi_df_sorted = fi_df_sorted.sort_values('abs_importance', ascending=False).head(15)
                fi_df_sorted = fi_df_sorted.drop('abs_importance', axis=1)
                
                # Crear gráfico de barras
                chart_fi = (
                    alt.Chart(fi_df_sorted)
                    .mark_bar()
                    .encode(
                        x=alt.X(fi_df_sorted.columns[1], title="Importancia"),
                        y=alt.Y(fi_df_sorted.columns[0], title="Feature", sort='-x'),
                        color=alt.condition(
                            alt.datum[fi_df_sorted.columns[1]] > 0,
                            alt.value("#1f77b4"),  # azul para positivo
                            alt.value("#ff7f0e")   # naranja para negativo
                        )
                    )
                    .properties(height=400, width=600)
                )
                
                st.altair_chart(chart_fi, use_container_width=True)
                
                # Mostrar tabla completa en expander
                with st.expander("📋 Ver todas las features"):
                    st.dataframe(fi_df, use_container_width=True)
            else:
                st.warning(f"No se encontró el archivo de feature importance para {model_name}")
        else:
            st.warning(f"Archivo no configurado para {model_name}")

# Mostrar el rendimiento de cada modelo en su respectiva tab
show_model_performance(tab1, "Logistic Regression", model_data["Logistic Regression"])
show_model_performance(tab2, "XGBoost", model_data["XGBoost"])
show_model_performance(tab3, "LightGBM", model_data["LightGBM"])
