import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Análisis EDA", page_icon="📊")
st.title("📊 Análisis Exploratorio de Datos (EDA) – NBA Games")

# --- Cargar datos ---
DATA_PATH = Path("data/graph/df_final.csv")

@st.cache_data
def load_df():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_df()

# ==============================================================  
# 🧮 Calcular columnas derivadas necesarias  
# ==============================================================  

df['target'] = df['home_result'].apply(lambda x: 1 if x == 'W' else 0)
df['net_rating_diff'] = df['home_offensive_rating'] - df['visitor_defensive_rating']

# --- Parseo de rachas ---
def parse_streak(value):
    if isinstance(value, str):
        if value.startswith('W'):
            return int(value[1:])
        elif value.startswith('L'):
            return -int(value[1:])
    return 0

df['home_streak_num'] = df['home_streak'].apply(parse_streak)
df['visitor_streak_num'] = df['visitor_streak'].apply(parse_streak)
df['streak_extreme_diff'] = df['home_streak_num'] - df['visitor_streak_num']

df['turnover_percent_diff'] = df['home_turnover_percent'] - df['visitor_turnover_percent']

df['home_wins_percent'] = df.groupby('home_team')['target'].transform('mean')
df['visitor_wins_percent'] = 1 - df.groupby('visitor_team')['target'].transform('mean')

# ==============================================================
# 🧠 Hipótesis Evaluadas
# ==============================================================

st.markdown("""
### 🧠 Hipótesis Evaluadas
A continuación se resumen los resultados más importantes del análisis exploratorio de datos.
""")

# ==============================================================  
# 🏠 H1: Ventaja de Localía  
# ==============================================================  

with st.expander("**H1: Ventaja de Localía – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** Los equipos locales tienen mayor probabilidad de ganar.  
    **Evidencia:**
    - ~54% de victorias locales  
    - Ratings ofensivos más consistentes en casa  
    - True Shooting % superior en casa  

    **Conclusión:** La ventaja de localía existe (~+4pp), aunque no es el factor más determinante.
    """)

    df['is_home_win'] = df['home_result'].apply(lambda x: 1 if x == 'W' else 0)
    win_rate = df['is_home_win'].mean() * 100

    chart = (
        alt.Chart(pd.DataFrame({'Condición': ['Local', 'Visitante'],
                                'Porcentaje de Victorias': [win_rate, 100 - win_rate]}))
        .mark_bar(size=80)
        .encode(
            x=alt.X('Condición:N', title=None),
            y=alt.Y('Porcentaje de Victorias:Q', title='Porcentaje de Victorias (%)'),
            color=alt.Color('Condición:N', scale=alt.Scale(scheme='tableau10')),
            tooltip=['Condición', 'Porcentaje de Victorias']
        )
        .properties(title="Probabilidad de Victoria según Condición")
    )
    st.altair_chart(chart, use_container_width=True)

# ==============================================================  
# 📊 H2: Diferencial Ofensivo-Defensivo  
# ==============================================================  

with st.expander("**H2: Diferencial Ofensivo-Defensivo como Predictor Principal – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** El diferencial entre el *rating ofensivo* del equipo y el *rating defensivo* del rival es el predictor más fuerte.  
    """)

    box = (
        alt.Chart(df)
        .mark_boxplot(size=50)
        .encode(
            x=alt.X('target:N', title='Resultado (0=Derrota, 1=Victoria)'),
            y=alt.Y('net_rating_diff:Q', title='Diferencial Neto (Rating Ofensivo - Rating Defensivo Rival)'),
            color='target:N'
        )
        .properties(title="Distribución del Diferencial Ofensivo-Defensivo")
    )
    st.altair_chart(box, use_container_width=True)

# ==============================================================  
# 🔁 H3: Efecto Acumulativo de Rachas  
# ==============================================================  

with st.expander("**H3: Efecto Acumulativo de Rachas – ⚠️ Parcialmente Confirmada**"):
    st.markdown("""
    **Hipótesis:** Solo las rachas largas (≥6) tienen efecto significativo.  
    """)

    streak_df = (
        df.groupby('streak_extreme_diff', as_index=False)
        .agg({'target': 'mean'})
        .rename(columns={'target': 'Porcentaje de Victorias'})
    )

    if streak_df['Porcentaje de Victorias'].nunique() > 1:
        streak_plot = (
            alt.Chart(streak_df)
            .mark_bar(size=60)
            .encode(
                x=alt.X('streak_extreme_diff:O', title='Diferencia de Racha (Local - Visitante)'),
                y=alt.Y('Porcentaje de Victorias:Q', title='Probabilidad de Victoria', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('streak_extreme_diff:O', scale=alt.Scale(scheme='plasma')),
                tooltip=['streak_extreme_diff', 'Porcentaje de Victorias']
            )
            .properties(title="Efecto de las Rachas en el Resultado")
        )
        st.altair_chart(streak_plot, use_container_width=True)
    else:
        st.info("ℹ️ No se encontraron variaciones significativas en las rachas, gráfico omitido.")

# ==============================================================  
# 📈 H4: Porcentaje de Victorias vs Momentum  
# ==============================================================  

with st.expander("**H4: El Porcentaje de Victorias es Más Relevante que el Momentum – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** El historial completo de temporada predice mejor que el rendimiento reciente.  
    """)

    chart = (
        alt.Chart(df)
        .mark_circle(size=70, opacity=0.6)
        .encode(
            x=alt.X('visitor_wins_percent:Q', title='Porcentaje de Victorias del Visitante'),
            y=alt.Y('home_wins_percent:Q', title='Porcentaje de Victorias del Local'),
            color=alt.Color('target:N', title='Resultado', scale=alt.Scale(scheme='category10')),
            tooltip=['home_team', 'visitor_team', 'home_wins_percent', 'visitor_wins_percent', 'target']
        )
        .properties(title="Relación entre Porcentaje de Victorias Local y Visitante")
    )
    st.altair_chart(chart, use_container_width=True)

# ==============================================================  
# 🔒 H5: Disciplina Ofensiva  
# ==============================================================  

with st.expander("**H5: La Disciplina Ofensiva es Crítica – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** Los equipos con menor porcentaje de pérdidas (*turnovers*) tienen más chances de ganar.  
    """)

    scatter = (
        alt.Chart(df)
        .mark_circle(size=70, opacity=0.6)
        .encode(
            x=alt.X('turnover_percent_diff:Q', title='Diferencial de Pérdidas (Local - Visitante)'),
            y=alt.Y('target:Q', title='Resultado (1=Victoria)'),
            color=alt.Color('target:N', title='Resultado', scale=alt.Scale(scheme='set1')),
            tooltip=['home_team', 'visitor_team', 'turnover_percent_diff', 'target']
        )
        .properties(title="Impacto del Diferencial de Pérdidas en la Victoria")
    )
    st.altair_chart(scatter, use_container_width=True)

# ==============================================================  
# 📊 Ranking de Correlaciones  
# ==============================================================  

st.markdown("---")
st.header("📈 Ranking de Variables Más Correlacionadas con la Victoria")

num_df = df.select_dtypes(include=['number']).dropna(axis=1, how='all')
corr = num_df.corr(numeric_only=True)['target'].sort_values(ascending=False)
corr_df = corr.reset_index()
corr_df.columns = ['Variable', 'Correlación con la Victoria']
corr_df['Abs'] = corr_df['Correlación con la Victoria'].abs()
top_corr = corr_df.sort_values('Abs', ascending=False).head(15)

# Traducción de nombres de variables (simplificada)
trad = {
    'net_rating_diff': 'Diferencial Neto',
    'turnover_percent_diff': 'Diferencial de Pérdidas',
    'home_wins_percent': 'Win% Local',
    'visitor_wins_percent': 'Win% Visitante',
    'home_offensive_rating': 'Rating Ofensivo Local',
    'visitor_defensive_rating': 'Rating Defensivo Visitante',
    'home_streak_num': 'Racha Local',
    'visitor_streak_num': 'Racha Visitante',
}
top_corr['Variable'] = top_corr['Variable'].replace(trad)

# Mostrar tabla y debajo gráfico
st.dataframe(top_corr[['Variable', 'Correlación con la Victoria']], use_container_width=True)

bar_chart = (
    alt.Chart(top_corr)
    .mark_bar()
    .encode(
        y=alt.Y('Variable:N', sort='-x', title='Variable'),
        x=alt.X('Correlación con la Victoria:Q', title='Correlación con el Resultado'),
        color=alt.condition(
            alt.datum['Correlación con la Victoria'] > 0,
            alt.value("green"),
            alt.value("red")
        ),
        tooltip=['Variable', 'Correlación con la Victoria']
    )
    .properties(title="Top 15 Variables más Correlacionadas con la Victoria")
)
st.altair_chart(bar_chart, use_container_width=True)

# ==============================================================  
# 📋 Conclusiones Generales  
# ==============================================================  

st.markdown("""
---
## 🧾 Conclusiones Generales

- **La localía** da ventaja constante pero no decisiva.  
- **El diferencial ofensivo–defensivo** es el mejor predictor del resultado.  
- **Las rachas solo importan si son extremas (≥6)**.  
- **Wins% históricos** predicen mejor que el momentum reciente.  
- **Menos pérdidas (turnovers)** y **mayor TS%** son claves de consistencia.  
- **Correlaciones** confirman la relevancia de *wins_percent_diff*, *net_rating_diff* y *estimated_point_diff* como variables líderes del modelo.
""")
