import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="📊 Análisis EDA", page_icon="📊")
st.title("📊 Análisis Exploratorio de Datos (EDA) – NBA Games")

DATA_PATH = Path("data/processed/games_final_csv.csv")

@st.cache_data
def load_df():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_df()

st.markdown("""
### 🧠 Hipótesis Evaluadas
A continuación se resumen los resultados más importantes del análisis exploratorio de datos.
""")

# --- H1 ---
with st.expander("**H1: Ventaja de Localía – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** Los equipos locales tienen mayor probabilidad de ganar.  
    **Evidencia:**
    - 54% de victorias locales vs 46% visitantes  
    - Ratings ofensivos más consistentes en casa  
    - True Shooting % superior en casa  

    **Conclusión:** La ventaja de localía existe (~+4pp), aunque no es el factor más determinante.
    """)
    home_win_rate = (df['home_result'] == 'W').mean() * 100
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('is_home:N', title='Condición'),
        y=alt.Y('mean(target):Q', title='Win Rate'),
        color='is_home:N'
    ).properties(title="Probabilidad de Victoria según Localía")
    st.altair_chart(chart, use_container_width=True)

# --- H2 ---
with st.expander("**H2: Diferencial Ofensivo-Defensivo como Predictor Principal – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** El diferencial entre el offensive rating del equipo y el defensive rating del oponente es el predictor más fuerte.  
    **Evidencia:**
    - Mediana +1.5 en victorias vs -2.0 en derrotas  
    - Clara separación en boxplots  
    """)
    box = alt.Chart(df).mark_boxplot().encode(
        x=alt.X('target:N', title='Resultado (0=Derrota, 1=Victoria)'),
        y=alt.Y('net_rating_diff:Q', title='Diferencial Neto')
    ).properties(title="Distribución del Diferencial Ofensivo-Defensivo")
    st.altair_chart(box, use_container_width=True)

# --- H3 ---
with st.expander("**H3: Efecto Acumulativo de Rachas – ⚠️ Parcialmente Confirmada**"):
    st.markdown("""
    **Hipótesis:** Solo las rachas largas (≥6) tienen efecto significativo.  
    **Evidencia:**
    - Rachas ≥+6 → 66% Win%
    - Rachas ≤-6 → 28% Win%
    - Rachas cortas → ~50% (sin efecto)
    """)
    streak_plot = alt.Chart(df).mark_bar().encode(
        x=alt.X('streak_extreme_diff:N', title='Tipo de Racha (Local - Visitante)'),
        y=alt.Y('mean(target):Q', title='Win Rate'),
        color='streak_extreme_diff:N'
    ).properties(title="Efecto de las Rachas Extremas en el Resultado")
    st.altair_chart(streak_plot, use_container_width=True)

# --- H4 ---
with st.expander("**H4: El Porcentaje de Victorias es Más Relevante que el Momentum – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** El historial completo de temporada predice mejor que el rendimiento reciente.  
    **Evidencia:**
    - Momentum: diferencia de solo 8% entre extremos  
    - Wins_percent: clara separación en boxplots  
    """)
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('visitor_wins_percent:Q', title='Visitor Win%'),
        y=alt.Y('home_wins_percent:Q', title='Home Win%'),
        color=alt.Color('target:N', title='Resultado'),
        tooltip=['home_wins_percent', 'visitor_wins_percent', 'target']
    ).properties(title="Relación entre Win% Local vs Visitante")
    st.altair_chart(chart, use_container_width=True)

# --- H5 ---
with st.expander("**H5: La Disciplina Ofensiva es Crítica – ✅ Confirmada**"):
    st.markdown("""
    **Hipótesis:** Los equipos con menor turnover_percent tienen más chances de ganar.  
    **Evidencia:**
    - Cleveland: -9.3% en turnovers vs promedio  
    - Métrica con mayor diferencia porcentual  
    """)
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('turnover_percent_diff:Q', title='Diferencial de Turnovers'),
        y=alt.Y('target:Q', title='Resultado (1=Victoria)'),
        color='target:N',
        tooltip=['turnover_percent_diff', 'target']
    ).properties(title="Impacto del Diferencial de Turnovers en la Victoria")
    st.altair_chart(scatter, use_container_width=True)

# --- Ranking de Correlaciones ---
st.markdown("---")
st.header("📈 Ranking de Variables Más Correlacionadas con la Victoria")

# Filtrar solo numéricas
num_df = df.select_dtypes(include=['number']).dropna(axis=1, how='all')

# Calcular correlación con el target
corr = num_df.corr(numeric_only=True)['target'].sort_values(ascending=False)

corr_df = corr.reset_index()
corr_df.columns = ['Variable', 'Correlación con Victoria']

# Mostrar top 15 correlaciones absolutas
corr_df['Abs'] = corr_df['Correlación con Victoria'].abs()
top_corr = corr_df.sort_values('Abs', ascending=False).head(15)

# Mostrar tabla + gráfico
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(top_corr[['Variable', 'Correlación con Victoria']], use_container_width=True)
with col2:
    bar_chart = alt.Chart(top_corr).mark_bar().encode(
        y=alt.Y('Variable:N', sort='-x'),
        x=alt.X('Correlación con Victoria:Q', title='Correlación con Resultado'),
        color=alt.condition(
            alt.datum['Correlación con Victoria'] > 0,
            alt.value("green"),
            alt.value("red")
        ),
        tooltip=['Variable', 'Correlación con Victoria']
    ).properties(title="Top 15 Variables más Correlacionadas con la Victoria")
    st.altair_chart(bar_chart, use_container_width=True)

# --- Conclusión general ---
st.markdown("""
---
## 🧾 Conclusiones Generales

- **La localía** da ventaja constante pero no decisiva.  
- **El diferencial ofensivo–defensivo** es el mejor predictor del resultado.  
- **Las rachas solo importan si son extremas (≥6)**.  
- **Wins% históricos** predicen mucho mejor que el momentum reciente.  
- **Turnovers bajos** y **eficiencia ofensiva alta (TS%)** son claves de consistencia.  
- **Correlaciones** confirman la importancia de *wins_percent_diff*, *net_rating_diff* y *estimated_point_diff* como variables líderes del modelo.
""")
