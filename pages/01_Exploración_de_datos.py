import streamlit as st
import pandas as pd
import altair as alt

st.title("📊 Exploración de Datos NBA 2024-25")

# ================================
# 🔹 1. Dataset principal (por equipo)
# ================================
@st.cache_data
def load_team_data():
    df = pd.read_csv("data/graph/games_clean.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_team_data()

# --- Diccionario nombres de equipos ---
team_names = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

# --- Traducción de métricas ---
metric_labels = {
    "W_percent": "% Victorias",
    "net_rating": "Net Rating",
    "oRtg": "Rating Ofensivo",
    "dRtg": "Rating Defensivo",
    "tsPercent": "% Tiro Verdadero",
    "win_game": "Victorias Totales",
    "loss_game": "Derrotas Totales"
}

# --- Resumen por equipo ---
team_summary = (
    df.groupby('team')
    .agg({
        'oRtg': 'mean',
        'dRtg': 'mean',
        'net_rating': 'mean',
        'tsPercent': 'mean',
        'W_percent': 'mean',
        'win_game': 'sum',
        'loss_game': 'sum'
    })
    .reset_index()
)

team_summary['team_name'] = team_summary['team'].map(team_names)
team_summary_long = team_summary.melt(
    id_vars=['team', 'team_name'],
    value_vars=['oRtg', 'dRtg', 'net_rating', 'tsPercent', 'W_percent', 'win_game', 'loss_game'],
    var_name='metric',
    value_name='value'
)

# --- Selectores ---
metric_es = st.selectbox(
    "📈 Elegí una métrica:",
    [metric_labels[m] for m in ['W_percent', 'net_rating', 'oRtg', 'dRtg', 'tsPercent']],
    index=0
)

# Volver al nombre técnico de la métrica
metric = {v: k for k, v in metric_labels.items()}[metric_es]

teams_sorted = team_summary.sort_values('W_percent', ascending=False)
teams_options = ['Todos'] + teams_sorted['team_name'].tolist()
selected_team_name = st.selectbox("🏀 Elegí un equipo:", teams_options)

# Obtener código del equipo
if selected_team_name != "Todos":
    selected_team = team_summary.loc[team_summary['team_name'] == selected_team_name, 'team'].iloc[0]
else:
    selected_team = None

st.markdown(f"### 📊 Evolución de **{metric_labels[metric]}**")

# --- Filtrar dataset ---
if selected_team_name != "Todos":
    df_filtered = df[df['team'] == selected_team]
else:
    df_filtered = df.copy()

# --- Gráfico principal ---
line_chart = (
    alt.Chart(df_filtered)
    .transform_calculate(metric_value=f"datum['{metric}']")
    .mark_line(point=True, interpolate='monotone', strokeWidth=2)
    .encode(
        x=alt.X('game_number:Q', title='Número de Partido'),
        y=alt.Y('metric_value:Q', title=metric_labels[metric]),
        color=alt.Color(
            'team:N',
            legend=alt.Legend(title='Equipo'),
            scale=alt.Scale(scheme='tableau10')
        ),
        tooltip=[
            alt.Tooltip('team:N', title='Equipo'),
            alt.Tooltip('game_number:Q', title='Partido'),
            alt.Tooltip('W_percent:Q', title='% Victorias', format='.2f'),
            alt.Tooltip('net_rating:Q', title='Net Rating', format='.2f'),
            alt.Tooltip('oRtg:Q', title='Rating Ofensivo', format='.2f'),
            alt.Tooltip('dRtg:Q', title='Rating Defensivo', format='.2f'),
            alt.Tooltip('tsPercent:Q', title='% Tiro Verdadero', format='.2f')
        ]
    )
    .properties(width=700, height=400, title="Evolución de la Métrica por Partido")
)

if selected_team_name != "Todos":
    line_chart = line_chart.encode(color=alt.value("#1f77b4"))

# --- Gráfico de barras resumen ---
col1, col2 = st.columns([2.5, 1])

with col1:
    st.altair_chart(line_chart, use_container_width=True)

with col2:
    if selected_team_name != "Todos":
        team_data = team_summary_long[team_summary_long['team_name'] == selected_team_name].copy()
        team_data['metric_es'] = team_data['metric'].map(metric_labels)

        bars = (
            alt.Chart(team_data)
            .mark_bar(size=30)
            .encode(
                y=alt.Y('metric_es:N', sort='-x', title='Métrica'),
                x=alt.X('value:Q', title='Valor Promedio'),
                color=alt.Color('metric_es:N', scale=alt.Scale(scheme='set2'), legend=None),
                tooltip=[alt.Tooltip('metric_es:N', title='Métrica'),
                         alt.Tooltip('value:Q', title='Valor', format='.2f')]
            )
            .properties(title=f'Estadísticas promedio de {selected_team_name}', width=350, height=400)
        )

        st.altair_chart(bars, use_container_width=True)
        wins = int(team_summary.loc[team_summary['team_name'] == selected_team_name, 'win_game'].iloc[0])
        losses = int(team_summary.loc[team_summary['team_name'] == selected_team_name, 'loss_game'].iloc[0])
        st.metric("🏆 Victorias", wins)
        st.metric("💔 Derrotas", losses)
    else:
        st.info("Seleccioná un equipo para ver estadísticas detalladas ⬆️")

# ================================
# 🔹 2. Nuevos gráficos con el CSV de partidos
# ================================
st.markdown("---")
st.header("📈 Análisis por Partido")

@st.cache_data
def load_games_data():
    df = pd.read_csv("data/processed/games_final_csv.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df_games = load_games_data()

# --- Diccionario para nombres de métricas ---
metricas_map = {
    "offensive_rating": "Rating Ofensivo",
    "defensive_rating": "Rating Defensivo",
    "ts_percent": "% Tiro Verdadero",
    "assist_percent": "% Asistencias",
    "rebound_percent": "% Rebotes",
    "turnover_percent": "% Pérdidas",
}

# --- Selector de métrica ---
metrica_es = st.selectbox("📊 Elegí la métrica a comparar:", list(metricas_map.values()), index=0)
metrica = {v: k for k, v in metricas_map.items()}[metrica_es]

# --- Datos por condición ---
home_data = df_games.groupby('home_team')[f'home_{metrica}'].mean().reset_index()
home_data.columns = ['team', 'home_value']

visitor_data = df_games.groupby('visitor_team')[f'visitor_{metrica}'].mean().reset_index()
visitor_data.columns = ['team', 'visitor_value']

# Merge para incluir todos los equipos
home_away = pd.merge(home_data, visitor_data, on='team', how='outer')
home_away['team_name'] = home_away['team'].map(team_names)

# Reorganizar formato largo
home_away_long = home_away.melt(
    id_vars=['team', 'team_name'],
    value_vars=['home_value', 'visitor_value'],
    var_name='condicion',
    value_name='valor'
)

# Traducir condición
home_away_long['condicion'] = home_away_long['condicion'].map({
    'home_value': 'Local',
    'visitor_value': 'Visitante'
})

# --- Gráfico con interactividad ---
highlight = alt.selection_point(fields=['team_name'], on='mouseover', clear='mouseout')

chart_home_away = (
    alt.Chart(home_away_long)
    .mark_bar()
    .encode(
        x=alt.X('team_name:N', sort='-y', title='Equipo'),
        y=alt.Y('valor:Q', title=metrica_es),
        color=alt.Color('condicion:N', title='Condición', scale=alt.Scale(scheme='set2')),
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.3)),
        tooltip=[
            alt.Tooltip('team_name:N', title='Equipo'),
            alt.Tooltip('condicion:N', title='Condición'),
            alt.Tooltip('valor:Q', title=metrica_es, format='.2f')
        ]
    )
    .add_params(highlight)
    .properties(
        width=900,
        height=500,
        title=f"🏠 Comparación Local vs Visitante ({metrica_es} Promedio)"
    )
)

st.altair_chart(chart_home_away, use_container_width=True)

# --- Distribución de % Tiro Verdadero ---
st.subheader("🎯 Distribución de Tiro Verdadero (TS%) por Equipo")

df_long_ts = pd.concat([
    df_games[['home_team', 'home_ts_percent']].rename(columns={'home_team': 'team', 'home_ts_percent': 'ts'}),
    df_games[['visitor_team', 'visitor_ts_percent']].rename(columns={'visitor_team': 'team', 'visitor_ts_percent': 'ts'})
])

df_long_ts['team_name'] = df_long_ts['team'].map(team_names)

chart_ts = (
    alt.Chart(df_long_ts)
    .mark_boxplot(extent='min-max')
    .encode(
        x=alt.X('team_name:N', sort='-y', title='Equipo'),
        y=alt.Y('ts:Q', title='% Tiro Verdadero', scale=alt.Scale(domain=[0.4, 0.7])),
        color=alt.Color('team_name:N', legend=None),
        tooltip=[alt.Tooltip('team_name:N', title='Equipo'), alt.Tooltip('ts:Q', title='% Tiro Verdadero', format='.2f')]
    )
    .properties(width=900, height=500, title='Distribución del % de Tiro Verdadero')
)

st.altair_chart(chart_ts, use_container_width=True)

st.subheader("⚖️ Correlación entre Rating Ofensivo y Defensivo")

team_eff = (
    df_games.groupby('home_team')
    .agg({'home_offensive_rating': 'mean', 'home_defensive_rating': 'mean'})
    .reset_index()
)
team_eff['team_name'] = team_eff['home_team'].map(team_names)

mean_off = team_eff['home_offensive_rating'].mean()
mean_def = team_eff['home_defensive_rating'].mean()

scatter = (
    alt.Chart(team_eff)
    .mark_circle(size=150)
    .encode(
        x=alt.X('home_offensive_rating:Q',
                title='Rating Ofensivo Promedio',
                scale=alt.Scale(domain=[95, 130])),
        y=alt.Y('home_defensive_rating:Q',
                title='Rating Defensivo Promedio (↓ mejor)',
                scale=alt.Scale(domain=[95, 130], reverse=True)),
        color=alt.Color('team_name:N', legend=None),
        tooltip=[
            alt.Tooltip('team_name:N', title='Equipo'),
            alt.Tooltip('home_offensive_rating:Q', title='OffRtg', format='.2f'),
            alt.Tooltip('home_defensive_rating:Q', title='DefRtg', format='.2f')
        ]
    )
)

text = scatter.mark_text(align='left', dx=8, dy=0, fontSize=11).encode(
    text='team_name:N'
)

mean_lines = (
    alt.Chart(pd.DataFrame({'x': [mean_off], 'y': [mean_def]}))
    .mark_rule(strokeDash=[5, 5], color='gray')
    .encode(x='x:Q')
    +
    alt.Chart(pd.DataFrame({'x': [mean_off], 'y': [mean_def]}))
    .mark_rule(strokeDash=[5, 5], color='gray')
    .encode(y='y:Q')
)

quadrant_labels = pd.DataFrame({
    'x': [mean_off + 8, mean_off + 8, mean_off - 8, mean_off - 8],
    'y': [mean_def + 8, mean_def - 8, mean_def + 8, mean_def - 8],
    'label': ['⚔️ Buen ataque', '🔥 Elite', '💤 Débil', '🛡️ Buena defensa']
})

labels_chart = (
    alt.Chart(quadrant_labels)
    .mark_text(fontSize=14, fontWeight='bold', color='gray')
    .encode(x='x:Q', y='y:Q', text='label:N')
)

corr_chart = (scatter + text + mean_lines + labels_chart).properties(
    width=750,
    height=550,
    title="Relación entre Rating Ofensivo y Defensivo por Equipo (Promedio de Local)"
)

st.altair_chart(corr_chart, use_container_width=True)

st.info("💡 Ejes: hacia la **derecha** → mejor ataque; hacia **abajo** ↓ mejor defensa.")



st.markdown("---")
st.caption("Visualización interactiva creada con Altair y Streamlit • Datos NBA 2024-25")
