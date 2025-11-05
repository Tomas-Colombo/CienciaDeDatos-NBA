import streamlit as st
import pandas as pd
import altair as alt

st.title("📊 Exploración de Datos NBA 2024-25")

# --- Cargar dataset procesado ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/graph/games_clean.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

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

# Ordenar equipos por rendimiento y mapear nombres completos
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
    .properties(
        width=700,
        height=400,
        title="Evolución de la Métrica por Partido"
    )
)

if selected_team_name != "Todos":
    line_chart = line_chart.encode(color=alt.value("#1f77b4"))

# --- Gráfico de barras resumen ---
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
            tooltip=[
                alt.Tooltip('metric_es:N', title='Métrica'),
                alt.Tooltip('value:Q', title='Valor', format='.2f')
            ]
        )
        .properties(
            title=f'Estadísticas promedio de {selected_team_name}',
            width=350,
            height=400
        )
    )


# --- Mostrar gráficos ---
col1, col2 = st.columns([2.5, 1])

with col1:
    st.altair_chart(line_chart, use_container_width=True)

with col2:
    if selected_team_name != "Todos":
        st.altair_chart(bars, use_container_width=True)
        wins = int(team_summary.loc[team_summary['team_name'] == selected_team_name, 'win_game'].iloc[0])
        losses = int(team_summary.loc[team_summary['team_name'] == selected_team_name, 'loss_game'].iloc[0])
        st.metric("🏆 Victorias", wins)
        st.metric("💔 Derrotas", losses)
    else:
        st.info("Seleccioná un equipo para ver estadísticas detalladas ⬆️")

st.markdown("---")
st.caption("Visualización interactiva creada con Altair y Streamlit • Datos NBA 2024-25")
