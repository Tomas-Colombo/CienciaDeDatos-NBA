import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("🤖 Modelo y Predicción")
MODEL_PATH = Path("models/logreg_no_percents_pipeline.pkl")
TEAMS_PATH = Path("data/prediction/teams_advanced_2024_25.csv")


# --- parche robusto para unpickle de __main__.DropColumns ---
import sys, types
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, cols=None, to_drop=None, drop_cols=None):
        # Estos son para el caso de que se instancie “nuevo”
        self.columns   = list(columns) if columns is not None else None
        self.cols      = list(cols) if cols is not None else None
        self.to_drop   = list(to_drop) if to_drop is not None else None
        self.drop_cols = list(drop_cols) if drop_cols is not None else None

    def _get_cols_to_drop(self):
        # Cubrimos los nombres comunes que puede haber tenido el objeto entrenado
        candidate_names = [
            "columns", "cols", "to_drop", "drop_cols",
            "columns_", "cols_", "to_drop_", "drop_cols_",
        ]
        for name in candidate_names:
            val = getattr(self, name, None)
            if val is not None:
                try:
                    return list(val)
                except Exception:
                    pass
        return []  # default seguro si no había nada serializado

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = self._get_cols_to_drop()
        if not isinstance(X, pd.DataFrame):
            # Si llegara un ndarray, no dropeamos (o convertir a DF con mismas cols si las conocés)
            return X
        safe_cols = [c for c in cols if c in X.columns]
        return X.drop(columns=safe_cols, errors="ignore")

# Asegurar que __main__ tenga DropColumns exactamente con ese nombre
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
setattr(sys.modules['__main__'], 'DropColumns', DropColumns)

def _ensure_dropcolumns(model):
    from sklearn.pipeline import Pipeline
    def touch(step):
        if isinstance(step, DropColumns) and not hasattr(step, "columns"):
            step.columns = []
        # manejar ColumnTransformer / Pipelines anidados
        try:
            from sklearn.compose import ColumnTransformer
            if isinstance(step, ColumnTransformer):
                for _, transformer, _ in step.transformers:
                    touch(transformer)
            if isinstance(step, Pipeline):
                for _, sub in step.steps:
                    touch(sub)
        except Exception:
            pass

    touch(model)
    return model
# --- fin del parche ---


# ====== Carga del pipeline entrenado desde el archivo .pkl ======
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el modelo: {MODEL_PATH}")
        st.stop()
    m = joblib.load(MODEL_PATH)
    m = _ensure_dropcolumns(m)  # el parche 
    return m

model = load_model()


# ====== Carga de datos actuales de equipos desde el csv ======
@st.cache_data
def load_teams():
    try:
        df = pd.read_csv(TEAMS_PATH,  sep=";") 
    except FileNotFoundError:
        st.warning(f"No se encontró el archivo de equipos: {TEAMS_PATH}")
        return pd.DataFrame()

    # columnas que realmente usás de ese Excel
    cols_necesarias = ["TEAM", "GP", "W", "OffRtg", "DefRtg", "streak", "streak_as_local", "streak_as_visitor"]
    faltan = [c for c in cols_necesarias if c not in df.columns]
    if faltan:
        st.error(f"Faltan columnas {faltan} en {TEAMS_PATH}")
        return pd.DataFrame()

    return df

teams_df = load_teams()


def get_team_data(team_name, df):
    match = df[df["TEAM"] == team_name]
    if match.empty:
        return None
    row = match.iloc[0]
    return normalize_team_row(row)


def get_team_data(team_name: str, df: pd.DataFrame):
    """
    Busca el equipo por nombre en el DataFrame y devuelve el dict normalizado.
    """
    if df.empty or not team_name:
        return None

    match = df[df["TEAM"] == team_name]
    if match.empty:
        return None

    row = match.iloc[0]
    return normalize_team_row(row)

def normalize_team_row(row):
    """
    Convierte una fila del CSV (TEAM, GP, W, OffRtg, DefRtg)
    a las estadísticas que usa tu modelo para ese equipo.
    """
    return {
        "off_rating": float(row["OffRtg"]),
        "def_rating": float(row["DefRtg"]),
        "wins": int(row["W"]),
        "game_number": int(row["GP"]),
        "streak": int(row["streak"]),
        "streak_as_local": int(row["streak_as_local"]),
        "streak_as_visitor": int(row["streak_as_visitor"]),
    }




import pandas as pd

def categorize_team_quality(wins_pct: float) -> str:
    """Asigna una etiqueta de calidad basada en el porcentaje de victorias."""
    if pd.isna(wins_pct):
        return 'Desconocido'
    elif wins_pct < 0.35:
        return 'Muy Débil'
    elif wins_pct < 0.45:
        return 'Débil'
    elif wins_pct < 0.55:
        return 'Promedio'
    elif wins_pct < 0.65:
        return 'Fuerte'
    else:
        return 'Muy Fuerte'

def categorize_streak_extreme(streak: int) -> int:
    """Clasifica la racha en extrema (+1), muy negativa (-1), o normal (0)."""
    if pd.isna(streak):
        return 0
    elif streak >= 6:
        return 1
    elif streak <= -6:
        return -1
    else:
        return 0

# ====== Formulario en español ======
st.markdown("Completá los datos del **equipo local** y **visitante**. Los nombres están en lenguaje común (NBA).")

PLACEHOLDER = "— Seleccioná —"
TEAM_LIST = sorted(teams_df["TEAM"].unique())

with st.form("pred_v3"):
    c1, c2 = st.columns(2)

    # ---------- Local (home) ----------
    c1.subheader("Equipo Local")
    home_choice = c1.selectbox("Elegí el equipo local", [PLACEHOLDER] + TEAM_LIST, index=0, key="home_lbl")
    home_name = None if home_choice == PLACEHOLDER else home_choice

    home_stats = get_team_data(home_name, teams_df) or {}

    h_off_rating   = home_stats.get("off_rating", 110.0)
    h_def_rating   = home_stats.get("def_rating", 110.0)
    h_wins         = home_stats.get("wins", 50)
    h_game_number  = home_stats.get("game_number", 60)

    h_streak = home_stats.get("streak", 0)
    home_home_str = home_stats.get("streak_as_local", 0)
    home_away_str = home_stats.get("streak_as_visitor", 0)

    # ---------- Visitante (visitor) ----------
    c2.subheader("Equipo Visitante")
    visitor_choice = c2.selectbox("Elegí el equipo visitante", [PLACEHOLDER] + TEAM_LIST, index=0, key="away_lbl")
    visitor_name = None if visitor_choice == PLACEHOLDER else visitor_choice

    # Datos del visitante automáticamente desde el CSV
    visitor_stats = get_team_data(visitor_name, teams_df) or {}

    v_off_rating = visitor_stats.get("off_rating", 109.0)
    v_def_rating = visitor_stats.get("def_rating", 109.0)
    v_wins = visitor_stats.get("wins", 35)
    v_game_number = visitor_stats.get("game_number", 60)

    v_streak = visitor_stats.get("streak", 0)
    vis_home_str = visitor_stats.get("streak_as_local", 0)
    vis_away_str = visitor_stats.get("streak_as_visitor", 0)

    # Botón (sin disabled; validamos después)
    ok = st.form_submit_button("Predecir", use_container_width=True)

# Validación mínima post-submit (evita equipos iguales o sin elegir)
if ok:
    if (home_name is None) or (visitor_name is None):
        st.error("Debés seleccionar **ambos** equipos.")
        st.stop()
    if home_name == visitor_name:
        st.error("El equipo visitante no puede ser el mismo que el local.")
        st.stop()

# ====== Construcción EXACTA de features que pide tu pipeline ====== h_wins 
if ok:
    # Etiquetas para mostrar resultado
    home_label = home_name if home_name else "Local"
    visitor_label = visitor_name if visitor_name else "Visitante"

    # Diferenciales 
    h_win_perent = h_wins / h_game_number           #home_wins_percent
    v_win_perent = v_wins / v_game_number

    wins_percent_diff     = h_win_perent - v_win_perent     
    offensive_rating_diff = h_off_rating - v_off_rating     
    defensive_rating_diff = h_def_rating - v_def_rating      
    
    net_rating_diff       = offensive_rating_diff + defensive_rating_diff #offensive_rating_diff + defensive_rating_diff
    
    h_estimated_points = h_off_rating - (h_off_rating - v_def_rating) / 2
    v_estimated_points = v_off_rating - (v_off_rating - h_def_rating) / 2
    
    estimated_point_diff  = h_estimated_points - v_estimated_points     #home_pts - vis_pts
    streak_diff           = h_streak - v_streak                         #home_str - vis_str

    # Categóricas/umbrales
    home_quality          = categorize_team_quality(h_win_perent)
    visitor_quality       = categorize_team_quality(v_win_perent)
    home_much_better     = int(wins_percent_diff > 0.20)
    visitor_much_better  = int(wins_percent_diff < -0.20)
    teams_evenly_matched = int(abs(wins_percent_diff) <= 0.10)
    home_streak_extreme    = categorize_streak_extreme(h_streak)    #streak_extreme(home_str)
    visitor_streak_extreme = categorize_streak_extreme(v_streak)    #streak_extreme(vis_str)
    streak_extreme_diff    = home_streak_extreme - visitor_streak_extreme

    home_last10 = h_win_perent 
    vis_last10  = v_win_perent


  

    # DataFrame con TODAS las features de entrenamiento (nombres EXACTOS)
    X = pd.DataFrame([{
        # --- Núcleo base ---
        "home_game_number": h_game_number,
        "home_streak": h_streak,
        "home_home_streak": home_home_str,
        "home_away_streak": home_away_str,
        "home_offensive_rating": h_off_rating,
        "home_defensive_rating": h_def_rating,

        "visitor_game_number": v_game_number,
        "visitor_streak": v_streak,
        "visitor_home_streak": vis_home_str,
        "visitor_away_streak": vis_away_str,
        "visitor_offensive_rating": v_off_rating,
        "visitor_defensive_rating": v_def_rating,

        "home_last_10": home_last10,
        "visitor_last_10": vis_last10,
        "home_wins_percent": h_win_perent,
        "visitor_wins_percent": v_win_perent,

        # --- Derivadas / flags (idénticas a entrenamiento) ---
        "wins_percent_diff": wins_percent_diff,
        "offensive_rating_diff": offensive_rating_diff,
        "defensive_rating_diff": defensive_rating_diff,
        "net_rating_diff": net_rating_diff,

        "home_estimated_points": h_estimated_points,
        "visitor_estimated_points": v_estimated_points,
        "estimated_point_diff": estimated_point_diff,

        "home_streak_extreme": home_streak_extreme,
        "visitor_streak_extreme": visitor_streak_extreme,
        "streak_extreme_diff": streak_extreme_diff,
        "streak_diff": streak_diff,

        "home_quality": home_quality,
        "visitor_quality": visitor_quality,
        "home_much_better": home_much_better,
        "visitor_much_better": visitor_much_better,
        "teams_evenly_matched": teams_evenly_matched,
    }])

    # Predicción
    try:
        y = model.predict(X)[0]
        # Mensaje usando nombres si se ingresaron (sino Local/Visitante)
        if int(y) == 1:
            st.success(f"Predicción: **{visitor_label} gana** frente a **{home_label}**")
        else:
            st.success(f"Predicción: **{home_label} gana** frente a **{visitor_label}**")
        st.caption(f"Valor binario predicho: {int(y)} (1 = gana {home_label}, 0 = gana {visitor_label})")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write({
                f"Probabilidad de ganar ({home_label} )": float(proba[0]),
                f"Probabilidad de ganar({visitor_label} )": float(proba[1]),
            })
        with st.expander("Ver vector de entrada (features)"):
            st.dataframe(X.T, use_container_width=True)
    except Exception as e:
        st.error("El pipeline no pudo predecir con las columnas construidas.")
        st.exception(e)
 