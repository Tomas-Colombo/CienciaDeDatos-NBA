import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import math

st.title("🤖 Modelo y Predicción")

MODEL_PATH = Path("models/logreg_no_percents_pipeline.pkl")

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

# ====== Helpers  ======
def categorize_team_quality(wins_pct):
    if pd.isna(wins_pct): return 'Desconocido'
    elif wins_pct < 0.35: return 'Muy Débil'
    elif wins_pct < 0.45: return 'Débil'
    elif wins_pct < 0.55: return 'Promedio'
    elif wins_pct < 0.65: return 'Fuerte'
    else: return 'Muy Fuerte'

def streak_extreme(s):
    try:
        s = int(s)
    except Exception:
        s = 0
    return 1 if s >= 6 else (-1 if s <= -6 else 0)

# ====== Formulario en español ======
st.markdown("Completá los datos del **equipo local** y **visitante**. Los nombres están en lenguaje común (NBA).")

with st.form("pred_v3"):
    c1, c2 = st.columns(2)

    # ---------- Local (home) ----------
    c1.subheader("Equipo Local")
    home_name = c1.text_input("Nombre equipo local (opcional)", value="")
    home_off = c1.number_input("Rating ofensivo del local", value=110.0, step=0.1,
                               help="Puntos anotados por 100 posesiones (OFF RTG).")
    home_def = c1.number_input("Rating defensivo del local", value=110.0, step=0.1,
                               help="Puntos recibidos por 100 posesiones (DEF RTG).")
    home_win = c1.slider("Victorias del local en la temporada (0 a 1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01,
                         help="Win% acumulado del local en la temporada regular.")
    home_pts = c1.number_input("Puntos estimados del local", value=112.0, step=0.1,
                               help="Estimación de puntos del local para el partido.")
    home_str = c1.number_input("Racha actual del local (±)", value=1, step=1,
                               help="Racha total (positiva o negativa) sin discriminar local/visita.")
    home_home_str = c1.number_input("Racha del local jugando de local (±)", value=1, step=1,
                                    help="Racha del local solo en partidos como local.")
    home_away_str = c1.number_input("Racha del local jugando de visitante (±)", value=0, step=1,
                                    help="Racha del local solo en partidos como visitante.")
    home_game_n = c1.number_input("Número de partido del local en la temporada (1–82)", min_value=1, max_value=82, value=41, step=1,
                                  help="Progresión del calendario del local (1 a 82).")
    home_last10 = c1.slider("Rendimiento del local en últimos 10 (0 a 1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01,
                            help="Win% del local en los últimos 10 partidos.")

    # ---------- Visitante (visitor) ----------
    c2.subheader("Equipo Visitante")
    visitor_name = c2.text_input("Nombre equipo visitante (opcional)", value="")
    vis_off = c2.number_input("Rating ofensivo del visitante", value=109.0, step=0.1)
    vis_def = c2.number_input("Rating defensivo del visitante", value=109.0, step=0.1)
    vis_win = c2.slider("Victorias del visitante en la temporada (0 a 1)", min_value=0.0, max_value=1.0, value=0.52, step=0.01)
    vis_pts = c2.number_input("Puntos estimados del visitante", value=110.0, step=0.1)
    vis_str = c2.number_input("Racha actual del visitante (±)", value=0, step=1)
    vis_home_str = c2.number_input("Racha del visitante jugando de local (±)", value=0, step=1)
    vis_away_str = c2.number_input("Racha del visitante jugando de visitante (±)", value=0, step=1)
    vis_game_n = c2.number_input("Número de partido del visitante en la temporada (1–82)", min_value=1, max_value=82, value=41, step=1)
    vis_last10 = c2.slider("Rendimiento del visitante en últimos 10 (0 a 1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    ok = st.form_submit_button("Predecir")

# ====== Construcción EXACTA de features que pide tu pipeline ======
if ok:
    # Etiquetas para mostrar resultado
    home_label = home_name.strip() if home_name.strip() else "Local"
    visitor_label = visitor_name.strip() if visitor_name.strip() else "Visitante"

    # Diferenciales 
    wins_percent_diff     = home_win - vis_win
    offensive_rating_diff = home_off - vis_off
    defensive_rating_diff = vis_def - home_def   
    net_rating_diff       = offensive_rating_diff + defensive_rating_diff
    estimated_point_diff  = home_pts - vis_pts
    streak_diff           = home_str - vis_str

    # Categóricas/umbrales
    home_quality          = categorize_team_quality(home_win)
    visitor_quality       = categorize_team_quality(vis_win)
    home_much_better     = int(wins_percent_diff > 0.20)
    visitor_much_better  = int(wins_percent_diff < -0.20)
    teams_evenly_matched = int(abs(wins_percent_diff) <= 0.10)
    home_streak_extreme    = streak_extreme(home_str)
    visitor_streak_extreme = streak_extreme(vis_str)
    streak_extreme_diff    = home_streak_extreme - visitor_streak_extreme

    # DataFrame con TODAS las features de entrenamiento (nombres EXACTOS)
    X = pd.DataFrame([{
        # --- Núcleo base ---
        "home_game_number": home_game_n,
        "home_streak": home_str,
        "home_home_streak": home_home_str,
        "home_away_streak": home_away_str,
        "home_offensive_rating": home_off,
        "home_defensive_rating": home_def,

        "visitor_game_number": vis_game_n,
        "visitor_streak": vis_str,
        "visitor_home_streak": vis_home_str,
        "visitor_away_streak": vis_away_str,
        "visitor_offensive_rating": vis_off,
        "visitor_defensive_rating": vis_def,

        "home_last_10": home_last10,
        "visitor_last_10": vis_last10,
        "home_wins_percent": home_win,
        "visitor_wins_percent": vis_win,

        # --- Derivadas / flags (idénticas a entrenamiento) ---
        "wins_percent_diff": wins_percent_diff,
        "offensive_rating_diff": offensive_rating_diff,
        "defensive_rating_diff": defensive_rating_diff,
        "net_rating_diff": net_rating_diff,

        "home_estimated_points": home_pts,
        "visitor_estimated_points": vis_pts,
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
            st.success(f"Predicción: **{home_label} gana** frente a **{visitor_label}**")
        else:
            st.success(f"Predicción: **{visitor_label} gana** frente a **{home_label}**")
        st.caption(f"Valor binario predicho: {int(y)} (1 = gana {home_label}, 0 = gana {visitor_label})")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write({
                f"Prob({home_label} pierde = 0)": float(proba[0]),
                f"Prob({home_label} gana = 1)": float(proba[1]),
            })
        with st.expander("Ver vector de entrada (features)"):
            st.dataframe(X.T, use_container_width=True)
    except Exception as e:
        st.error("El pipeline no pudo predecir con las columnas construidas.")
        st.exception(e)
