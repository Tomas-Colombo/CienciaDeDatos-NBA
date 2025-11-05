import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")
st.title("👥 Equipo de Desarrollo")

st.markdown("""
Proyecto realizado para la materia **Ciencia de Datos – UTN-FRM**  
**Integrantes del equipo:**
""")

# --- CSS: hace circulares las imágenes y centra todo ---
st.markdown("""
<style>
/* Hace circulares todas las imágenes insertadas con st.image */
.stImage img{
  border-radius: 50%;
  width: 180px !important;
  height: 180px !important;
  object-fit: cover;           /* recorta para que queden bien aunque la foto no sea cuadrada */
  box-shadow: 0 2px 10px rgba(0,0,0,.25);
  display: block;
  margin-left: auto;
  margin-right: auto;
}

/* Centra los textos debajo de cada foto */
.team-name { text-align:center; margin: .6rem 0 0; font-weight: 600; font-size: 1.1rem; }
.team-role { text-align:center; color: #9aa0a6; margin: .1rem 0 0; font-size: .95rem; }
</style>
""", unsafe_allow_html=True)

# --- Configuración de integrantes ---
integrantes = [
    {"nombre": "Tomás Colombo",   "foto": "assets/Tomy.jpg", "rol": "Desarrollo del modelo y visualización"},
    {"nombre": "Facundo Sampieri","foto": "assets/Facu.jpg", "rol": "Ingesta de datos y limpieza"},
    {"nombre": "Joaquin Prato",   "foto": "assets/Joaco.jpg","rol": "Evaluación de modelos y documentación"},
]

# --- Mostrar integrantes en columnas (alineadas horizontalmente) ---
sp1, c1, sp2, c2, sp3, c3, sp4 = st.columns([3, 2, 1, 2, 1, 2, 3])
cols = [c1, c2, c3]
for col, integrante in zip(cols, integrantes):
    with col:
        foto_path = Path(integrante["foto"])
        if foto_path.exists():
            st.image(str(foto_path))
        else:
            st.image("https://via.placeholder.com/180?text=Foto")
        st.markdown(f"<div class='team-name'>{integrante['nombre']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='team-role'>{integrante['rol']}</div>", unsafe_allow_html=True)
