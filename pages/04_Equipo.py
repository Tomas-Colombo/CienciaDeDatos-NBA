import streamlit as st
from pathlib import Path

st.title("👥 Equipo de Desarrollo")

st.markdown("""
Proyecto realizado para la materia **Ciencia de Datos – UTN-FRM**  
**Integrantes del equipo:**
""")

# --- Configuración de integrantes ---
integrantes = [
    {
        "nombre": "Tomás Colombo",
        "foto": "assets/Tomy.jpg",
        "rol": "Desarrollo del modelo y visualización"
    },
    {
        "nombre": "Facundo Sampieri",
        "foto": "assets/Facu.jpg",
        "rol": "Ingesta de datos y limpieza"
    },
    {
        "nombre": "Joaquin Prato",
        "foto": "assets/Joaco.jpg",
        "rol": "Evaluación de modelos y documentación"
    },
]

# --- Mostrar integrantes ---
cols = st.columns(len(integrantes))
for col, integrante in zip(cols, integrantes):
    foto_path = Path(integrante["foto"])
    with col:
        if foto_path.exists():
            st.image(str(foto_path), width=200)
        else:
            st.image("https://via.placeholder.com/200?text=Foto", width=200)
        st.markdown(f"### {integrante['nombre']}")
        st.caption(integrante["rol"])
