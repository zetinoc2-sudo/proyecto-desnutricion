import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CSV_LIMPIO,
    DIR_DATOS,
    DIR_GRAFICOS,
    DIR_MODELOS,
    ETIQUETAS,
    NOMBRES_MODELOS,
    ARCHIVO_MODELO,
    OBJ_INSEG,
    OBJ_NUT,
    TITULOS,
)
from preparar_datos import agregar_derivadas, columnas_modelo

st.set_page_config(page_title="Riesgo nutricional infantil", page_icon="🥗", layout="wide")

AVISO = (
    "Herramienta académica entrenada con datos sintéticos. Sus resultados son orientativos "
    "y no sustituyen la evaluación de un médico, nutricionista o institución de salud."
)

FUENTES = pd.DataFrame(
    [
        ["UNICEF", "Perfil de país: El Salvador", "https://data.unicef.org/country/slv/"],
        [
            "UNICEF / OMS / Banco Mundial",
            "Joint Child Malnutrition Estimates",
            "https://data.unicef.org/resources/dataset/malnutrition-data/",
        ],
        [
            "Banco Mundial",
            "Prevalencia de retraso del crecimiento, El Salvador",
            "https://data.worldbank.org/indicator/SH.STA.STNT.ME.ZS?locations=SV",
        ],
        ["Ministerio de Salud (MINSAL)", "Portal institucional", "https://www.salud.gob.sv/"],
        [
            "FAO",
            "Panorama regional de la seguridad alimentaria y la nutrición 2024",
            "https://www.fao.org/americas/publicaciones/panorama/2024/es",
        ],
        ["Programa Mundial de Alimentos", "El Salvador", "https://es.wfp.org/paises/el-salvador"],
        [
            "OMS",
            "Patrones de crecimiento infantil",
            "https://www.who.int/tools/child-growth-standards",
        ],
    ],
    columns=["Institución", "Recurso", "Enlace"],
)


def artefactos_listos():
    return (DIR_MODELOS / "mejores.json").exists() and (DIR_MODELOS / "kmeans.joblib").exists()


@st.cache_resource
def cargar_modelos():
    modelos = {}
    for objetivo in (OBJ_NUT, OBJ_INSEG):
        for nombre, archivo in ARCHIVO_MODELO.items():
            modelos[(objetivo, nombre)] = joblib.load(DIR_MODELOS / f"{objetivo}_{archivo}.joblib")
    agrupamiento = joblib.load(DIR_MODELOS / "kmeans.joblib")
    mejores = json.loads((DIR_MODELOS / "mejores.json").read_text(encoding="utf-8"))
    return modelos, agrupamiento, mejores


@st.cache_data
def cargar_tablas():
    datos = pd.read_csv(CSV_LIMPIO)
    metricas = pd.read_csv(DIR_DATOS / "metricas_modelos.csv")
    perfiles = pd.read_csv(DIR_DATOS / "perfiles_grupos.csv", index_col=0)
    return datos, metricas, perfiles


def mostrar_imagen(nombre, leyenda=None):
    ruta = DIR_GRAFICOS / nombre
    if ruta.exists():
        st.image(str(ruta), caption=leyenda)


def si_no(etiqueta, clave):
    return st.radio(etiqueta, ["Sí", "No"], horizontal=True, key=clave) == "Sí"


def pagina_inicio():
    st.title("Identificación de factores de riesgo de desnutrición e inseguridad alimentaria")
    st.subheader("Primera infancia en El Salvador, 2026")
    st.warning(AVISO)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Qué hace la aplicación**

            - Recibe información de un niño o niña menor de cinco años y de su hogar.
            - Estima el nivel de **riesgo nutricional** y de **inseguridad alimentaria** con modelos de Machine Learning.
            - Ubica al hogar dentro de un grupo de características similares con K-Means.
            """
        )
    with c2:
        st.markdown(
            """
            **Algoritmos utilizados**

            - **K-Means** (no supervisado): agrupa hogares con características parecidas.
            - **Regresión logística** (supervisado): estima probabilidades de cada nivel y es fácil de interpretar.
            - **Random Forest** (supervisado): clasifica niveles de riesgo y mide la importancia de las variables.
            - **XGBoost** (supervisado): detecta patrones complejos en los datos.
            """
        )


def pagina_datos(datos):
    st.header("Análisis exploratorio de los datos")
    st.caption("Base sintética de hogares, ya limpia y preparada para el modelado.")
    st.metric("Registros", f"{len(datos):,}")
    st.dataframe(datos.head(20))
    st.dataframe(datos.describe().round(2))
    for archivo, texto in [
        ("distribucion_objetivos.png", "Distribución de las variables a predecir"),
        ("riesgo_por_zona.png", "Niveles de riesgo según zona de residencia"),
        ("variables_clave.png", "Relación entre variables clave y los niveles de riesgo"),
        ("correlacion.png", "Matriz de correlación"),
    ]:
        st.subheader(texto)
        mostrar_imagen(archivo)


def pagina_modelos(metricas, perfiles, agrupamiento, mejores):
    st.header("Resultados de los modelos")
    for objetivo in (OBJ_NUT, OBJ_INSEG):
        st.subheader(TITULOS[objetivo])
        tabla = metricas[metricas["objetivo"] == objetivo].drop(columns="objetivo")
        st.dataframe(tabla, hide_index=True)
        st.caption(f"Mejor modelo según validación cruzada: {mejores[objetivo]}")
        mostrar_imagen(f"matrices_{objetivo}.png")
        mostrar_imagen(f"importancia_{objetivo}.png")
    st.subheader("Agrupamiento de hogares con K-Means")
    st.write(f"Número de grupos seleccionado: {agrupamiento['k']}")
    mostrar_imagen("kmeans_seleccion_k.png")
    mostrar_imagen("kmeans_grupos.png")
    st.dataframe(perfiles)


def pagina_evaluacion(modelos, agrupamiento, mejores, perfiles):
    st.header("Evaluación preventiva de un hogar")
    st.warning(AVISO)
    opcion = st.selectbox(
        "Modelo supervisado a utilizar",
        ["Mejor modelo según validación cruzada"] + NOMBRES_MODELOS,
    )

    with st.form("formulario"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Niño o niña**")
            edad = st.slider("Edad (meses)", 6, 59, 24)
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
            enfermedades = st.slider("Enfermedades en los últimos 3 meses", 0, 6, 1)
            controles = si_no("Controles de crecimiento al día", "controles")
            lactancia = si_no("Lactancia exclusiva los primeros 6 meses", "lactancia")
        with c2:
            st.markdown("**Hogar**")
            zona = st.selectbox("Zona de residencia", ["Urbana", "Rural"])
            integrantes = st.number_input("Personas en el hogar", 2, 10, 5)
            ninos = st.number_input("Niños menores de 5 años", 1, 4, 1)
            ingreso = st.number_input("Ingreso mensual del hogar (USD)", 0.0, 10000.0, 400.0, step=10.0)
            gasto = st.slider("Porcentaje del ingreso gastado en alimentos", 10, 85, 40)
            escolaridad = st.slider("Años de estudio de la madre o cuidadora", 0, 16, 8)
            jefatura = si_no("Jefatura de hogar femenina", "jefatura")
        with c3:
            st.markdown("**Alimentación y servicios**")
            diversidad = st.slider("Grupos de alimentos consumidos (0 a 12)", 0, 12, 6)
            comidas = st.slider("Comidas al día", 1, 5, 3)
            agua = si_no("Acceso a agua segura", "agua")
            saneamiento = si_no("Saneamiento adecuado", "saneamiento")
            apoyo = si_no("Recibe apoyo alimentario", "apoyo")
        enviar = st.form_submit_button("Evaluar")

    if not enviar:
        return

    fila = {
        "edad_meses": edad,
        "ingreso_mensual": ingreso,
        "integrantes_hogar": int(integrantes),
        "ninos_menores_5": int(ninos),
        "gasto_alimentos_pct": gasto,
        "diversidad_dietaria": diversidad,
        "comidas_dia": comidas,
        "escolaridad_madre": escolaridad,
        "enfermedades_3m": enfermedades,
        "acceso_agua_segura": int(agua),
        "saneamiento_adecuado": int(saneamiento),
        "controles_crecimiento_al_dia": int(controles),
        "lactancia_exclusiva_6m": int(lactancia),
        "recibe_apoyo_alimentario": int(apoyo),
        "jefatura_femenina": int(jefatura),
        "sexo": sexo,
        "zona": zona,
    }
    X = agregar_derivadas(pd.DataFrame([fila]))[columnas_modelo()]

    st.divider()
    columnas = st.columns(2)
    for columna, objetivo in zip(columnas, (OBJ_NUT, OBJ_INSEG)):
        nombre = mejores[objetivo] if opcion.startswith("Mejor") else opcion
        modelo = modelos[(objetivo, nombre)]
        probabilidades = modelo.predict_proba(X)[0]
        nivel = int(np.argmax(probabilidades))
        with columna:
            st.subheader(TITULOS[objetivo])
            texto = f"{ETIQUETAS[objetivo][nivel]} ({probabilidades[nivel] * 100:.0f}%)"
            if nivel == 2:
                st.error(texto)
            elif nivel == 1:
                st.warning(texto)
            else:
                st.success(texto)
            st.caption(f"Modelo: {nombre}")
            grafico = pd.DataFrame(
                {"Probabilidad": probabilidades},
                index=[ETIQUETAS[objetivo][i] for i in range(len(probabilidades))],
            )
            st.bar_chart(grafico)

    matriz = agrupamiento["prep"].transform(X)
    bruto = int(agrupamiento["kmeans"].predict(matriz)[0])
    grupo = agrupamiento["orden"][bruto]
    st.subheader(f"Grupo de hogares similares: {grupo} de {agrupamiento['k']}")
    st.dataframe(perfiles.loc[[grupo]])

    if max(
        int(np.argmax(modelos[(OBJ_NUT, mejores[OBJ_NUT])].predict_proba(X)[0])),
        int(np.argmax(modelos[(OBJ_INSEG, mejores[OBJ_INSEG])].predict_proba(X)[0])),
    ) == 2:
        st.info(
            "Se sugiere acudir a un centro de salud o a un programa de apoyo alimentario "
            "para una valoración profesional."
        )


def pagina_fuentes():
    st.header("Fuentes de información")
    st.dataframe(
        FUENTES,
        hide_index=True,
        column_config={"Enlace": st.column_config.LinkColumn("Enlace")},
    )
    st.caption(
        "Los datos con los que se entrenaron los modelos son sintéticos y no provienen de estas "
        "instituciones; las fuentes se utilizan como referencia del problema y de las variables."
    )


def main():
    st.sidebar.title("Menú")
    pagina = st.sidebar.radio(
        "Ir a",
        ["Inicio", "Análisis de datos", "Modelos", "Evaluación de un hogar", "Fuentes"],
    )
    if pagina == "Inicio":
        pagina_inicio()
        return
    if pagina == "Fuentes":
        pagina_fuentes()
        return
    if not artefactos_listos():
        st.error("Faltan los modelos entrenados. Ejecuta primero: python entrenar_modelos.py")
        st.stop()
    modelos, agrupamiento, mejores = cargar_modelos()
    datos, metricas, perfiles = cargar_tablas()
    if pagina == "Análisis de datos":
        pagina_datos(datos)
    elif pagina == "Modelos":
        pagina_modelos(metricas, perfiles, agrupamiento, mejores)
    else:
        pagina_evaluacion(modelos, agrupamiento, mejores, perfiles)


main()
