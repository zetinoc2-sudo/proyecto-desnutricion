from pathlib import Path

BASE = Path(__file__).resolve().parent
DIR_DATOS = BASE / "datos"
DIR_MODELOS = BASE / "modelos"
DIR_GRAFICOS = BASE / "graficos"
CSV_CRUDO = DIR_DATOS / "hogares_sinteticos.csv"
CSV_LIMPIO = DIR_DATOS / "hogares_limpios.csv"

SEMILLA = 42
N_REGISTROS = 3000

NUMERICAS = [
    "edad_meses",
    "ingreso_mensual",
    "integrantes_hogar",
    "ninos_menores_5",
    "gasto_alimentos_pct",
    "diversidad_dietaria",
    "comidas_dia",
    "escolaridad_madre",
    "enfermedades_3m",
    "ingreso_per_capita",
]
BINARIAS = [
    "acceso_agua_segura",
    "saneamiento_adecuado",
    "controles_crecimiento_al_dia",
    "lactancia_exclusiva_6m",
    "recibe_apoyo_alimentario",
    "jefatura_femenina",
]
CATEGORICAS = ["sexo", "zona"]

OBJ_NUT = "riesgo_nutricional"
OBJ_INSEG = "inseguridad_alimentaria"

ETIQUETAS = {
    OBJ_NUT: {0: "Riesgo bajo", 1: "Riesgo moderado", 2: "Riesgo alto"},
    OBJ_INSEG: {
        0: "Seguridad alimentaria",
        1: "Inseguridad leve",
        2: "Inseguridad moderada o severa",
    },
}
ETIQUETAS_CORTAS = {
    OBJ_NUT: ["Bajo", "Moderado", "Alto"],
    OBJ_INSEG: ["Segura", "Leve", "Mod./Sev."],
}
TITULOS = {
    OBJ_NUT: "Riesgo nutricional infantil",
    OBJ_INSEG: "Inseguridad alimentaria del hogar",
}

NOMBRES_MODELOS = ["Regresión logística", "Random Forest", "XGBoost"]
ARCHIVO_MODELO = {
    "Regresión logística": "logistica",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
