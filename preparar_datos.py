import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BINARIAS, CATEGORICAS, NUMERICAS


def agregar_derivadas(df):
    df = df.copy()
    df["ingreso_per_capita"] = df["ingreso_mensual"] / df["integrantes_hogar"]
    return df


def limpiar(df):
    df = df.drop_duplicates().reset_index(drop=True).copy()
    df["ingreso_mensual"] = df["ingreso_mensual"].where(df["ingreso_mensual"].between(20, 8000))
    for columna in ["ingreso_mensual", "escolaridad_madre", "gasto_alimentos_pct", "diversidad_dietaria"]:
        df[columna] = df[columna].fillna(df[columna].median())
    df["escolaridad_madre"] = df["escolaridad_madre"].round().astype(int)
    df["diversidad_dietaria"] = df["diversidad_dietaria"].round().astype(int)
    df = df[df["edad_meses"].between(0, 59)]
    return agregar_derivadas(df).reset_index(drop=True)


def columnas_modelo():
    return NUMERICAS + BINARIAS + CATEGORICAS


def construir_preprocesador():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS + BINARIAS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ]
    )


def cargar_limpios(ruta):
    return pd.read_csv(ruta)
