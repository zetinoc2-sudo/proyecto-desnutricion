import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import generar_datos
from config import (
    ARCHIVO_MODELO,
    BINARIAS,
    CSV_CRUDO,
    CSV_LIMPIO,
    DIR_DATOS,
    DIR_GRAFICOS,
    DIR_MODELOS,
    ETIQUETAS,
    ETIQUETAS_CORTAS,
    NOMBRES_MODELOS,
    NUMERICAS,
    OBJ_INSEG,
    OBJ_NUT,
    SEMILLA,
    TITULOS,
)
from preparar_datos import columnas_modelo, construir_preprocesador, limpiar

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def crear_modelos():
    return {
        "Regresión logística": LogisticRegression(max_iter=3000, random_state=SEMILLA),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=SEMILLA, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=SEMILLA,
            n_jobs=1,
        ),
    }


def armar(clasificador):
    return Pipeline([("prep", construir_preprocesador()), ("clf", clasificador)])


def ajustar(pipe, X, y):
    pesos = compute_sample_weight("balanced", y)
    pipe.fit(X, y, clf__sample_weight=pesos)
    return pipe


def validacion_cruzada(nombre, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    puntajes = []
    for tr, va in cv.split(X, y):
        pipe = ajustar(armar(crear_modelos()[nombre]), X.iloc[tr], y.iloc[tr])
        puntajes.append(f1_score(y.iloc[va], pipe.predict(X.iloc[va]), average="macro"))
    return float(np.mean(puntajes))


def guardar_grafico(nombre):
    plt.tight_layout()
    plt.savefig(DIR_GRAFICOS / nombre, dpi=130)
    plt.close()


def exploracion(df):
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for eje, objetivo in zip(ejes, [OBJ_NUT, OBJ_INSEG]):
        conteo = df[objetivo].value_counts(normalize=True).sort_index() * 100
        eje.bar(ETIQUETAS_CORTAS[objetivo], conteo.values, color=["#2E75B6", "#F2B134", "#C0392B"])
        eje.set_title(TITULOS[objetivo])
        eje.set_ylabel("Porcentaje de hogares (%)")
    guardar_grafico("distribucion_objetivos.png")

    columnas = NUMERICAS + BINARIAS + [OBJ_NUT, OBJ_INSEG]
    plt.figure(figsize=(11, 9))
    sns.heatmap(df[columnas].corr(), cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Matriz de correlación")
    guardar_grafico("correlacion.png")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for eje, objetivo in zip(ejes, [OBJ_NUT, OBJ_INSEG]):
        tabla = pd.crosstab(df["zona"], df[objetivo], normalize="index") * 100
        tabla.columns = ETIQUETAS_CORTAS[objetivo]
        tabla.plot(kind="bar", stacked=True, ax=eje, color=["#2E75B6", "#F2B134", "#C0392B"])
        eje.set_title(f"{TITULOS[objetivo]} por zona")
        eje.set_ylabel("%")
        eje.set_xlabel("")
        eje.tick_params(axis="x", rotation=0)
    guardar_grafico("riesgo_por_zona.png")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x=OBJ_INSEG, y="ingreso_per_capita", ax=ejes[0], color="#9CC3E6")
    ejes[0].set_xticklabels(ETIQUETAS_CORTAS[OBJ_INSEG])
    ejes[0].set_xlabel("Inseguridad alimentaria")
    ejes[0].set_ylabel("Ingreso per cápita (USD)")
    sns.boxplot(data=df, x=OBJ_NUT, y="diversidad_dietaria", ax=ejes[1], color="#F5CBA7")
    ejes[1].set_xticklabels(ETIQUETAS_CORTAS[OBJ_NUT])
    ejes[1].set_xlabel("Riesgo nutricional")
    ejes[1].set_ylabel("Diversidad dietaria")
    guardar_grafico("variables_clave.png")


def entrenar_objetivo(df, objetivo):
    X = df[columnas_modelo()]
    y = df[objetivo]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEMILLA
    )
    X_tr = X_tr.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)

    filas = []
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    for eje, nombre in zip(ejes, NOMBRES_MODELOS):
        f1_cv = validacion_cruzada(nombre, X_tr, y_tr)
        pipe = ajustar(armar(crear_modelos()[nombre]), X_tr, y_tr)
        pred = pipe.predict(X_te)
        filas.append(
            {
                "objetivo": objetivo,
                "modelo": nombre,
                "exactitud": accuracy_score(y_te, pred),
                "precision_macro": precision_score(y_te, pred, average="macro"),
                "recall_macro": recall_score(y_te, pred, average="macro"),
                "f1_macro": f1_score(y_te, pred, average="macro"),
                "f1_validacion_cruzada": f1_cv,
            }
        )
        ConfusionMatrixDisplay.from_predictions(
            y_te,
            pred,
            display_labels=ETIQUETAS_CORTAS[objetivo],
            ax=eje,
            colorbar=False,
            cmap="Blues",
        )
        eje.set_title(nombre)
        joblib.dump(pipe, DIR_MODELOS / f"{objetivo}_{ARCHIVO_MODELO[nombre]}.joblib")

        if nombre == "Random Forest":
            nombres = [
                n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()
            ]
            importancia = pd.Series(pipe.named_steps["clf"].feature_importances_, index=nombres)
            importancia = importancia.sort_values().tail(12)
            graf_imp = importancia
    fig.suptitle(TITULOS[objetivo])
    guardar_grafico(f"matrices_{objetivo}.png")

    plt.figure(figsize=(8, 5))
    graf_imp.plot(kind="barh", color="#2E75B6")
    plt.title(f"Importancia de variables (Random Forest)\n{TITULOS[objetivo]}")
    plt.xlabel("Importancia")
    guardar_grafico(f"importancia_{objetivo}.png")

    return pd.DataFrame(filas)


def agrupar(df):
    X = df[columnas_modelo()]
    prep = construir_preprocesador()
    matriz = prep.fit_transform(X)

    ks = list(range(2, 9))
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(matriz)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(matriz, km.labels_, sample_size=2000, random_state=SEMILLA))

    candidatos = {k: s for k, s in zip(ks, siluetas) if 3 <= k <= 6}
    k_final = max(candidatos, key=candidatos.get)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(ks, inercias, marker="o", color="#2E75B6")
    ejes[0].set_title("Método del codo")
    ejes[0].set_xlabel("Número de grupos (k)")
    ejes[0].set_ylabel("Inercia")
    ejes[1].plot(ks, siluetas, marker="o", color="#C55A11")
    ejes[1].axvline(k_final, linestyle="--", color="gray")
    ejes[1].set_title("Coeficiente de silueta")
    ejes[1].set_xlabel("Número de grupos (k)")
    guardar_grafico("kmeans_seleccion_k.png")

    km = KMeans(n_clusters=k_final, n_init=10, random_state=SEMILLA).fit(matriz)
    df = df.copy()
    df["grupo_bruto"] = km.labels_
    ranking = df.groupby("grupo_bruto")["ingreso_per_capita"].mean().sort_values().index
    orden = {int(c): i + 1 for i, c in enumerate(ranking)}
    df["grupo"] = df["grupo_bruto"].map(orden)

    perfil = df.groupby("grupo").agg(
        hogares=("grupo", "size"),
        ingreso_per_capita=("ingreso_per_capita", "mean"),
        diversidad_dietaria=("diversidad_dietaria", "mean"),
        comidas_dia=("comidas_dia", "mean"),
        enfermedades_3m=("enfermedades_3m", "mean"),
        acceso_agua_segura=("acceso_agua_segura", "mean"),
        saneamiento_adecuado=("saneamiento_adecuado", "mean"),
        recibe_apoyo_alimentario=("recibe_apoyo_alimentario", "mean"),
    )
    perfil["pct_riesgo_nutricional_alto"] = df.groupby("grupo")[OBJ_NUT].apply(lambda s: (s == 2).mean())
    perfil["pct_inseguridad_mod_severa"] = df.groupby("grupo")[OBJ_INSEG].apply(lambda s: (s == 2).mean())
    perfil = perfil.round(2)
    perfil.to_csv(DIR_DATOS / "perfiles_grupos.csv")

    coordenadas = PCA(n_components=2, random_state=SEMILLA).fit_transform(matriz)
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(
        x=coordenadas[:, 0],
        y=coordenadas[:, 1],
        hue=df["grupo"],
        palette="viridis",
        s=14,
        alpha=0.7,
    )
    plt.title("Grupos de hogares (K-Means) proyectados con PCA")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    guardar_grafico("kmeans_grupos.png")

    joblib.dump({"prep": prep, "kmeans": km, "orden": orden, "k": k_final}, DIR_MODELOS / "kmeans.joblib")
    return k_final, perfil


def main():
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)

    if not CSV_CRUDO.exists():
        generar_datos.guardar()
    crudo = pd.read_csv(CSV_CRUDO)
    print(f"Filas originales: {len(crudo)} | duplicados: {crudo.duplicated().sum()}")
    print("Valores faltantes:\n", crudo.isna().sum()[crudo.isna().sum() > 0].to_string())

    df = limpiar(crudo)
    df.to_csv(CSV_LIMPIO, index=False)
    print(f"Filas después de la limpieza: {len(df)}")

    exploracion(df)

    resultados = pd.concat(
        [entrenar_objetivo(df, OBJ_NUT), entrenar_objetivo(df, OBJ_INSEG)], ignore_index=True
    ).round(4)
    resultados.to_csv(DIR_DATOS / "metricas_modelos.csv", index=False)
    print("\n", resultados.to_string(index=False))

    mejores = {}
    for objetivo in [OBJ_NUT, OBJ_INSEG]:
        sub = resultados[resultados["objetivo"] == objetivo]
        mejores[objetivo] = sub.loc[sub["f1_validacion_cruzada"].idxmax(), "modelo"]
    (DIR_MODELOS / "mejores.json").write_text(json.dumps(mejores, ensure_ascii=False), encoding="utf-8")
    print("\nMejor modelo por objetivo:", mejores)

    k, perfil = agrupar(df)
    print(f"\nK-Means: k = {k}\n", perfil.to_string())


if __name__ == "__main__":
    main()
