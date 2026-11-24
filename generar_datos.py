import numpy as np
import pandas as pd

from config import CSV_CRUDO, DIR_DATOS, N_REGISTROS, SEMILLA


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def generar(n=N_REGISTROS, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)

    zona = rng.choice(["Urbana", "Rural"], n, p=[0.62, 0.38])
    rural = (zona == "Rural").astype(int)
    sexo = rng.choice(["Femenino", "Masculino"], n)
    edad = rng.integers(6, 60, n)
    integrantes = np.clip(rng.poisson(3.2, n) + 2, 2, 10)
    ninos = np.clip(rng.poisson(0.9, n) + 1, 1, np.minimum(integrantes - 1, 4))

    ingreso = rng.lognormal(np.log(420), 0.55, n) * np.where(rural == 1, 0.72, 1.0)
    ingreso = np.round(ingreso, 2)
    per_capita = ingreso / integrantes

    escolaridad = np.clip(
        np.round(rng.normal(8.5 - 2.2 * rural + 1.5 * np.log(ingreso / 420), 3.2)), 0, 16
    ).astype(int)
    agua = (rng.random(n) < np.where(rural == 1, 0.70, 0.93)).astype(int)
    saneamiento = (rng.random(n) < np.where(rural == 1, 0.62, 0.90)).astype(int)
    lactancia = (rng.random(n) < np.clip(0.5 + 0.03 * (escolaridad - 8), 0.1, 0.9)).astype(int)

    carga = (
        0.8 * (-np.log(per_capita / 100))
        + 0.45 * rural
        + 0.35 * (1 - agua)
        + 0.30 * (1 - saneamiento)
        + 0.25 * (ninos - 1)
        - 0.04 * (escolaridad - 8)
    )

    apoyo = (rng.random(n) < sigmoide(1.2 * carga - 1.6)).astype(int)
    diversidad = np.clip(np.round(7.5 - 1.4 * carga + rng.normal(0, 1.3, n)), 0, 12).astype(int)
    comidas = np.clip(np.round(3.2 - 0.6 * carga + rng.normal(0, 0.6, n)), 1, 5).astype(int)
    gasto = np.round(np.clip(35 + 14 * carga + rng.normal(0, 7, n), 10, 85), 1)
    enfermedades = np.clip(
        rng.poisson(np.clip(1.0 + 0.45 * carga + 0.25 * (1 - agua), 0.1, None)), 0, 6
    )
    controles = (rng.random(n) < sigmoide(1.1 - 0.7 * carga - 0.01 * (edad - 30))).astype(int)
    jefatura = (rng.random(n) < 0.28 + 0.05 * (carga > 0.5)).astype(int)

    puntaje_inseg = (
        carga
        + 0.15 * (7.5 - diversidad)
        + 0.25 * (3.2 - comidas)
        - 0.20 * apoyo
        + rng.normal(0, 0.55, n)
    )
    puntaje_nut = (
        0.6 * carga
        + 0.35 * (enfermedades - 1)
        - 0.45 * controles
        - 0.35 * lactancia
        + 0.12 * (7.5 - diversidad)
        + 0.01 * (edad - 30)
        + rng.normal(0, 0.6, n)
    )
    inseguridad = np.digitize(puntaje_inseg, np.quantile(puntaje_inseg, [0.5, 0.8]))
    nutricional = np.digitize(puntaje_nut, np.quantile(puntaje_nut, [0.6, 0.88]))

    df = pd.DataFrame(
        {
            "edad_meses": edad,
            "sexo": sexo,
            "zona": zona,
            "integrantes_hogar": integrantes,
            "ninos_menores_5": ninos,
            "ingreso_mensual": ingreso,
            "gasto_alimentos_pct": gasto,
            "escolaridad_madre": escolaridad,
            "jefatura_femenina": jefatura,
            "acceso_agua_segura": agua,
            "saneamiento_adecuado": saneamiento,
            "diversidad_dietaria": diversidad,
            "comidas_dia": comidas,
            "enfermedades_3m": enfermedades,
            "controles_crecimiento_al_dia": controles,
            "lactancia_exclusiva_6m": lactancia,
            "recibe_apoyo_alimentario": apoyo,
            "riesgo_nutricional": nutricional,
            "inseguridad_alimentaria": inseguridad,
        }
    )

    for columna, proporcion in [
        ("ingreso_mensual", 0.03),
        ("escolaridad_madre", 0.04),
        ("gasto_alimentos_pct", 0.03),
        ("diversidad_dietaria", 0.02),
    ]:
        df[columna] = df[columna].astype(float)
        posiciones = rng.choice(n, int(n * proporcion), replace=False)
        df.loc[posiciones, columna] = np.nan

    atipicos = rng.choice(n, 8, replace=False)
    df.loc[atipicos, "ingreso_mensual"] = rng.choice([0.0, 15000.0, 22000.0], 8)

    df = pd.concat([df, df.sample(25, random_state=semilla)], ignore_index=True)
    return df.sample(frac=1, random_state=semilla).reset_index(drop=True)


def guardar():
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    df = generar()
    df.to_csv(CSV_CRUDO, index=False)
    return df


if __name__ == "__main__":
    datos = guardar()
    print(f"Base generada: {len(datos)} filas -> {CSV_CRUDO}")
