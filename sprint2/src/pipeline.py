"""
============================================================================
 pipeline.py  ·  Paso 5-6 · Pipeline reproducible (sklearn) + persistencia
============================================================================

Ensambla un Pipeline de scikit-learn que encapsula:
    - Imputación de numéricas + escalado (StandardScaler)
    - Imputación de categóricas + One-Hot Encoding
    - Modelo clasificador (Random Forest como baseline del Sprint 2)

El objeto resultante es 100% reproducible y serializable a .pkl, de modo
que el mismo artefacto se reutiliza para puntuar datos nuevos cada mes
(paso "Aplicar el Pickle" del flujo de trabajo).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import features as F


# ════════════════════════════════════════════════════════════════════════
#  PREPROCESADOR  (ColumnTransformer)
# ════════════════════════════════════════════════════════════════════════
def construir_preprocesador(variables: list[str]) -> ColumnTransformer:
    """Construye el ColumnTransformer según el tipo de cada variable."""
    numericas = [c for c in variables if c in F.VARIABLES_NUMERICAS]
    binarias = [c for c in variables if c in F.VARIABLES_BINARIAS]
    categoricas = [c for c in variables if c in F.VARIABLES_CATEGORICAS]

    # Numéricas: imputar con mediana y escalar
    rama_numerica = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # Binarias: solo imputar (ya vienen en 0/1, no hace falta escalar)
    rama_binaria = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    # Categóricas: imputar a "OTHER" y codificar con One-Hot
    rama_categorica = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=C.ETIQUETA_CATEGORIA_RARA)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                 sparse_output=False)),
    ])

    transformadores = []
    if numericas:
        transformadores.append(("num", rama_numerica, numericas))
    if binarias:
        transformadores.append(("bin", rama_binaria, binarias))
    if categoricas:
        transformadores.append(("cat", rama_categorica, categoricas))

    return ColumnTransformer(transformadores, remainder="drop",
                             verbose_feature_names_out=False)


# ════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════
def construir_pipeline(variables: list[str], modelo=None) -> Pipeline:
    """Pipeline completo: preprocesador + modelo.

    Modelo por defecto: Random Forest (baseline reproducible del Sprint 2).
    En el Sprint 3 se sustituye/optimiza este estimador (Optuna, etc.).
    """
    if modelo is None:
        modelo = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=30,
            class_weight="balanced", n_jobs=-1, random_state=C.SEMILLA,
        )
    preprocesador = construir_preprocesador(variables)
    return Pipeline([("preprocessor", preprocesador), ("model", modelo)])


# ════════════════════════════════════════════════════════════════════════
#  PERSISTENCIA  (versionado del pickle)
# ════════════════════════════════════════════════════════════════════════
def guardar_pipeline(pipe: Pipeline, variables: list[str], metadatos: dict,
                     nombre: str = "satisfaction_pipeline") -> Path:
    """Serializa pipeline + variables + metadatos en un único .pkl versionado."""
    artefacto = {
        "pipeline": pipe,
        "features": variables,
        "metadata": metadatos,
        "pipeline_version": C.VERSION_PIPELINE,
    }
    ruta = C.DIR_MODELOS / f"{nombre}_v{C.VERSION_PIPELINE}.pkl"
    with open(ruta, "wb") as f:
        pickle.dump(artefacto, f)
    print(f"Pipeline guardado: {ruta}")
    return ruta


def cargar_pipeline(ruta: Path | str) -> dict:
    """Carga el artefacto .pkl (pipeline + variables + metadatos)."""
    with open(ruta, "rb") as f:
        return pickle.load(f)


def predecir_datos_nuevos(artefacto: dict, datos_nuevos: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pickle a datos nuevos (paso de predicción mensual).

    `datos_nuevos` debe haber pasado por features.construir_variables() y por
    la limpieza. Devuelve el DataFrame con columnas de probabilidad y predicción.
    """
    pipe = artefacto["pipeline"]
    variables = artefacto["features"]
    proba = pipe.predict_proba(datos_nuevos[variables])[:, 1]
    salida = datos_nuevos.copy()
    salida["proba_satisfecho"] = proba
    salida["pred_satisfecho"] = (proba >= 0.5).astype(int)
    return salida
