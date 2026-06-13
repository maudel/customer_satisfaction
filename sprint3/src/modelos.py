"""
modelos.py · Panel de 8 modelos del Sprint 3 + preprocesador propio.

El preprocesador clasifica por dtype (no por listas fijas del Sprint 2), así
las variables de interacción del Sprint 3 entran sin fricción. Mismas ramas:
numéricas -> imputar mediana + escalar; categóricas -> imputar + One-Hot.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from . import config as C3

warnings.filterwarnings("ignore")

# Panel completo (orden = el del deck del grupo)
PANEL = ["Logistic Regression", "LightGBM", "CatBoost", "Random Forest",
         "Extra Trees", "Hist Gradient Boosting", "linearSVM", "SGD"]


def _tipos(datos: pd.DataFrame, variables: list[str]):
    num = [c for c in variables if datos[c].dtype.kind in "biufc"]
    cat = [c for c in variables if c not in num]
    return num, cat


def construir_preprocesador(datos: pd.DataFrame, variables: list[str]) -> ColumnTransformer:
    num, cat = _tipos(datos, variables)
    rama_num = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    rama_cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="OTHER")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    return ColumnTransformer([("num", rama_num, num), ("cat", rama_cat, cat)],
                             remainder="drop")


def _modelo(nombre: str, params: dict | None = None):
    p = dict(params or {})
    s = C3.SEMILLA
    if nombre == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s, **p)
    if nombre == "SGD":
        return SGDClassifier(loss="log_loss", class_weight="balanced", random_state=s, **p)
    if nombre == "linearSVM":
        # LinearSVC no da probas; se calibra fuera. Aquí basta para F1.
        return LinearSVC(class_weight="balanced", random_state=s, **p)
    if nombre == "Random Forest":
        base = dict(n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=2,
                    class_weight="balanced", random_state=s)
        base.update(p); return RandomForestClassifier(**base)
    if nombre == "Extra Trees":
        base = dict(n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=2,
                    class_weight="balanced_subsample", random_state=s)
        base.update(p); return ExtraTreesClassifier(**base)
    if nombre == "Hist Gradient Boosting":
        base = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                    min_samples_leaf=30, random_state=s, class_weight="balanced")
        base.update(p); return HistGradientBoostingClassifier(**base)
    if nombre == "LightGBM":
        from lightgbm import LGBMClassifier
        base = dict(n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
                    class_weight="balanced", random_state=s, n_jobs=2, verbose=-1)
        base.update(p); return LGBMClassifier(**base)
    if nombre == "CatBoost":
        from catboost import CatBoostClassifier
        base = dict(iterations=200, learning_rate=0.05, depth=5, random_seed=s,
                    auto_class_weights="Balanced", verbose=0, allow_writing_files=False,
                    thread_count=2)
        base.update(p); return CatBoostClassifier(**base)
    raise ValueError(nombre)


def construir_candidato(nombre: str, datos: pd.DataFrame, variables: list[str],
                        params: dict | None = None) -> Pipeline:
    return Pipeline([("prep", construir_preprocesador(datos, variables)),
                     ("modelo", _modelo(nombre, params))])


def predice_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """P(clase 1). linearSVM no da proba -> se usa la función de decisión escalada."""
    modelo = pipe.named_steps["modelo"]
    if hasattr(modelo, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    d = pipe.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))   # sigmoide sobre el margen
