"""
busqueda.py · Optuna (TPE) maximizando F1(clase 0) en VALIDACIÓN.

Config del grupo: un espacio de búsqueda por modelo (continuo/entero/categórico),
sampler TPE con semilla fija, N trials por modelo. Cada trial entrena en TRAIN
y mide F1(cls0) en VAL. Optuna maximiza ese valor a lo largo de los trials.
"""
from __future__ import annotations

import optuna
import pandas as pd

from . import config as C3, metricas, modelos

optuna.logging.set_verbosity(optuna.logging.WARNING)


def espacio(nombre: str, trial: optuna.Trial) -> dict:
    if nombre == "Logistic Regression":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"])}
    if nombre == "SGD":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"])}
    if nombre == "linearSVM":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    if nombre in ("Random Forest", "Extra Trees"):
        return {"n_estimators": trial.suggest_int("n_estimators", 150, 350),
                "max_depth": trial.suggest_int("max_depth", 2, 14),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])}
    if nombre == "Hist Gradient Boosting":
        return {"max_iter": trial.suggest_int("max_iter", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 150),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-7, 10, log=True)}
    if nombre == "LightGBM":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 64),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)}
    if nombre == "CatBoost":
        return {"iterations": trial.suggest_int("iterations", 120, 250),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "depth": trial.suggest_int("depth", 2, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True)}
    raise ValueError(nombre)


def optimizar(nombre, datos, variables, tr, va, n_trials=C3.N_TRIALS_OPTUNA):
    # subsample de TRAIN para la BÚSQUEDA (memoria/tiempo); el refit final
    # del orquestador usa TRAIN completo con los mejores params.
    tr_s = tr.sample(n=min(len(tr), 15000), random_state=C3.SEMILLA)
    Xtr, ytr = tr_s[variables], tr_s[C3.OBJETIVO]
    Xva, yva = va[variables], va[C3.OBJETIVO]

    def objetivo(trial):
        params = espacio(nombre, trial)
        pipe = modelos.construir_candidato(nombre, datos, variables, params)
        pipe.fit(Xtr, ytr)
        return metricas.f1_clase0(yva, modelos.predice_proba(pipe, Xva))

    est = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=C3.SEMILLA),
                              study_name=f"{nombre}_f1cls0")
    est.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    hist = pd.DataFrame([{"modelo": nombre, "trial": t.number, "f1_cls0_val": t.value,
                          **t.params} for t in est.trials if t.value is not None])
    return {"modelo": nombre, "mejores_params": est.best_params,
            "mejor_f1_cls0_val": round(est.best_value, 4), "historial": hist}
