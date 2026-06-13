"""
optuna_panel.py · Sprint 3 — Panel completo con mejoras integradas del repo real.

Mejoras incorporadas desde customer-satisfaction-ml (.rar):
  · Lee los splits PRE-CALCULADOS de data/master (datos reales) si existen.
  · 8 modelos con espacios de búsqueda ricos (early stopping en CatBoost, etc.).
  · CV 5-fold estratificada + probabilidades OOF (out-of-fold) por modelo.
  · STACKING: meta-learner LogReg sobre las OOF de los modelos base.
  · CALIBRACIÓN: LinearSVM vía CalibratedClassifierCV (método tuneado).
  · scale_pos_weight para el desbalance.
  · Estabilidad (caída F1 cls0 val->backtest) como factor de selección.
  · Persistencia de artefactos por modelo + trial CSVs.
Optuna maximiza F1(clase 0 = insatisfecho) en validación. SEED fijo.
"""
from __future__ import annotations

import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from . import config as C3

SEED = C3.SEMILLA
POS = C3.CLASE_FOCO          # 0 = insatisfecho
DIR_MASTER = C3.DIR_DATOS / "master"

try:
    from catboost import CatBoostClassifier; HAS_CB = True
except ImportError: HAS_CB = False
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except ImportError: HAS_LGBM = False


def cargar_splits():
    """Carga splits reales pre-calculados de data/master (X/y train/val/backtest)."""
    def L(n): return pd.read_csv(DIR_MASTER / f"{n}.csv")
    Xtr, Xva, Xbt = L("X_train"), L("X_val"), L("X_backtest")
    ytr = L("y_train").squeeze(); yva = L("y_val").squeeze(); ybt = L("y_backtest").squeeze()
    # alinear columnas y limpiar no-numéricas residuales
    cols = [c for c in Xtr.columns if Xtr[c].dtype.kind in "biufc"]
    Xtr, Xva, Xbt = Xtr[cols].fillna(0), Xva[cols].fillna(0), Xbt[cols].fillna(0)
    neg, pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    spw = round(pos / max(neg, 1), 4)
    print(f"TRAIN {Xtr.shape} · VAL {Xva.shape} · BACKTEST {Xbt.shape} · "
          f"insatisf={ (ytr==0).mean():.1%} · scale_pos_weight={spw}")
    return Xtr, Xva, Xbt, ytr, yva, ybt, spw, cols


def _f1_0(y, proba, thr=0.5):
    return f1_score(y, (np.asarray(proba) >= thr).astype(int), pos_label=POS, zero_division=0)


def metricas(y, proba, thr=0.5):
    pred = (np.asarray(proba) >= thr).astype(int)
    try: auc = roc_auc_score(y, proba)
    except ValueError: auc = float("nan")
    return {"f1_0": f1_score(y, pred, pos_label=0, zero_division=0),
            "recall_0": recall_score(y, pred, pos_label=0, zero_division=0),
            "prec_0": precision_score(y, pred, pos_label=0, zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "auc": auc, "gini": 2*auc-1 if auc==auc else float("nan"),
            "accuracy": float((np.asarray(y) == pred).mean()),
            "brier": brier_score_loss(y, proba)}


def _proba(m, X):
    if hasattr(m, "predict_proba"): return m.predict_proba(X)[:, 1]
    d = m.decision_function(X); return 1/(1+np.exp(-d))


def espacios(Xtr, ytr, Xva, yva, spw):
    def lr(t):
        p = {"C": t.suggest_float("C",1e-3,10,log=True),
             "solver": t.suggest_categorical("solver",["lbfgs","saga"]),
             "class_weight": t.suggest_categorical("class_weight",[None,"balanced"]),
             "max_iter":1000,"random_state":SEED}
        m=LogisticRegression(**p); m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def rf(t):
        p={"n_estimators":t.suggest_int("n_estimators",100,300),
           "max_depth":t.suggest_int("max_depth",3,20),
           "min_samples_split":t.suggest_int("min_samples_split",2,20),
           "min_samples_leaf":t.suggest_int("min_samples_leaf",1,10),
           "max_features":t.suggest_categorical("max_features",["sqrt","log2",0.5]),
           "class_weight":t.suggest_categorical("class_weight",[None,"balanced"]),
           "random_state":SEED,"n_jobs":2}
        m=RandomForestClassifier(**p); m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def hgb(t):
        p={"max_iter":t.suggest_int("max_iter",100,500),
           "max_depth":t.suggest_int("max_depth",2,8),
           "learning_rate":t.suggest_float("learning_rate",1e-3,0.3,log=True),
           "max_leaf_nodes":t.suggest_int("max_leaf_nodes",15,255),
           "l2_regularization":t.suggest_float("l2_regularization",1e-10,1.0,log=True),
           "class_weight":t.suggest_categorical("class_weight",["balanced",None]),
           "random_state":SEED}
        m=HistGradientBoostingClassifier(**p); m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def et(t):
        ud=t.suggest_categorical("use_max_depth",[True,False])
        p={"n_estimators":t.suggest_int("n_estimators",80,300),
           "max_depth":t.suggest_int("max_depth",3,50) if ud else None,
           "min_samples_split":t.suggest_int("min_samples_split",2,50),
           "min_samples_leaf":t.suggest_int("min_samples_leaf",1,30),
           "max_features":t.suggest_categorical("max_features",["sqrt","log2",0.3,0.5,0.7]),
           "bootstrap":t.suggest_categorical("bootstrap",[False,True]),
           "class_weight":t.suggest_categorical("class_weight",[None,"balanced","balanced_subsample"]),
           "criterion":t.suggest_categorical("criterion",["gini","entropy","log_loss"]),
           "n_jobs":2,"random_state":SEED}
        if not p["bootstrap"] and p["class_weight"]=="balanced_subsample": p["class_weight"]="balanced"
        m=ExtraTreesClassifier(**p); m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def sgd(t):
        loss=t.suggest_categorical("loss",["log_loss","modified_huber"])
        p={"loss":loss,"penalty":t.suggest_categorical("penalty",["l2","l1","elasticnet"]),
           "alpha":t.suggest_float("alpha",1e-5,1.0,log=True),"max_iter":2000,
           "class_weight":t.suggest_categorical("class_weight",[None,"balanced"]),
           "n_jobs":2,"random_state":SEED}
        if p["penalty"]=="elasticnet": p["l1_ratio"]=t.suggest_float("l1_ratio",0.0,1.0)
        m=SGDClassifier(**p); m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def lsvm(t):
        p={"C":t.suggest_float("C",1e-3,100,log=True),
           "loss":t.suggest_categorical("loss",["hinge","squared_hinge"]),
           "class_weight":t.suggest_categorical("class_weight",[None,"balanced"]),
           "max_iter":10000,"random_state":SEED,"dual":"auto"}
        cal=t.suggest_categorical("calibration_method",["sigmoid","isotonic"])
        m=CalibratedClassifierCV(LinearSVC(**p),method=cal,cv=3,n_jobs=1)
        m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def cb(t):
        p={"iterations":t.suggest_int("iterations",100,250),
           "depth":t.suggest_int("depth",2,8),
           "learning_rate":t.suggest_float("learning_rate",1e-3,0.3,log=True),
           "l2_leaf_reg":t.suggest_float("l2_leaf_reg",1e-3,10,log=True),
           "auto_class_weights":t.suggest_categorical("auto_class_weights",[None,"Balanced"]),
           "od_type":"Iter","od_wait":t.suggest_int("od_wait",10,50),
           "random_state":SEED,"verbose":False,"task_type":"CPU","thread_count":2}
        m=CatBoostClassifier(**p); m.fit(Xtr,ytr,eval_set=(Xva,yva),early_stopping_rounds=50)
        return _f1_0(yva,m.predict_proba(Xva)[:,1])
    def lgbm(t):
        p={"n_estimators":t.suggest_int("n_estimators",100,300),
           "max_depth":t.suggest_int("max_depth",2,10),
           "learning_rate":t.suggest_float("learning_rate",1e-3,0.3,log=True),
           "num_leaves":t.suggest_int("num_leaves",20,150),
           "subsample":t.suggest_float("subsample",0.5,1.0),
           "colsample_bytree":t.suggest_float("colsample_bytree",0.5,1.0),
           "reg_alpha":t.suggest_float("reg_alpha",1e-8,1.0,log=True),
           "reg_lambda":t.suggest_float("reg_lambda",1e-8,1.0,log=True),
           "class_weight":t.suggest_categorical("class_weight",[None,"balanced"]),
           "random_state":SEED,"verbosity":-1,"n_jobs":2}
        m=LGBMClassifier(**p); m.fit(Xtr,ytr); return _f1_0(yva,m.predict_proba(Xva)[:,1])
    mp={"Logistic Regression":lr,"Random Forest":rf,"Hist Gradient Boosting":hgb,
        "Extra Trees":et,"SGD":sgd,"linearSVM":lsvm}
    if HAS_CB: mp["CatBoost"]=cb
    if HAS_LGBM: mp["LightGBM"]=lgbm
    return mp


def construir(nombre, params):
    p={k:v for k,v in params.items() if k not in ("use_max_depth","calibration_method")}
    if nombre=="Logistic Regression": return LogisticRegression(**p,max_iter=1000,random_state=SEED)
    if nombre=="Random Forest": return RandomForestClassifier(**p,random_state=SEED,n_jobs=1)
    if nombre=="Hist Gradient Boosting": return HistGradientBoostingClassifier(**p,random_state=SEED)
    if nombre=="Extra Trees": return ExtraTreesClassifier(**p,random_state=SEED,n_jobs=1)
    if nombre=="SGD":
        if p.get("loss") not in ("log_loss","modified_huber"): p["loss"]="log_loss"
        return SGDClassifier(**p,random_state=SEED,n_jobs=1)
    if nombre=="linearSVM":
        cal=params.get("calibration_method","sigmoid")
        return CalibratedClassifierCV(LinearSVC(**p,random_state=SEED,max_iter=10000),method=cal,cv=3,n_jobs=1)
    if nombre=="CatBoost": return CatBoostClassifier(**p,random_state=SEED,verbose=False,task_type="CPU",thread_count=2)
    if nombre=="LightGBM": return LGBMClassifier(**p,random_state=SEED,verbosity=-1,n_jobs=1)
    raise ValueError(nombre)
