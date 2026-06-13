"""metricas.py · F1 de la clase 0 y métricas completas por modelo."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C3


def f1_clase0(y_real, y_proba, umbral: float = 0.5) -> float:
    pred = (np.asarray(y_proba) >= umbral).astype(int)
    return f1_score(y_real, pred, pos_label=C3.CLASE_FOCO, zero_division=0)


def metricas_completas(y_real, y_proba, umbral: float = 0.5) -> dict:
    y_real = np.asarray(y_real)
    pred = (np.asarray(y_proba) >= umbral).astype(int)
    try:
        auc = roc_auc_score(y_real, y_proba)
    except ValueError:
        auc = float("nan")
    return {
        "f1_cls0":  f1_score(y_real, pred, pos_label=0, zero_division=0),
        "rec_cls0": recall_score(y_real, pred, pos_label=0, zero_division=0),
        "pre_cls0": precision_score(y_real, pred, pos_label=0, zero_division=0),
        "f1_cls1":  f1_score(y_real, pred, pos_label=1, zero_division=0),
        "rec_cls1": recall_score(y_real, pred, pos_label=1, zero_division=0),
        "pre_cls1": precision_score(y_real, pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_real, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_real, pred),
        "auc_roc":  auc,
        "gini":     2 * auc - 1 if auc == auc else float("nan"),
        "brier":    brier_score_loss(y_real, y_proba),
    }
