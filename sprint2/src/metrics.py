"""
============================================================================
 metrics.py  ·  Métricas técnicas y de negocio (versión pipeline)
============================================================================

Calcula el set de métricas exigido por el caso de estudio #1:
    Técnicas : Accuracy, Precision, Recall, F1, ROC-AUC, Gini
    Negocio  : Customer Satisfaction Rate, Unsatisfied Rate,
               Revenue at Risk, % detección de insatisfechos, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from . import config as C


# ════════════════════════════════════════════════════════════════════════
#  MÉTRICAS TÉCNICAS
# ════════════════════════════════════════════════════════════════════════
def metricas_tecnicas(y_real, y_proba, umbral: float = 0.5) -> dict:
    """Métricas de clasificación a partir de las probabilidades del modelo."""
    y_real = np.asarray(y_real).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= umbral).astype(int)

    # El AUC necesita ambas clases presentes; si no, lo dejamos como NaN
    auc = roc_auc_score(y_real, y_proba) if len(np.unique(y_real)) > 1 else float("nan")
    return {
        "accuracy":  round(accuracy_score(y_real, y_pred), 4),
        "precision": round(precision_score(y_real, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_real, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_real, y_pred, zero_division=0), 4),
        "roc_auc":   round(auc, 4),
        "gini":      round(2 * auc - 1, 4) if np.isfinite(auc) else float("nan"),
        "n":         int(len(y_real)),
        "tasa_positivos_real": round(float(y_real.mean()), 4),
    }


# ════════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE NEGOCIO
# ════════════════════════════════════════════════════════════════════════
def metricas_negocio(datos: pd.DataFrame, y_proba=None, umbral: float = 0.5) -> dict:
    """KPIs de negocio sobre una partición.

    Si se pasa `y_proba`, añade KPIs de la utilidad del modelo
    (p.ej. cuántos insatisfechos detecta para intervención temprana).
    """
    salida = {}
    puntaje = datos[C.COL_PUNTAJE_RESENA]
    salida["customer_satisfaction_rate_pct"] = round(float((puntaje >= 4).mean() * 100), 2)
    salida["unsatisfied_customer_rate_pct"]  = round(float((puntaje <= 3).mean() * 100), 2)
    salida["average_review_score"]           = round(float(puntaje.mean()), 3)

    if "delivered_on_time" in datos:
        salida["on_time_delivery_rate_pct"] = round(float(datos["delivered_on_time"].mean() * 100), 2)
    if "delivery_delay_days" in datos:
        tardios = datos.loc[datos["delivery_delay_days"] > 0, "delivery_delay_days"]
        salida["average_delay_days_when_late"] = round(float(tardios.mean()), 2) if len(tardios) else 0.0

    if "payment_value" in datos:
        ingreso_insatisfecho = float(datos.loc[puntaje <= 3, "payment_value"].sum())
        ingreso_total = float(datos["payment_value"].sum())
        salida["revenue_at_risk_brl"] = round(ingreso_insatisfecho, 2)
        salida["revenue_at_risk_pct"] = (
            round(ingreso_insatisfecho / ingreso_total * 100, 2) if ingreso_total else 0.0
        )
        salida["average_order_value_brl"] = round(float(datos["payment_value"].mean()), 2)

    # Utilidad del modelo para intervención temprana sobre insatisfechos
    if y_proba is not None:
        y_real = (datos[C.OBJETIVO]).astype(int).values
        y_pred = (np.asarray(y_proba) >= umbral).astype(int)
        # "alerta" = el modelo predice insatisfecho (clase 0)
        alerta = (y_pred == 0)
        insatisfecho_real = (y_real == 0)
        detectados = int((alerta & insatisfecho_real).sum())
        total_insatisfechos = int(insatisfecho_real.sum())
        salida["insatisfechos_detectados"] = detectados
        salida["insatisfechos_totales"] = total_insatisfechos
        salida["deteccion_insatisfechos_pct"] = (
            round(detectados / total_insatisfechos * 100, 2) if total_insatisfechos else 0.0
        )
        if "payment_value" in datos:
            salida["revenue_en_riesgo_detectado_brl"] = round(
                float(datos.loc[alerta & insatisfecho_real, "payment_value"].sum()), 2)
    return salida


# ════════════════════════════════════════════════════════════════════════
#  TABLA DE LIFT  (priorización por riesgo · la "palanca del umbral")
# ════════════════════════════════════════════════════════════════════════
def tabla_lift(datos: pd.DataFrame, y_proba, percentiles: list[int] | None = None) -> pd.DataFrame:
    """Captura, precisión y lift al contactar solo el top X% más riesgoso.

    Traduce el modelo a la decisión operativa de soporte: "si solo puedo
    contactar al X% de los pedidos, ¿a cuántos insatisfechos reales llego?".

        captura_pct   : % de los insatisfechos del mes que caen en el top X%
        precision_pct : % de contactos del top X% que son insatisfechos reales
        lift          : precisión vs contactar al azar (tasa base del mes)

    El umbral NO es un número fijo del modelo: es una palanca de negocio.
    Con capacidad limitada se sube el corte y cada contacto rinde más.
    """
    percentiles = percentiles if percentiles is not None else C.PERCENTILES_LIFT
    y_real = datos[C.OBJETIVO].astype(int).to_numpy()
    riesgo = 1.0 - np.asarray(y_proba, dtype=float)   # riesgo = P(insatisfecho)
    insatisfecho = (y_real == 0)
    n = len(y_real)
    total_insatisfechos = int(insatisfecho.sum())
    tasa_base = total_insatisfechos / max(n, 1)

    orden = np.argsort(-riesgo)                       # de mayor a menor riesgo
    filas = []
    for pct in percentiles:
        k = max(int(round(n * pct / 100)), 1)
        top = orden[:k]
        detectados = int(insatisfecho[top].sum())
        precision = detectados / k
        filas.append({
            "top_pct": pct,
            "contactados": k,
            "insatisfechos_capturados": detectados,
            "captura_pct": round(detectados / max(total_insatisfechos, 1) * 100, 1),
            "precision_pct": round(precision * 100, 1),
            "lift": round(precision / max(tasa_base, 1e-9), 2),
        })
    return pd.DataFrame(filas)


def matriz_confusion(y_real, y_proba, umbral: float = 0.5) -> pd.DataFrame:
    """Matriz de confusión etiquetada (insatisfecho/satisfecho)."""
    y_real = np.asarray(y_real).astype(int)
    y_pred = (np.asarray(y_proba) >= umbral).astype(int)
    cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["real: insatisfecho", "real: satisfecho"],
        columns=["pred: insatisfecho", "pred: satisfecho"],
    )
