"""
features_extra.py · Variables de interacción y derivadas del Sprint 3.

El Sprint 2 entrega ~25 features base. El Sprint 3 añade las interacciones y
derivadas que aparecen en el análisis de importancia (interaccion_retraso_items,
delay_ratio, freight_ratio, dispatch_time_hours, dimensiones de producto, etc.),
de modo que la importancia de variables sea comparable entre todos los modelos
del panel y se pueda auditar la regla del 15 %.

Todas se construyen con información disponible ANTES de la reseña (sin leakage):
tiempos de entrega, valor/peso/volumen del pedido, flete y calendario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def construir_features_extra(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade interacciones y derivadas sobre las features base del Sprint 2."""
    d = datos.copy()

    # ── Derivadas de tiempo ────────────────────────────────────────────────
    d["promised_delivery_days"] = d.get("estimated_delivery_days", np.nan)
    d["dispatch_time_hours"] = d.get("handling_days", 0).astype(float) * 24.0
    # ratio de retraso: cuánto se desvió respecto a lo prometido (acotado)
    prometido = d["promised_delivery_days"].replace(0, np.nan)
    d["delay_ratio"] = (d["delivery_delay_days"] / prometido).clip(-3, 3).fillna(0)

    # ── Valor y flete ──────────────────────────────────────────────────────
    d["log_payment_value"] = np.log1p(d.get("payment_value", d.get("total_price", 0)).clip(lower=0))
    # freight_ratio ya existe en sprint2; si no, lo derivamos
    if "freight_ratio" not in d:
        d["freight_ratio"] = (d["total_freight_value"] / d["total_price"].replace(0, np.nan)).clip(0, 5).fillna(0)

    # ── Dimensiones de producto (si están; si no, neutras) ─────────────────
    for col in ["product_width_cm", "product_height_cm", "product_length_cm"]:
        if col not in d:
            d[col] = 0.0
    d["product_volume_cm3"] = (
        d["product_width_cm"] * d["product_height_cm"] * d["product_length_cm"]
    ).fillna(0)

    # ── Geografía / vendedores ─────────────────────────────────────────────
    d["seller_customer_same_state"] = d.get("same_state_seller", 0).astype(int)
    if "unique_sellers" not in d:
        d["unique_sellers"] = 1  # 1 vendedor por pedido en la muestra

    # ── Calendario ─────────────────────────────────────────────────────────
    d["purchase_dayofweek"] = d.get("purchase_dow", 0)

    # ══ INTERACCIONES (el corazón del análisis de importancia) ═════════════
    # retraso × cantidad de ítems: pedidos grandes que además llegan tarde
    d["interaccion_retraso_items"] = (
        d["delivery_delay_days"].clip(lower=0) * d["order_item_count"]
    )
    # precio × llegó tarde: pedidos caros que llegan tarde duelen más
    d["interaccion_precio_tarde"] = d["log_payment_value"] * d.get("is_late", 0).astype(int)
    # tiempo de entrega × flete: logística cara y lenta
    d["interaccion_entrega_flete"] = d["actual_delivery_days"] * d["freight_ratio"]

    return d


# Conjunto completo de candidatas del Sprint 3 (orden del análisis de importancia)
FEATURES_SPRINT3 = [
    "interaccion_retraso_items", "delivery_delay_days", "delay_ratio",
    "order_item_count", "delivered_on_time", "actual_delivery_days",
    "interaccion_precio_tarde", "is_multi_item", "interaccion_entrega_flete",
    "dispatch_time_hours", "product_category_name_english", "purchase_dayofweek",
    "unique_sellers", "total_freight_value", "payment_installments",
    "promised_delivery_days", "freight_ratio", "product_width_cm",
    "product_height_cm", "product_length_cm", "log_payment_value",
    "product_volume_cm3", "total_price", "product_weight_g",
    "seller_customer_same_state",
]


def features_disponibles(datos: pd.DataFrame) -> list[str]:
    """Subconjunto de FEATURES_SPRINT3 realmente presente y con varianza > 0."""
    out = []
    for c in FEATURES_SPRINT3:
        if c in datos.columns:
            col = datos[c]
            if col.dtype.kind in "biufc":
                if col.nunique(dropna=True) > 1:
                    out.append(c)
            else:
                out.append(c)  # categóricas
    return out
