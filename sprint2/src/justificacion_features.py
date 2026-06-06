"""
============================================================================
 justificacion_features.py  ·  Paso 3b · Evidencia del Feature Engineering
============================================================================

Cada familia de features del Paso 3 nace de una hipótesis del Sprint 1
(la logística manda, los pedidos complejos fallan más, etc.). Este módulo
cierra el círculo midiendo, SOLO sobre TRAIN (leak-safe), cuánta señal
aporta realmente cada variable candidata:

    gini_univariante : |2·AUC − 1| de la variable sola contra el target
                       (la métrica del umbral del método univariante)
    information_value: IV por WOE en 10 bins (escala clásica de scoring)
    gap_satisfaccion : para banderas binarias, diferencia de tasa de
                       satisfechos entre el grupo 1 y el grupo 0 (en pp)

Interpretación rápida (mismas escalas que usa la cascada):
    Gini  < 0.05 : ruido      ·  0.05–0.15 : débil  ·  > 0.15 : señal real
    IV    < 0.02 : sin poder  ·  0.02–0.10 : débil  ·  0.10–0.30 : medio
          0.30–0.50 : fuerte  ·  > 0.50 : sospechoso (revisar fuga)

Salidas:
    reports/justificacion_features.csv
    reports/figures/s2_fig08_justificacion_features.png  (vía figuras.py)
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import features as F
from . import selection

# Mapa variable -> familia del Paso 3 (mismo orden que la lámina A-G).
# Mantenerlo aquí, junto a la evidencia, hace explícito qué hipótesis
# respalda cada variable cuando el docente pregunta "¿y esta por qué?".
FAMILIAS = {
    "A · Tiempos de entrega": [
        "delivery_delay_days", "actual_delivery_days", "estimated_delivery_days",
        "approval_time_hours", "handling_days",
    ],
    "B · Banderas de entrega": [
        "delivered_on_time", "is_late", "delivered_early_5d",
    ],
    "C · Valor del pedido": [
        "total_price", "total_freight_value", "order_total_value",
        "freight_ratio", "price_per_item", "payment_value", "is_multi_item",
        "order_item_count",
    ],
    "D · Pago": ["payment_installments", "is_installment"],
    "E · Producto": ["product_weight_g", "product_photos_qty", "is_heavy_item"],
    "F · Señal de reseña": ["has_comment"],
    "G · Calendario y geografía": [
        "purchase_dow", "purchase_month_num", "purchase_is_weekend",
        "same_state_seller",
    ],
}


def _familia_de(variable: str) -> str:
    """Familia del catálogo a la que pertenece la variable (o 'categórica')."""
    for familia, miembros in FAMILIAS.items():
        if variable in miembros:
            return familia
    return "H · Categóricas (IV aparte)"


# ════════════════════════════════════════════════════════════════════════
#  TABLA DE EVIDENCIA  (una fila por variable candidata · solo TRAIN)
# ════════════════════════════════════════════════════════════════════════
def tabla_justificacion(train: pd.DataFrame) -> pd.DataFrame:
    """Gini univariante + IV + gap de satisfacción por variable candidata.

    Se calcula sobre el TRAIN LIMPIO (después del clipado): es la misma
    población con la que decide la cascada, así la evidencia y la
    selección hablan el mismo idioma.
    """
    y = train[C.OBJETIVO].astype(int)
    filas = []
    for variable in F.todas_las_columnas():
        if variable not in train.columns:
            continue
        es_numerica = variable in F.VARIABLES_NUMERICAS + F.VARIABLES_BINARIAS

        gini = selection.fuerza_univariante(train[variable], y) if es_numerica else float("nan")
        iv = selection.valor_informacion(train[variable], y) if es_numerica else float("nan")

        # Gap de satisfacción: solo tiene lectura directa en banderas 0/1
        gap = float("nan")
        if variable in F.VARIABLES_BINARIAS:
            tasas = train.groupby(train[variable].fillna(0).astype(int))[C.OBJETIVO].mean()
            if {0, 1} <= set(tasas.index):
                gap = (tasas[1] - tasas[0]) * 100

        filas.append({
            "familia": _familia_de(variable),
            "variable": variable,
            "gini_univariante": round(gini, 4) if gini == gini else None,
            "information_value": round(iv, 4) if iv == iv else None,
            "gap_satisfaccion_pp": round(gap, 1) if gap == gap else None,
        })

    tabla = (pd.DataFrame(filas)
             .sort_values(["familia", "gini_univariante"],
                          ascending=[True, False], na_position="last")
             .reset_index(drop=True))
    tabla.to_csv(C.DIR_REPORTES / "justificacion_features.csv", index=False)
    return tabla


def resumen_por_familia(tabla: pd.DataFrame) -> pd.DataFrame:
    """La mejor variable de cada familia: el titular para la lámina.

    Responde en una fila por hipótesis: "¿la familia aporta señal?" — si
    su campeona supera Gini 0.05, la hipótesis sobrevive a los datos.
    """
    numericas = tabla.dropna(subset=["gini_univariante"])
    idx = numericas.groupby("familia")["gini_univariante"].idxmax()
    resumen = numericas.loc[idx, ["familia", "variable",
                                  "gini_univariante", "information_value"]]
    return resumen.sort_values("gini_univariante", ascending=False).reset_index(drop=True)
