"""
============================================================================
 features.py  ·  Paso 2 (Target) + Paso 3 (Generación de Features)
============================================================================

Toda la lógica es "leak-safe" y se calcula a nivel de pedido, de modo que
puede ejecutarse mes a mes sobre datos nuevos sin mirar el futuro.

NINGUNA feature usa información posterior a la fecha de compra excepto las
que describen la entrega (que ya ocurrió cuando el cliente puntúa) — esto es
correcto porque la predicción se hace en el momento en que el pedido llega
al cliente, justo antes de que emita la reseña.

La generación de features está partida en varias funciones pequeñas (una por
"familia" de variables) para que sea fácil de leer y de extender: si mañana
queremos una feature nueva de logística, la añadimos a `_features_tiempos`
y listo, sin tocar el resto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ════════════════════════════════════════════════════════════════════════
#  PASO 2 · POBLACIÓN OBJETIVO Y TARGET
# ════════════════════════════════════════════════════════════════════════
def filtrar_poblacion(datos: pd.DataFrame) -> pd.DataFrame:
    """Filtra la población objetivo: pedidos entregados con review válido."""
    poblacion = datos[datos["order_status"] == C.ESTADO_PEDIDO_VALIDO].copy()
    # Sin puntaje de reseña no hay target que predecir
    poblacion = poblacion.dropna(subset=[C.COL_PUNTAJE_RESENA])
    # Pedidos sin fecha de entrega real no aportan a la predicción de entrega
    poblacion = poblacion.dropna(subset=["order_delivered_customer_date"])
    return poblacion


def construir_objetivo(datos: pd.DataFrame) -> pd.DataFrame:
    """Crea el target preliminar binario `is_satisfied` (1 = satisfecho)."""
    datos = datos.copy()
    datos[C.COL_PUNTAJE_RESENA] = datos[C.COL_PUNTAJE_RESENA].astype(int)
    datos[C.OBJETIVO] = (
        datos[C.COL_PUNTAJE_RESENA] >= C.UMBRAL_SATISFACCION
    ).astype(int)
    return datos


# ════════════════════════════════════════════════════════════════════════
#  PASO 3 · GENERACIÓN DE FEATURES
#  Cada bloque temático vive en su propia función auxiliar (módulo más limpio).
# ════════════════════════════════════════════════════════════════════════
def _feature_mes(datos: pd.DataFrame) -> pd.DataFrame:
    """Mes de la compra (clave de control temporal para el split mensual)."""
    datos[C.COL_MES] = (
        datos["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )
    return datos


def _features_tiempos(datos: pd.DataFrame) -> pd.DataFrame:
    """Tiempos de entrega: el factor que más mueve la satisfacción del cliente."""
    # Retraso vs lo estimado (>0 = llegó tarde, <0 = llegó antes)
    datos["delivery_delay_days"] = (
        datos["order_delivered_customer_date"] - datos["order_estimated_delivery_date"]
    ).dt.days

    # Tiempo real de entrega (compra → entrega al cliente)
    datos["actual_delivery_days"] = (
        datos["order_delivered_customer_date"] - datos["order_purchase_timestamp"]
    ).dt.days

    # Tiempo estimado prometido (compra → fecha estimada)
    datos["estimated_delivery_days"] = (
        datos["order_estimated_delivery_date"] - datos["order_purchase_timestamp"]
    ).dt.days

    # Tiempo de aprobación del pago (en horas)
    datos["approval_time_hours"] = (
        datos["order_approved_at"] - datos["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600.0

    # Tiempo de manipulación: compra → entrega al transportista (días)
    datos["handling_days"] = (
        datos["order_delivered_carrier_date"] - datos["order_purchase_timestamp"]
    ).dt.days
    return datos


def _features_banderas_entrega(datos: pd.DataFrame) -> pd.DataFrame:
    """Banderas binarias derivadas del retraso de entrega."""
    datos["delivered_on_time"] = (datos["delivery_delay_days"] <= 0).astype(int)
    datos["is_late"]           = (datos["delivery_delay_days"] > 0).astype(int)
    # ¿Llegó bastante antes de lo prometido? (suele dar una experiencia muy buena)
    datos["delivered_early_5d"] = (datos["delivery_delay_days"] <= -5).astype(int)
    return datos


def _features_valor_pedido(datos: pd.DataFrame) -> pd.DataFrame:
    """Variables económicas del pedido: precio, flete y su relación."""
    datos["order_item_count"] = datos["order_item_count"].fillna(0)
    datos["total_price"]         = pd.to_numeric(datos["total_price"], errors="coerce")
    datos["total_freight_value"] = pd.to_numeric(datos["total_freight_value"], errors="coerce")
    datos["payment_value"]       = pd.to_numeric(datos.get("payment_value"), errors="coerce")

    # Valor total = mercadería + flete
    datos["order_total_value"] = (
        datos["total_price"].fillna(0) + datos["total_freight_value"].fillna(0)
    )
    # Peso del flete sobre el total (un flete caro suele molestar al cliente)
    datos["freight_ratio"] = datos["total_freight_value"] / (datos["order_total_value"] + 1e-9)
    # Precio promedio por ítem
    datos["price_per_item"] = datos["total_price"] / (datos["order_item_count"].replace(0, np.nan))
    datos["is_multi_item"] = (datos["order_item_count"] > 1).astype(int)
    return datos


def _features_pago(datos: pd.DataFrame) -> pd.DataFrame:
    """Variables del medio de pago."""
    datos["payment_installments"] = pd.to_numeric(
        datos.get("payment_installments"), errors="coerce"
    ).fillna(1)
    datos["is_installment"] = (datos["payment_installments"] > 1).astype(int)
    return datos


def _features_producto(datos: pd.DataFrame) -> pd.DataFrame:
    """Atributos del producto (peso, fotos)."""
    datos["product_weight_g"]   = pd.to_numeric(datos.get("product_weight_g"), errors="coerce")
    datos["product_photos_qty"] = pd.to_numeric(datos.get("product_photos_qty"), errors="coerce")
    # Productos muy pesados (>5 kg) son más propensos a problemas de envío
    datos["is_heavy_item"] = (datos["product_weight_g"] > 5000).astype(int)
    return datos


def _features_resena(datos: pd.DataFrame) -> pd.DataFrame:
    """Señales de la reseña que SÍ conocemos al momento de predecir."""
    # Que el cliente se tome la molestia de comentar suele indicar emoción (buena o mala)
    datos["has_comment"] = datos["review_comment_message"].notna().astype(int)
    return datos


def _features_calendario(datos: pd.DataFrame) -> pd.DataFrame:
    """Estacionalidad y geografía (compra en finde, mismo estado que el vendedor…)."""
    datos["purchase_dow"]        = datos["order_purchase_timestamp"].dt.dayofweek
    datos["purchase_month_num"]  = datos["order_purchase_timestamp"].dt.month
    datos["purchase_is_weekend"] = (datos["purchase_dow"] >= 5).astype(int)
    # ¿Cliente y vendedor en el mismo estado? (suele acortar la logística)
    datos["same_state_seller"] = (
        datos["customer_state"].astype(str) == datos["seller_state"].astype(str)
    ).astype(int)
    return datos


def generar_variables(datos: pd.DataFrame) -> pd.DataFrame:
    """Genera todas las variables derivadas del pedido.

    Encadena los bloques temáticos en orden. Devuelve el DataFrame con las
    nuevas columnas añadidas (las originales se conservan).
    """
    datos = datos.copy()
    datos = _feature_mes(datos)
    datos = _features_tiempos(datos)
    datos = _features_banderas_entrega(datos)
    datos = _features_valor_pedido(datos)
    datos = _features_pago(datos)
    datos = _features_producto(datos)
    datos = _features_resena(datos)
    datos = _features_calendario(datos)
    return datos


# Catálogo de features producidas, por tipo (consumido por el preprocesador).
# OJO: son NOMBRES DE COLUMNA, por eso van en inglés (contrato con los datos).
VARIABLES_NUMERICAS = [
    "delivery_delay_days", "actual_delivery_days", "estimated_delivery_days",
    "approval_time_hours", "handling_days",
    "order_item_count", "total_price", "total_freight_value",
    "order_total_value", "freight_ratio", "price_per_item",
    "payment_value", "payment_installments",
    "product_weight_g", "product_photos_qty",
    "purchase_dow", "purchase_month_num",
]

VARIABLES_BINARIAS = [
    "delivered_on_time", "is_late", "delivered_early_5d",
    "is_multi_item", "is_installment", "is_heavy_item", "has_comment",
    "purchase_is_weekend", "same_state_seller",
]

VARIABLES_CATEGORICAS = [
    "product_category_name_english", "payment_type",
    "customer_state", "seller_state",
]


def todas_las_columnas() -> list[str]:
    """Lista plana con todas las features candidatas (numéricas + binarias + categóricas)."""
    return VARIABLES_NUMERICAS + VARIABLES_BINARIAS + VARIABLES_CATEGORICAS


def construir_variables(datos: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features: población → target → ingeniería.

    Es el único punto de entrada que se usa tanto en entrenamiento como
    en la simulación mensual de datos nuevos.
    """
    datos = filtrar_poblacion(datos)
    datos = construir_objetivo(datos)
    datos = generar_variables(datos)
    return datos
