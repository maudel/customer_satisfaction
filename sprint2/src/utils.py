"""
============================================================================
 utils.py  ·  Carga de datos, construcción de la Master Table y E/S
============================================================================

Mantiene compatibilidad con el Sprint 1:
    from src.utils import cargar_csvs_crudos, construir_master_table, guardar_master_table

y añade utilidades del Sprint 2:
    - split temporal por mes (train / val / backtest / producción / predicción)
    - versionado y guardado de artefactos
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ════════════════════════════════════════════════════════════════════════
#  CARGA DE CSVs CRUDOS
# ════════════════════════════════════════════════════════════════════════
def cargar_csvs_crudos(dir_crudo: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Carga las 8 tablas de Olist desde `data/raw/`.

    Devuelve un dict {alias: DataFrame}. Las fechas de `orders` se parsean
    a datetime de inmediato para poder calcular tiempos de entrega después.
    """
    dir_crudo = Path(dir_crudo) if dir_crudo else C.DIR_CRUDO

    pedidos = pd.read_csv(
        dir_crudo / C.ARCHIVOS_CRUDOS["orders"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    resenas    = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["reviews"])
    articulos  = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["items"])
    pagos      = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["payments"])
    clientes   = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["customers"])
    productos  = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["products"])
    vendedores = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["sellers"])
    categorias = pd.read_csv(dir_crudo / C.ARCHIVOS_CRUDOS["category"])

    # Las claves del dict son los alias internos usados en construir_master_table
    return {
        "orders": pedidos, "reviews": resenas, "items": articulos,
        "payments": pagos, "customers": clientes, "products": productos,
        "sellers": vendedores, "category": categorias,
    }


# ════════════════════════════════════════════════════════════════════════
#  MASTER TABLE  (paso 4-5 del flujo de trabajo)
# ════════════════════════════════════════════════════════════════════════
def _agregar_articulos(articulos: pd.DataFrame, productos: pd.DataFrame) -> pd.DataFrame:
    """Agrega los ítems a nivel de pedido y les pega los atributos del producto.

    Un pedido puede tener varios ítems (relación 1-N), así que primero los
    resumimos: cantidad, precio total, flete total y el primer producto/vendedor.
    """
    articulos_agg = (
        articulos.groupby("order_id")
        .agg(
            order_item_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            product_id=("product_id", "first"),
            seller_id=("seller_id", "first"),
        )
        .reset_index()
    )
    return articulos_agg.merge(
        productos[["product_id", "product_category_name_english",
                   "product_weight_g", "product_photos_qty"]],
        on="product_id", how="left",
    )


def _agregar_pagos(pagos: pd.DataFrame) -> pd.DataFrame:
    """Agrega los pagos a nivel de pedido (también es relación 1-N)."""
    return (
        pagos.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )


def construir_master_table(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une las tablas relacionales en una única Master Table a nivel de pedido.

    Granularidad: 1 fila = 1 order_id (con review).
    Las tablas 1-N (items, payments) se agregan antes de unir.
    """
    pedidos    = tablas["orders"].copy()
    resenas    = tablas["reviews"].copy()
    articulos  = tablas["items"].copy()
    pagos      = tablas["payments"].copy()
    clientes   = tablas["customers"].copy()
    productos  = tablas["products"].copy()
    vendedores = tablas["sellers"].copy()
    categorias = tablas["category"].copy()

    # Traducir las categorías de producto a inglés (tabla puente de Olist)
    productos = productos.merge(categorias, on="product_category_name", how="left")

    # Agregar las tablas 1-N antes de unir
    articulos_agg = _agregar_articulos(articulos, productos)
    pagos_agg = _agregar_pagos(pagos)

    # Unir todo a nivel order_id (inner con reviews: solo pedidos con reseña)
    master = (
        pedidos
        .merge(resenas[["order_id", "review_score", "review_comment_message"]],
               on="order_id", how="inner")
        .merge(articulos_agg, on="order_id", how="left")
        .merge(pagos_agg, on="order_id", how="left")
        .merge(clientes[["customer_id", "customer_state",
                         "customer_zip_code_prefix"]],
               on="customer_id", how="left")
        .merge(vendedores[["seller_id", "seller_state"]], on="seller_id", how="left")
    )
    return master


def guardar_master_table(master: pd.DataFrame, ruta: Path | str | None = None) -> Path:
    """Guarda la Master Table en CSV (data/processed/master_table.csv)."""
    ruta = Path(ruta) if ruta else C.RUTA_MASTER_TABLE
    ruta.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(ruta, index=False)
    print(f"Master Table guardada: {master.shape[0]:,} filas x {master.shape[1]} columnas")
    print(f"  -> {ruta}")
    return ruta


def cargar_master_table(ruta: Path | str | None = None) -> pd.DataFrame:
    """Carga la Master Table parseando fechas y tipos básicos."""
    ruta = Path(ruta) if ruta else C.RUTA_MASTER_TABLE
    cols_fecha = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    master = pd.read_csv(ruta)
    for col in cols_fecha:
        if col in master.columns:
            master[col] = pd.to_datetime(master[col], errors="coerce")
    return master


# ════════════════════════════════════════════════════════════════════════
#  SPLIT TEMPORAL  (paso 4 del flujo · incorporación mensual)
# ════════════════════════════════════════════════════════════════════════
def asignar_split_temporal(datos: pd.DataFrame, col_mes: str = C.COL_MES) -> pd.DataFrame:
    """Añade la columna `dataset_split` ∈ {train, val, backtest, live, predict}.

    Respeta el orden temporal definido en config (nunca entrenamos con el futuro):
      train < val < backtest < live < predict
    """
    datos = datos.copy()
    meses = datos[col_mes].astype(str)

    split = pd.Series("excluded", index=datos.index, dtype=object)
    split[meses <= C.FIN_ENTRENAMIENTO]      = "train"
    split[meses.isin(C.MESES_VALIDACION)]    = "val"
    split[meses.isin(C.MESES_BACKTEST)]      = "backtest"
    split[meses == C.MES_PRODUCCION]         = "live"
    split[meses == C.MES_PREDICCION]         = "predict"

    datos["dataset_split"] = split
    return datos


def resumen_split(datos: pd.DataFrame) -> pd.DataFrame:
    """Resumen de filas y rango de meses por partición temporal."""
    resumen = (
        datos.groupby("dataset_split")
        .agg(
            filas=("order_id", "count") if "order_id" in datos.columns
                  else (datos.columns[0], "count"),
            mes_min=(C.COL_MES, "min"),
            mes_max=(C.COL_MES, "max"),
        )
    )
    orden = ["train", "val", "backtest", "live", "predict", "excluded"]
    return resumen.reindex([o for o in orden if o in resumen.index])


# ════════════════════════════════════════════════════════════════════════
#  VERSIONADO / E-S DE ARTEFACTOS
# ════════════════════════════════════════════════════════════════════════
def escribir_json(obj: dict, ruta: Path | str) -> Path:
    """Vuelca un dict a JSON (convirtiendo tipos de numpy/pandas a nativos)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(_a_serializable(obj), f, indent=2, ensure_ascii=False)
    return ruta


def _a_serializable(obj):
    """Convierte numpy/pandas a tipos nativos para poder serializar a JSON."""
    if isinstance(obj, dict):
        return {str(k): _a_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_a_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def metadatos_ejecucion(extra: dict | None = None) -> dict:
    """Metadatos de trazabilidad para cada ejecución del pipeline."""
    meta = {
        "pipeline_version": C.VERSION_PIPELINE,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "random_state": C.SEMILLA,
        "target": C.OBJETIVO,
        "train_end": C.FIN_ENTRENAMIENTO,
        "val_months": C.MESES_VALIDACION,
        "backtest_months": C.MESES_BACKTEST,
        "live_month": C.MES_PRODUCCION,
        "prediction_month": C.MES_PREDICCION,
    }
    if extra:
        meta.update(extra)
    return meta
