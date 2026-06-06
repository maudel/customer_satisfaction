"""
============================================================================
 generate_synthetic_data.py  ·  DATOS DE DEMOSTRACIÓN (NO usar en producción)
============================================================================

⚠️  ESTE SCRIPT ES SOLO PARA PROBAR EL PIPELINE SIN DESCARGAR KAGGLE.

Genera CSVs sintéticos con EL MISMO ESQUEMA que el Olist Brazilian
E-Commerce Dataset y con relaciones realistas (el retraso reduce la
satisfacción, ~78% de clientes satisfechos, periodo 2016-09 a 2018-09).

La generación está partida en etapas (`_generar_*`) para que se entienda de
un vistazo de dónde sale cada señal: primero las fechas, luego la logística,
después los valores económicos y, por último, la probabilidad de satisfacción.

EN PRODUCCIÓN: borra estos CSVs y coloca los reales de Kaggle en data/raw/:
    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

# Generador aleatorio con semilla fija → datos reproducibles entre corridas
RNG = np.random.default_rng(C.SEMILLA)

N_PEDIDOS = 40_000          # subconjunto suficiente para validar el pipeline

# Estados de Brasil con su probabilidad aproximada (SP domina el e-commerce)
ESTADOS = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "DF", "CE",
           "PE", "ES", "MA", "AL", "PA", "RN", "MT", "MS", "PB", "TO"]
PROB_ESTADOS = np.array([0.42, 0.13, 0.12, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02,
                         0.015, 0.015, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
PROB_ESTADOS = PROB_ESTADOS / PROB_ESTADOS.sum()

CATEGORIAS = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
              "computers_accessories", "housewares", "watches_gifts", "telephony",
              "auto", "toys", "perfumery", "office_furniture", "pet_shop",
              "stationery", "fashion_bags_accessories", "books_general_interest",
              "food_drink", "luggage_accessories"]

TIPOS_PAGO = ["credit_card", "boleto", "voucher", "debit_card"]
PROB_PAGOS = [0.74, 0.19, 0.05, 0.02]


def _id_hex(n, k=32):
    """Genera `n` identificadores hexadecimales de `k` caracteres (como los de Olist)."""
    caracteres = np.array(list("0123456789abcdef"))
    return ["".join(RNG.choice(caracteres, k)) for _ in range(n)]


def _generar_fechas() -> pd.Series:
    """Fechas de compra entre 2016-09 y 2018-09, con más volumen en meses recientes."""
    inicio = pd.Timestamp("2016-09-01")
    dias_rango = (pd.Timestamp("2018-09-15") - inicio).days
    # La beta sesga las compras hacia los meses más recientes (negocio en crecimiento)
    fraccion = RNG.beta(2.2, 1.4, N_PEDIDOS)
    compra = inicio + pd.to_timedelta((fraccion * dias_rango).astype(int), unit="D")
    compra += pd.to_timedelta(RNG.integers(0, 86400, N_PEDIDOS), unit="s")
    return compra


def _generar_logistica(compra: pd.Series) -> dict:
    """Tiempos logísticos y fechas derivadas (aprobación, transportista, entrega)."""
    manipulacion = RNG.gamma(2.0, 1.2, N_PEDIDOS) + 0.3                  # días en bodega
    transportista_a_cliente = RNG.gamma(3.0, 3.0, N_PEDIDOS) + 1.0       # días en ruta
    entrega_real = manipulacion + transportista_a_cliente               # compra → entrega
    entrega_estimada = entrega_real + RNG.normal(5, 6, N_PEDIDOS)        # promesa (~10-12% tarde)
    entrega_estimada = np.clip(entrega_estimada, 3, None)

    aprobado = compra + pd.to_timedelta(
        np.clip(RNG.gamma(1.2, 6, N_PEDIDOS), 0.1, 200), unit="h")
    fecha_transportista = compra + pd.to_timedelta(manipulacion, unit="D")
    entregado = compra + pd.to_timedelta(entrega_real, unit="D")
    estimado = compra + pd.to_timedelta(entrega_estimada, unit="D")

    retraso = (entregado - estimado).days.to_numpy()  # >0 = llegó tarde

    return {
        "entrega_real": entrega_real,
        "aprobado": aprobado,
        "fecha_transportista": fecha_transportista,
        "entregado": entregado,
        "estimado": estimado,
        "retraso": retraso,
    }


def _generar_valores() -> dict:
    """Valores económicos y atributos del producto/pago de cada pedido."""
    n_items = RNG.choice([1, 2, 3, 4, 5], N_PEDIDOS, p=[0.83, 0.12, 0.03, 0.015, 0.005])
    precio = np.round(RNG.gamma(2.0, 60, N_PEDIDOS) + 5, 2)
    flete = np.round(np.clip(RNG.gamma(2.0, 9, N_PEDIDOS) + 2, 2, None), 2)
    categorias = RNG.choice(CATEGORIAS, N_PEDIDOS)
    peso = np.round(np.clip(RNG.gamma(2.0, 800, N_PEDIDOS), 50, None), 0)
    fotos = RNG.integers(1, 7, N_PEDIDOS)
    tipo_pago = RNG.choice(TIPOS_PAGO, N_PEDIDOS, p=PROB_PAGOS)
    # Solo las tarjetas de crédito permiten cuotas
    cuotas = np.where(tipo_pago == "credit_card", RNG.integers(1, 11, N_PEDIDOS), 1)
    valor_pago = np.round(precio * n_items + flete, 2)

    return {
        "n_items": n_items, "precio": precio, "flete": flete,
        "categorias": categorias, "peso": peso, "fotos": fotos,
        "tipo_pago": tipo_pago, "cuotas": cuotas, "valor_pago": valor_pago,
    }


def _calcular_satisfaccion(logistica: dict, valores: dict) -> dict:
    """Modela la satisfacción: el retraso manda, con señales secundarias.

    Construimos un logit "de negocio" y de ahí sacamos el review_score:
    satisfechos → {4,5}, insatisfechos → {1,2,3}.
    """
    retraso = logistica["retraso"]
    entrega_real = logistica["entrega_real"]
    n_items, precio, flete = valores["n_items"], valores["precio"], valores["flete"]
    fotos, categorias = valores["fotos"], valores["categorias"]

    # Algunas categorías arrastran fama (buena o mala) que mueve la satisfacción
    efecto_categoria = pd.Series(categorias).map({
        "office_furniture": -0.35, "telephony": -0.25, "computers_accessories": -0.15,
        "books_general_interest": 0.40, "food_drink": 0.35, "luggage_accessories": 0.25,
    }).fillna(0.0).to_numpy()

    dias_tarde = np.clip(retraso, 0, 60)      # solo días de tardanza real
    dias_adelanto = np.clip(-retraso, 0, 30)  # días de adelanto
    logit = (
        1.05
        - 0.130 * dias_tarde                       # llegar TARDE es catastrófico
        + 0.020 * dias_adelanto                    # llegar antes ayuda (leve)
        - 0.035 * np.clip(entrega_real - 8, 0, 40)
        - 0.20 * (n_items - 1)                      # más ítems, menos satisfacción
        - 0.90 * (flete / (precio * n_items + flete))  # flete alto molesta
        + 0.060 * (fotos - 3)                       # más fotos, expectativa cumplida
        + efecto_categoria
        + RNG.normal(0, 0.40, N_PEDIDOS)
    )
    prob_satisfecho = 1 / (1 + np.exp(-logit))
    satisfecho = RNG.random(N_PEDIDOS) < prob_satisfecho

    # review_score coherente con la satisfacción simulada
    puntaje = np.where(
        satisfecho,
        RNG.choice([4, 5], N_PEDIDOS, p=[0.25, 0.75]),
        RNG.choice([1, 2, 3], N_PEDIDOS, p=[0.45, 0.18, 0.37]),
    )
    # Los insatisfechos comentan más que los satisfechos
    tiene_comentario = RNG.random(N_PEDIDOS) < np.where(satisfecho, 0.30, 0.62)
    comentario = np.where(tiene_comentario, "comentario", None)

    return {"puntaje": puntaje, "comentario": comentario}


def _construir_tablas(ids: dict, compra, logistica, valores, satisfaccion) -> dict:
    """Arma las 9 tablas de Olist con su esquema real a partir de las señales generadas."""
    order_id = ids["order_id"]
    customer_id = ids["customer_id"]
    estados = ids["estados"]
    n_items = valores["n_items"]

    clientes = pd.DataFrame({
        "customer_id": customer_id,
        "customer_unique_id": _id_hex(N_PEDIDOS),
        "customer_zip_code_prefix": RNG.integers(1000, 99999, N_PEDIDOS),
        "customer_city": "city",
        "customer_state": estados,
    })

    pedidos = pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "order_status": "delivered",
        "order_purchase_timestamp": compra,
        "order_approved_at": logistica["aprobado"],
        "order_delivered_carrier_date": logistica["fecha_transportista"],
        "order_delivered_customer_date": logistica["entregado"],
        "order_estimated_delivery_date": logistica["estimado"],
    })
    # Una fracción de pedidos NO entregados (para probar el filtro de población)
    n_no_entregados = int(N_PEDIDOS * 0.03)
    idx_no_entregados = RNG.choice(N_PEDIDOS, n_no_entregados, replace=False)
    pedidos.loc[idx_no_entregados, "order_status"] = "shipped"
    pedidos.loc[idx_no_entregados, "order_delivered_customer_date"] = pd.NaT

    resenas = pd.DataFrame({
        "review_id": _id_hex(N_PEDIDOS),
        "order_id": order_id,
        "review_score": satisfaccion["puntaje"],
        "review_comment_title": np.nan,
        "review_comment_message": satisfaccion["comentario"],
        "review_creation_date": logistica["entregado"],
        "review_answer_timestamp": logistica["entregado"],
    })

    seller_ids = _id_hex(N_PEDIDOS)
    product_ids = _id_hex(N_PEDIDOS)
    articulos = pd.DataFrame({
        "order_id": np.repeat(order_id, n_items),
        "order_item_id": np.concatenate([np.arange(1, k + 1) for k in n_items]),
        "product_id": np.repeat(product_ids, n_items),
        "seller_id": np.repeat(seller_ids, n_items),
        "shipping_limit_date": np.repeat(logistica["fecha_transportista"].values, n_items),
        "price": np.repeat(valores["precio"], n_items),
        "freight_value": np.repeat(valores["flete"], n_items) / n_items.repeat(n_items),
    })

    pagos = pd.DataFrame({
        "order_id": order_id,
        "payment_sequential": 1,
        "payment_type": valores["tipo_pago"],
        "payment_installments": valores["cuotas"],
        "payment_value": valores["valor_pago"],
    })

    productos = pd.DataFrame({
        "product_id": product_ids,
        "product_category_name": valores["categorias"],
        "product_name_lenght": RNG.integers(20, 60, N_PEDIDOS),
        "product_description_lenght": RNG.integers(100, 2000, N_PEDIDOS),
        "product_photos_qty": valores["fotos"],
        "product_weight_g": valores["peso"],
        "product_length_cm": RNG.integers(10, 100, N_PEDIDOS),
        "product_height_cm": RNG.integers(2, 50, N_PEDIDOS),
        "product_width_cm": RNG.integers(5, 80, N_PEDIDOS),
    })

    vendedores = pd.DataFrame({
        "seller_id": seller_ids,
        "seller_zip_code_prefix": RNG.integers(1000, 99999, N_PEDIDOS),
        "seller_city": "city",
        "seller_state": RNG.choice(ESTADOS, N_PEDIDOS, p=PROB_ESTADOS),
    })

    categorias = pd.DataFrame({
        "product_category_name": CATEGORIAS,
        "product_category_name_english": CATEGORIAS,  # ya en inglés en la demo
    })

    geolocalizacion = pd.DataFrame({
        "geolocation_zip_code_prefix": RNG.integers(1000, 99999, 1000),
        "geolocation_lat": RNG.uniform(-33, 5, 1000),
        "geolocation_lng": RNG.uniform(-73, -34, 1000),
        "geolocation_city": "city",
        "geolocation_state": RNG.choice(ESTADOS, 1000, p=PROB_ESTADOS),
    })

    return {
        "orders": pedidos, "reviews": resenas, "items": articulos,
        "payments": pagos, "customers": clientes, "products": productos,
        "sellers": vendedores, "category": categorias, "geolocation": geolocalizacion,
    }


def generar():
    """Genera todas las tablas sintéticas y las escribe como CSV en data/raw/."""
    print("Generando datos sintéticos (demo)...")

    # 1) Identificadores y fechas base
    compra = _generar_fechas()
    ids = {
        "order_id": _id_hex(N_PEDIDOS),
        "customer_id": _id_hex(N_PEDIDOS),
        "estados": RNG.choice(ESTADOS, N_PEDIDOS, p=PROB_ESTADOS),
    }

    # 2) Señales encadenadas: logística → valores → satisfacción
    logistica = _generar_logistica(compra)
    valores = _generar_valores()
    satisfaccion = _calcular_satisfaccion(logistica, valores)

    # 3) Construir y volcar las 9 tablas con el esquema real de Olist
    tablas = _construir_tablas(ids, compra, logistica, valores, satisfaccion)
    nombres = dict(C.ARCHIVOS_CRUDOS)
    nombres["geolocation"] = "olist_geolocation_dataset.csv"
    for alias, tabla in tablas.items():
        ruta = C.DIR_CRUDO / nombres[alias]
        tabla.to_csv(ruta, index=False)
        print(f"  {nombres[alias]:<45} {tabla.shape}")
    print("Datos sintéticos generados en", C.DIR_CRUDO)


if __name__ == "__main__":
    generar()
