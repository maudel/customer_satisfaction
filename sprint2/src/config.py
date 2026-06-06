"""
============================================================================
 config.py  ·  Configuración central del pipeline
 Proyecto: Predicción de Satisfacción del Cliente (Olist) — Grupo 1
 Sprint 2 · Ingeniería de datos y pipeline reproducible
============================================================================

Toda la parametrización del flujo de trabajo vive aquí para garantizar
reproducibilidad. Si cambian las rutas, el split temporal o los umbrales
de selección de variables, se edita ESTE archivo y nada más.

Nota sobre los nombres: en este proyecto los IDENTIFICADORES de código
(constantes, funciones, variables) están en español, pero los NOMBRES DE
COLUMNA se dejan en inglés porque son el "contrato" con los CSV reales de
Olist (Kaggle). Así el código sigue cargando los datos originales sin
necesidad de traducir cabeceras.

Refleja el "flujo de trabajo de Modelos de Negocio" (8 pasos):
  1. EDA Básico            (Sprint 1)
  2. Población Objetivo / Target
  3. Generación de Features
  4. Creación de la Master Table  (split temporal Train/Val/BackTest/Live/Pred)
  5. Master Table (40-60 features)
  6. Limpieza de variables (clipado, NaN, agrupamiento)
  7. Selección de variables (missing → PSI → correlación → univariante)
  8. Hiperparametrización  (Sprint 3)
"""
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# VERSIONADO DEL PIPELINE
# ───────────────────────────────────────────────────────────────────────────
# Historial de versiones (cada cambio de lógica incrementa la versión):
#   2.0.0 : Sprint 2 base — pipeline modular, split temporal, cascada, RF.
#   2.1.0 : Mejoras post-defensa — comprobación del target + targets
#           alternativos (Paso 2b), Gini por paso en la cascada, tabla de
#           lift por riesgo, ablación de variables en revisión y figuras
#           de defensa generadas por el propio pipeline.
VERSION_PIPELINE = "2.1.0"          # se incrementa con cada cambio de lógica
SEMILLA          = 42               # semilla aleatoria global (reproducibilidad)

# ───────────────────────────────────────────────────────────────────────────
# RUTAS
# ───────────────────────────────────────────────────────────────────────────
# Este Sprint vive dentro del monorepo `customer-satisfaction-ml`, que tiene
# una carpeta `data/` COMPARTIDA en la raíz (los CSV reales de Olist viven ahí
# y los reutilizan todos los sprints). Por eso los DATOS se leen del repo
# (parents[2]) y los ARTEFACTOS propios del Sprint 2 (modelos, reportes) se
# escriben localmente bajo `sprint2/` para que cada sprint sea autocontenido.
DIR_RAIZ      = Path(__file__).resolve().parents[1]   # .../sprint2/
DIR_REPO      = Path(__file__).resolve().parents[2]   # .../customer-satisfaction-ml/
DIR_DATOS     = DIR_REPO / "data"            # data COMPARTIDA del repo
DIR_CRUDO     = DIR_DATOS / "raw"            # CSVs originales de Kaggle (Olist)
DIR_PROCESADO = DIR_DATOS / "processed"      # master tables generadas (compartidas)
DIR_MODELOS   = DIR_RAIZ / "models"          # pickles versionados (propios del Sprint 2)
DIR_REPORTES  = DIR_RAIZ / "reports"         # métricas y reportes (propios del Sprint 2)
DIR_FIGURAS   = DIR_REPORTES / "figures"

# Crea las carpetas si todavía no existen (la primera ejecución las necesita)
for _carpeta in (DIR_CRUDO, DIR_PROCESADO, DIR_MODELOS, DIR_REPORTES, DIR_FIGURAS):
    _carpeta.mkdir(parents=True, exist_ok=True)

# Nombres de los CSV crudos de Olist (la clave es nuestro alias interno)
ARCHIVOS_CRUDOS = {
    "orders":    "olist_orders_dataset.csv",
    "reviews":   "olist_order_reviews_dataset.csv",
    "items":     "olist_order_items_dataset.csv",
    "payments":  "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products":  "olist_products_dataset.csv",
    "sellers":   "olist_sellers_dataset.csv",
    "category":  "product_category_name_translation.csv",
}

RUTA_MASTER_TABLE = DIR_PROCESADO / "master_table.csv"

# ───────────────────────────────────────────────────────────────────────────
# PASO 2 · POBLACIÓN OBJETIVO Y TARGET
# ───────────────────────────────────────────────────────────────────────────
# Solo pedidos efectivamente entregados entran al modelo de satisfacción.
ESTADO_PEDIDO_VALIDO = "delivered"

# Target preliminar del Sprint 2: clasificación binaria de satisfacción.
#   is_satisfied = 1  si review_score >= UMBRAL_SATISFACCION  (4 ó 5 estrellas)
#   is_satisfied = 0  en caso contrario  (1-3 estrellas)
# Se usa clasificación binaria porque las métricas exigidas (F1, AUC, Gini)
# y el flujo de selección de variables del curso operan sobre AUC-ROC.
OBJETIVO            = "is_satisfied"      # nombre de la columna target
COL_PUNTAJE_RESENA  = "review_score"     # puntaje crudo 1-5 de la reseña
UMBRAL_SATISFACCION = 4

# Paso 2b · Comprobación del target + candidatos a TARGET FINAL (Sprint 3).
# El binario es el Target PRELIMINAR del Sprint 2; estas columnas conviven
# en la master para que el Sprint 3 compare alternativas con el mismo
# pipeline (respuesta a la observación del docente: "mejorar ese target").
OBJETIVO_ORDINAL  = "satisfaction_class"   # 0 detractor · 1 neutro · 2 promotor
OBJETIVO_ESTRICTO = "is_promoter"          # 1 solo si score = 5
UMBRAL_PROMOTOR   = 5

# Columna de control temporal (operación mensual)
COL_MES = "purchase_month"

# Columnas "y" / de control que NO se usan como features (paso 7: variables_y)
COLS_ID = ["order_id", "customer_id", "seller_id", "product_id"]
COLS_Y  = [OBJETIVO, COL_PUNTAJE_RESENA, COL_MES]

# ───────────────────────────────────────────────────────────────────────────
# PASO 4 · SPLIT TEMPORAL  (incorporación mensual de datos)
# ───────────────────────────────────────────────────────────────────────────
# El dataset Olist cubre pedidos entre 2016-09 y 2018-09.
# Se respeta el orden temporal: nunca se entrena con el futuro.
#
#   ENTRENAMIENTO : todos los meses hasta FIN_ENTRENAMIENTO (inclusive)
#   VALIDACION    : MESES_VALIDACION        (validación / early-stopping)
#   BACKTEST      : MESES_BACKTEST          (evaluación out-of-time mes a mes)
#   PRODUCCION    : MES_PRODUCCION          (último mes con label, simula producción)
#   PREDICCION    : MES_PREDICCION          (mes nuevo sin label → se predice)
FIN_ENTRENAMIENTO = "2018-03"
MESES_VALIDACION  = ["2018-04"]
MESES_BACKTEST    = ["2018-05", "2018-06", "2018-07"]
MES_PRODUCCION    = "2018-08"
MES_PREDICCION    = "2018-09"

# ───────────────────────────────────────────────────────────────────────────
# PASO 6 · LIMPIEZA DE VARIABLES
# ───────────────────────────────────────────────────────────────────────────
CUANTILES_CLIP         = (0.01, 0.99)   # clipado de outliers en variables continuas
PCT_MIN_CATEGORIA_RARA = 0.01           # categorías con <1% se agrupan en "OTHER"
ETIQUETA_CATEGORIA_RARA = "OTHER"

# ───────────────────────────────────────────────────────────────────────────
# PASO 7 · SELECCIÓN DE VARIABLES  (umbrales del flujo de trabajo)
# ───────────────────────────────────────────────────────────────────────────
UMBRAL_FALTANTES    = 0.10     # descarta features con >10% de nulos
NUM_BINS_PSI        = 10       # nº de bins para el cálculo del PSI
UMBRAL_PSI          = 0.25     # PSI > 0.25 ⇒ variable inestable en el tiempo
UMBRAL_CORRELACION  = 0.95     # |corr| > 0.95 ⇒ feature redundante
UMBRAL_UNIVARIANTE  = 0.05     # Gini univariante mínimo (2*AUC-1); <0.05 ⇒ ruido
UMBRAL_UNIVARIANTE_BARRIDO = [0.05, 0.10, 0.20, 0.30]  # barrido del flujo de trabajo

# Variables del set final bajo revisión metodológica (frontera de leakage).
# La ablación de sensibilidad (src/sensibilidad.py) entrena con y sin cada
# una y decide con TOLERANCIA_ABLACION si su aporte de AUC es real.
VARIABLES_EN_REVISION = ["has_comment"]
TOLERANCIA_ABLACION   = 0.005     # ΔAUC_val mínimo para considerarla necesaria

# Tabla de lift: percentiles de riesgo para la priorización de soporte
# ("contactar solo al top X% más riesgoso" — la palanca del umbral).
PERCENTILES_LIFT = [5, 10, 20]

# Modelo evaluador usado en cada paso de la cascada (RF, como en el flujo)
PARAMS_RF_SELECCION = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=SEMILLA,
    class_weight="balanced",
)
