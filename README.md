# Sprint 2 · Pipeline de Predicción de Satisfacción del Cliente (Olist)

**Grupo 1** · Caso de estudio #1 · *Implementación de Soluciones de IA Aplicada a los Negocios*

**Versión del pipeline:** `2.0.0`

---

## 1. Objetivo

Construir un **pipeline de datos robusto, reproducible y escalable** para predecir la
satisfacción del cliente sobre el dataset *Olist Brazilian E-Commerce*, contemplando la
**incorporación mensual de nuevos datos**, y definir el **Target Preliminar**.

El proyecto implementa el *flujo de trabajo de Modelos de Negocio* de 8 pasos. El Sprint 2
cubre los **pasos 2 a 7** más el pipeline reproducible y un modelo baseline; el Sprint 3
abordará el paso 8 (hiperparametrización y modelo final).

---

## 2. Target Preliminar

| | |
|---|---|
| **Variable** | `is_satisfied` (binaria) |
| **Regla** | `review_score ≥ 4` → satisfecho (1); `≤ 3` → insatisfecho (0) |
| **Tipo de problema** | Clasificación binaria |
| **Métricas** | F1, ROC-AUC, **Gini** (= 2·AUC − 1) |

Se eligió clasificación binaria porque las métricas exigidas y la cascada de selección de
variables del curso operan sobre **AUC-ROC**.

---

## 3. Flujo de datos

```
data/raw/  (9 CSV de Olist)
    │
    │  utils.load_raw_csvs()
    ▼
[9 tablas relacionales]
    │  utils.build_master_table()        ← Pasos 4-5: une y agrega a nivel order_id
    ▼
data/processed/master_table.csv
    │  features.make_features()          ← Pasos 2-3: población + target + features
    ▼
[Master Table con features + target]
    │  utils.assign_temporal_split()     ← Paso 4: split temporal por mes
    ▼
 train · val · backtest · live · predict
    │  cleaning.DataCleaner (fit en train) ← Paso 6: clipado, NaN, agrupamiento
    ▼
[Particiones limpias]
    │  selection.FeatureSelector()       ← Paso 7: cascada missing→PSI→corr→univariante→WOE
    ▼
[Features seleccionadas]
    │  pipeline.build_pipeline()         ← Pasos 5-6: ColumnTransformer + RandomForest
    ▼
models/satisfaction_pipeline_v2.0.0.pkl  ← se aplica mes a mes (predicción)
    │  metrics.technical_metrics / business_metrics
    ▼
reports/metrics_sprint2.json · monthly_backtest.csv · feature_selection_report.csv
```

### Split temporal (incorporación mensual)

| Partición | Meses | Propósito |
|-----------|-------|-----------|
| **train** | hasta `2018-03` | Entrenamiento |
| **val** | `2018-04` | Validación / ajuste |
| **backtest** | `2018-05`, `2018-06`, `2018-07` | Evaluación *out-of-time* mes a mes |
| **live** | `2018-08` | Último mes con label (simula producción) |
| **predict** | `2018-09` | Mes nuevo sin label → se predice |

---

## 4. Estructura del proyecto

```
sprint2/
├── src/                          # Paquete modular del pipeline
│   ├── config.py                 # Configuración central (rutas, split, umbrales, versión)
│   ├── utils.py                  # Carga de datos, master table, split temporal, E/S
│   ├── features.py               # Paso 2-3: target + ingeniería de variables
│   ├── cleaning.py               # Paso 6: limpieza fit/transform (clipado, NaN, agrupamiento)
│   ├── selection.py              # Paso 7: cascada de selección (RF + AUC, PSI, WOE/IV)
│   ├── pipeline.py               # Paso 5-6: Pipeline sklearn + persistencia (pickle)
│   └── metrics.py                # Métricas técnicas y de negocio
├── notebooks/
│   └── e_commerce_sprint2.ipynb  # ENTREGABLE PRINCIPAL (pipeline completo ejecutado)
├── data/
│   ├── raw/                      # CSV originales de Kaggle (ver sección 6)
│   └── processed/                # master_table.csv generada
├── models/                       # Pipelines serializados (.pkl versionados)
├── reports/
│   ├── figures/                  # Figuras del notebook
│   ├── feature_selection_report.csv
│   ├── monthly_backtest.csv
│   └── metrics_sprint2.json
├── run_pipeline.py               # Orquestador CLI (ejecución end-to-end / mensual)
├── generate_synthetic_data.py    # SOLO DEMO: datos sintéticos para probar sin Kaggle
├── requirements.txt
└── README.md
```

---

## 5. Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opción A) Colocar los CSV reales de Kaggle en data/raw/   ← producción
#    (Opción B) Generar datos de demostración para probar:
python generate_synthetic_data.py

# 3. Ejecutar el pipeline completo
python run_pipeline.py --rebuild

# 4. O abrir el notebook (entregable principal)
jupyter notebook notebooks/e_commerce_sprint2.ipynb
```

---

## 6. Datos

El proyecto usa el **Olist Brazilian E-Commerce Dataset**:
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

Descarga los CSV y colócalos en `data/raw/`:

```
olist_orders_dataset.csv          olist_order_reviews_dataset.csv
olist_order_items_dataset.csv     olist_order_payments_dataset.csv
olist_customers_dataset.csv       olist_products_dataset.csv
olist_sellers_dataset.csv         product_category_name_translation.csv
```

> **Nota.** Como el entorno de desarrollo no tiene acceso a Kaggle, el repositorio incluye
> `generate_synthetic_data.py`, que produce CSV sintéticos con **el mismo esquema** y relaciones
> realistas (el retraso de entrega reduce la satisfacción). Sirve para validar el pipeline de
> extremo a extremo. **En producción se reemplazan por los CSV reales sin tocar el código.**

---

## 7. Versionado del pipeline

`PIPELINE_VERSION` vive en `src/config.py`. Se incrementa con cada cambio de lógica.
Cada artefacto serializado lleva la versión en el nombre
(`satisfaction_pipeline_v2.0.0.pkl`) y cada corrida registra metadatos de trazabilidad
(timestamp, random_state, split, features) en `reports/metrics_sprint2.json`.

| Versión | Cambios |
|---------|---------|
| 2.0.0 | Sprint 2: pipeline modular, split temporal, selección en cascada, baseline RF. |

---

## 8. Próximo paso (Sprint 3)

Hiperparametrización con **Optuna**, comparación de modelos, validación cruzada y exportación
del **modelo final** con el **Target Final**.
