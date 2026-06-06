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
    │  utils.cargar_csvs_crudos()
    ▼
[9 tablas relacionales]
    │  utils.construir_master_table()    ← Pasos 4-5: une y agrega a nivel order_id
    ▼
data/processed/master_table.csv
    │  features.construir_objetivo() + features.*  ← Pasos 2-3: población + target + features
    ▼
[Master Table con features + target]
    │  utils.asignar_split_temporal()    ← Paso 4: split temporal por mes
    ▼
 train · val · backtest · live · predict
    │  cleaning.LimpiadorDatos (fit en train) ← Paso 6: clipado, NaN, agrupamiento
    ▼
[Particiones limpias]
    │  selection.SelectorVariables()     ← Paso 7: cascada faltantes→PSI→corr→univariante→WOE
    ▼
[Features seleccionadas]
    │  pipeline.construir_pipeline()     ← Pasos 5-6: ColumnTransformer + RandomForest
    ▼
sprint2/models/satisfaction_pipeline_v2.0.0.pkl  ← se aplica mes a mes (predicción)
    │  metrics.metricas_tecnicas / metricas_negocio
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
customer_satisfaction/                  # monorepo (data/ COMPARTIDA en la raíz)
├── data/
│   ├── raw/                            # CSV reales de Olist (9 datasets de Kaggle)
│   ├── processed/                      # master_table + features/parquets generados
│   └── splits/                         # particiones temporales
├── notebooks/                          # trabajo exploratorio organizado por sprint
│   ├── sprint_01_eda/                  # EDA crudo, master table, clientes premium
│   ├── sprint_02_pipeline/             # EDA master table, features, pipeline
│   ├── sprint_03_modeling/             # (Sprint 3 · pendiente)
│   ├── sprint_04_integration/          # (Sprint 4 · pendiente)
│   └── _legacy/                        # notebook monolítico original del Sprint 2
├── models/                             # pipelines serializados (.pkl versionados)
├── sprint2/                            # ◀ PAQUETE MODULAR DEL PIPELINE (v2.1.0)
│   ├── src/                            #   código del pipeline reproducible
│   │   ├── config.py                   #     config central (rutas, split, umbrales, versión)
│   │   ├── utils.py                    #     carga, master table, split temporal, E/S
│   │   ├── features.py                 #     Paso 2-3: target + ingeniería de variables
│   │   ├── cleaning.py                 #     Paso 6: limpieza fit/transform
│   │   ├── comprobacion_target.py      #     Paso 2b: comprobación + targets alternativos
│   │   ├── selection.py                #     Paso 7: cascada (RF+AUC/Gini, PSI, WOE/IV)
│   │   ├── sensibilidad.py             #     Paso 7b: ablación de variables en revisión
│   │   ├── figuras.py                  #     figuras de defensa generadas por el pipeline
│   │   ├── justificacion_features.py   #     justificación de features seleccionadas
│   │   ├── pipeline.py                 #     Paso 5-6: Pipeline sklearn + persistencia
│   │   └── metrics.py                  #     métricas técnicas y de negocio
│   ├── notebooks/e_commerce_sprint2.ipynb  # entregable ejecutado del paquete
│   ├── reports/                        #   métricas, figuras y CSVs del pipeline
│   ├── run_pipeline.py                 #   orquestador CLI (end-to-end / mensual)
│   ├── generate_synthetic_data.py      #   SOLO DEMO: datos sintéticos sin Kaggle
│   ├── requirements.txt
│   └── README.md                       #   documentación detallada del paquete
├── docs/  ·  docker/                   # documentación y contenedores (scaffolding)
├── requirements.txt
└── README.md
```

> **Sobre `sprint2/`.** Es el paquete modular y reproducible del pipeline (v2.1.0).
> Lee los CSV reales desde la `data/` **compartida en la raíz del repo**
> (`config.py` resuelve `parents[2]`) y escribe sus artefactos propios
> (modelos y reportes) **localmente bajo `sprint2/`**, de modo que el sprint es
> autocontenido sin duplicar los datos. Ver `sprint2/README.md` para el detalle
> de ejecución y las mejoras v2.1.0 trazadas al feedback de la defensa.

---

## 5. Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt        # o:  pip install -r sprint2/requirements.txt

# El pipeline modular vive en sprint2/ y lee la data/ compartida de la raíz.
cd sprint2

# 2. Datos: el repo ya trae los CSV reales de Olist en ../data/raw/ (compartidos).
#    Si faltaran, generá datos de demostración con el mismo esquema:
#    python generate_synthetic_data.py

# 3. Ejecutar el pipeline completo (lee ../data/raw/, escribe model+reports locales)
python run_pipeline.py --rebuild

# 3b. (opcional) incluir la ablación de variables en revisión (has_comment)
python run_pipeline.py --sensibilidad

# 4. O abrir el notebook entregable del paquete
jupyter notebook notebooks/e_commerce_sprint2.ipynb
```

> Los notebooks exploratorios por sprint están en `notebooks/sprint_01_eda/` y
> `notebooks/sprint_02_pipeline/`.

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

`VERSION_PIPELINE` vive en `sprint2/src/config.py`. Se incrementa con cada cambio de lógica.
Cada artefacto serializado lleva la versión en el nombre
(`satisfaction_pipeline_v2.0.0.pkl`) y cada corrida registra metadatos de trazabilidad
(timestamp, random_state, split, features) en `sprint2/reports/metrics_sprint2.json`.

| Versión | Cambios |
|---------|---------|
| 2.0.0 | Sprint 2: pipeline modular, split temporal, selección en cascada, baseline RF. |
| 2.1.0 | Mejoras post-defensa: **Paso 2b** de comprobación del target + targets alternativos (`comprobacion_target.py`), **Gini re-calculado por paso** en la cascada, **tabla de lift** por percentil de riesgo (`tabla_lift.csv`), **ablación de `has_comment`** (`sensibilidad.py`, flag `--sensibilidad`) y **figuras de defensa** generadas por el pipeline (`figuras.py`). Detalle completo en `sprint2/README.md`. |

---

## 8. Próximo paso (Sprint 3)

Hiperparametrización con **Optuna**, comparación de modelos, validación cruzada y exportación
del **modelo final** con el **Target Final**.
