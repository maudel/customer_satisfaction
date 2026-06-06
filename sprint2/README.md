# Sprint 2 · Pipeline de Predicción de Satisfacción del Cliente (Olist)

**Mauricio De La Quintana** · Caso de estudio #1 · *Implementación de Soluciones de IA Aplicada a los Negocios*


---

## 1. Objetivo

Construir un **pipeline de datos robusto, reproducible y escalable** para predecir la
satisfacción del cliente sobre el dataset *Olist Brazilian E-Commerce*, contemplando la
**incorporación mensual de nuevos datos**, y definir el **Target Preliminar**.


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
    │  features.construir_variables()    ← Pasos 2-3: población + target + features
    ▼
[Master Table con features + target]
    │  utils.asignar_split_temporal()    ← Paso 4: split temporal por mes
    ▼
 train · val · backtest · live · predict
    │  cleaning.LimpiadorDatos (ajustar en train) ← Paso 6: clipado, NaN, agrupamiento
    ▼
[Particiones limpias]
    │  selection.SelectorVariables()     ← Paso 7: cascada faltantes→PSI→corr→univariante→WOE
    ▼
[Features seleccionadas]
    │  pipeline.construir_pipeline()     ← Pasos 5-6: ColumnTransformer + RandomForest
    ▼
models/satisfaction_pipeline_v2.0.0.pkl  ← se aplica mes a mes (predicción)
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

Este Sprint vive dentro del monorepo `customer-satisfaction-ml`. Los **datos
son compartidos** en la raíz del repo (`../data/`, los CSV reales de Olist que
ya usó el Sprint 1); los **artefactos propios** del Sprint 2 (modelo y reportes)
se escriben localmente bajo `sprint2/`.

```
customer-satisfaction-ml/
├── data/                         # ← DATA COMPARTIDA del repo (raíz)
│   ├── raw/                      #     CSV reales de Olist (ver sección 6)
│   └── processed/                #     master_table.csv generada (compartida)
└── sprint2/
    ├── src/                      # Paquete modular del pipeline
    │   ├── config.py             # Configuración central (rutas, split, umbrales, versión)
    │   ├── utils.py              # Carga de datos, master table, split temporal, E/S
    │   ├── features.py           # Paso 2-3: target + ingeniería de variables
    │   ├── cleaning.py           # Paso 6: limpieza fit/transform (clipado, NaN, agrupamiento)
    │   ├── comprobacion_target.py# Paso 2b: comprobación del target + targets alternativos
    │   ├── selection.py          # Paso 7: cascada de selección (RF + AUC/Gini, PSI, WOE/IV)
    │   ├── sensibilidad.py       # Paso 7b: ablación de variables en revisión (has_comment)
    │   ├── figuras.py            # Figuras de defensa (volumen, target, cascada, estabilidad)
    │   ├── pipeline.py           # Paso 5-6: Pipeline sklearn + persistencia (pickle)
    │   └── metrics.py            # Métricas técnicas y de negocio
    ├── notebooks/
    │   └── e_commerce_sprint2.ipynb  # ENTREGABLE PRINCIPAL (pipeline completo ejecutado)
    ├── models/                   # Pipelines serializados (.pkl versionados)
    ├── reports/
    │   ├── figures/              # Figuras del notebook
    │   ├── feature_selection_report.csv
    │   ├── monthly_backtest.csv
    │   └── metrics_sprint2.json
    ├── run_pipeline.py           # Orquestador CLI (ejecución end-to-end / mensual)
    ├── generate_synthetic_data.py# SOLO DEMO: datos sintéticos para probar sin Kaggle
    ├── requirements.txt
    └── README.md
```

> Las rutas de datos/artefactos se resuelven en `src/config.py`: los CRUDOS y
> `processed/` apuntan a la `data/` compartida del repo (`parents[2]`), y
> `models/`/`reports/` quedan dentro de `sprint2/`. Ejecutá los comandos
> **desde la carpeta `sprint2/`**.

---

## 5. Cómo ejecutar

```bash
# (todo desde la carpeta sprint2/)
cd sprint2

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Datos: el repo ya trae los CSV reales de Olist en ../data/raw/ (compartidos).
#    Si no estuvieran, generá datos de demostración con el mismo esquema:
#    python generate_synthetic_data.py

# 3. Ejecutar el pipeline completo (lee ../data/raw/, escribe model+reports locales)
python run_pipeline.py --rebuild

# 3b. (opcional) incluir la ablación de variables en revisión (has_comment)
python run_pipeline.py --sensibilidad

# 4. O abrir el notebook (entregable principal)
jupyter notebook notebooks/e_commerce_sprint2.ipynb
```

---

## 6. Datos


Los CSV reales viven en la `data/raw/` **compartida en la raíz del repo**
(`../data/raw/` desde `sprint2/`). Si necesitás reponerlos, descargá de Kaggle:

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

`VERSION_PIPELINE` vive en `src/config.py`. Se incrementa con cada cambio de lógica.
Cada artefacto serializado lleva la versión en el nombre
(`satisfaction_pipeline_v2.0.0.pkl`) y cada corrida registra metadatos de trazabilidad
(timestamp, random_state, split, features) en `reports/metrics_sprint2.json`.

| Versión | Cambios |
|---------|---------|
| 2.0.0 | Sprint 2: pipeline modular, split temporal, selección en cascada, baseline RF. |
| 2.1.0 | Mejoras post-defensa: **Paso 2b** de comprobación del target + targets alternativos para el Target Final (`comprobacion_target.py`), **Gini re-calculado por paso** en la cascada con display como DataFrame, **tabla de lift** por percentil de riesgo (`tabla_lift.csv`), **ablación de `has_comment`** (`sensibilidad.py`, flag `--sensibilidad`) y **figuras de defensa** generadas por el pipeline (`figuras.py`: volumen mensual, comprobación del target, cascada, estabilidad mensual). |

---

---

## 7b. Mejoras v2.1.0 — trazables al feedback de la defensa

| Observación del docente (sesión Sprint 2) | Mejora en el código | Artefacto |
|---|---|---|
| "Un target así de simple hay que mejorarlo; busquen más opciones" | Paso 2b: comprobación con datos (estabilidad mensual, gap de entrega) + targets alternativos `satisfaction_class` (ordinal NPS) e `is_promoter` (≥5) listos para el Target Final | `comprobacion_target.json` · fig04 |
| "Ahí tienen que mostrar el gráfico" (volumen por mes) | Figura de pedidos por mes coloreada por partición del split | fig05 |
| "Eliminás algo, volvés a calcular… eliminás, volvés a calcular" | La cascada ya re-evaluaba por paso; ahora el reporte incluye **Gini train/val** por paso y su figura | `feature_selection_report.csv` · fig06 |
| "Prefiero un display, un DataFrame" | `SelectorVariables.mostrar_reporte()` imprime la tabla completa como DataFrame | salida de consola |
| Discusión de `has_comment` (frontera de leakage) | Ablación con el mismo evaluador de la cascada y regla de decisión por ΔAUC | `sensibilidad_variables.csv` |
| Priorización operativa ("¿a quién contacta soporte?") | Tabla de lift: captura/precisión/lift al contactar el top 5/10/20% de riesgo en LIVE | `tabla_lift.csv` |

---

## 8. Próximo paso (Sprint 3)

Hiperparametrización con **Optuna**, comparación de modelos (XGBoost vs RF vs LR), validación
cruzada temporal y exportación del **modelo final**. Para el **Target Final** se compararán,
con este mismo pipeline, el binario preliminar contra `satisfaction_class` (ordinal de 3
clases) e `is_promoter` (≥5), recogiendo el feedback del docente; la decisión se tomará por
AUC/Gini out-of-time + accionabilidad para soporte.
