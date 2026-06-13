# Sprint 3 · Stage 4 — Hiperparametrización y Modelo Final

**Proyecto:** Predicción de Satisfacción del Cliente (Olist) · **Pipeline:** `v3.0.0`

Cierra el Paso 8 del flujo de Modelos de Negocio. Reutiliza el paquete del
Sprint 2 (master compartida, limpieza fit-en-train, features) vía
`src/puente_sprint2.py` y añade las variables de interacción del Sprint 3.

## La métrica que define todo: F1 de la clase 0 (insatisfecho)

El ~21 % de los pedidos son insatisfechos (clase 0). El error caro para Olist no
es equivocarse con un satisfecho, sino **no detectar a quien está por irse**. Por
eso Optuna **maximiza F1(clase 0) en validación**, no accuracy ni AUC global.

## Diseño de validación — split de 4 vías + Regla de Oro

| Partición | Periodo | Rol |
|---|---|---|
| TRAIN | Sep 2016 – Mar 2018 | entrena cada trial |
| VAL | Abr 2018 | Optuna mide F1(cls0) aquí |
| BACKTEST | May – Jul 2018 | out-of-time; se evalúa **una sola vez** al final |
| LIVE | Ago 2018 | mes de producción simulado (lift) |

**Regla de Oro:** el BACKTEST no se toca durante la búsqueda — estimación honesta
de producción, sin contaminación.

## Qué incorpora este sprint (pedidos de la defensa)

1. **Panel de 8 modelos**: Logistic Regression, LightGBM, CatBoost, Random Forest,
   Extra Trees, Hist Gradient Boosting, linearSVM, SGD.
2. **Optuna TPE, 100 trials/modelo** (`N_TRIALS_OPTUNA` en `config.py`), objetivo
   F1(cls0) en VAL. Historial completo por modelo en `reports/optuna_trials_*.csv`.
3. **Estabilidad como factor de selección**: `score = F1(cls0)_val − penalización ·
   max(0, F1_val − F1_backtest)`. Un modelo que sobreajusta validación se descarta
   aunque gane ahí. (Caso real: Extra Trees 0.5692 val → 0.4077 test, −28 %.)
4. **Auditoría de importancia (regla del 15 %)**: importancia por permutación
   (comparable entre modelos) + `auditar_cap()` + `mitigar_iterativo()`: si una
   variable supera el 15 %, se quita la dominante y se redistribuye hasta cumplir.
5. **Muchas figuras** de comparación, estabilidad, trials, importancia y radar del
   ganador en `reports/figures/`.

## Cómo ejecutar

```bash
cd sprint3
pip install -r requirements.txt        # añade optuna, lightgbm, catboost
# requiere que el Sprint 2 haya corrido una vez (master compartida):
#   cd ../sprint2 && python run_pipeline.py --rebuild && cd ../sprint3
python run_sprint3.py                   # 8 modelos · 100 trials · estabilidad · 15 %
python run_sprint3.py --trials 25       # corrida rápida para demo
RESUME=1 python run_sprint3.py          # reanuda usando modelos ya calculados
```

## Estructura

```
sprint3/
├── src/
│   ├── config.py            # v3.0.0, métrica F1 cls0, 100 trials, cap 15 %, estabilidad
│   ├── puente_sprint2.py    # reutiliza el paquete del Sprint 2
│   ├── features_extra.py    # interacciones: interaccion_retraso_items, delay_ratio…
│   ├── modelos.py           # panel de 8 modelos + preprocesador propio por dtype
│   ├── busqueda.py          # Optuna TPE maximizando F1(cls0) en VAL
│   ├── metricas.py          # f1_clase0() y métricas completas (incl. Brier, Gini)
│   ├── importancia.py       # permutación + auditoría 15 % + mitigación iterativa
│   └── figuras.py           # figuras de defensa
├── notebooks/e_commerce_sprint3.ipynb
├── reports/                 # CSVs, figures/ y metrics_sprint3.json
├── models/                  # modelo final versionado (.pkl)
├── run_sprint3.py
└── README.md
```

## Nota sobre los datos

Los CSV de `data/raw/` son una **muestra sintética** de 40 000 pedidos con el
esquema de Olist. La **metodología es idéntica** con los datos reales: el deck de
defensa usa los números reales del grupo (≈95 000 pedidos); el código reproduce el
método sobre la muestra del repo. Reemplazá `data/raw/` con los CSV de Kaggle y
re-ejecutá: todos los reportes y figuras se regeneran sin tocar código.

| Versión | Cambios |
|---------|---------|
| 3.0.0 | Stage 4: F1(cls0), panel de 8 modelos, Optuna TPE 100 trials, estabilidad como factor de selección, auditoría de importancia (regla del 15 %), modelo final exportado. |

---

## Mejoras integradas del repo del equipo (`.rar`)

Tras comparar con `customer-satisfaction-ml` (ver `COMPARACION_RAR.md`) se integraron:
target encoding de la categoría, **CV 5-fold + OOF**, **stacking** (meta-learner LogReg),
**calibración** de probabilidades (linearSVM), `scale_pos_weight`, espacios de búsqueda
más ricos y persistencia de artefactos por modelo. El pipeline ahora corre sobre los
**datos reales** en `data/master/` mediante `run_real.py` (resumible, 1 modelo por vez):

```bash
python run_real.py --trials 100   # repetir hasta no quedar pendientes (resumible)
python run_real.py --finalize     # stacking + importancia + artefactos
```

Entregables nuevos: `GUION_ORADOR.md` (notas del orador), `COMPARACION_RAR.md`,
`reports/optuna_trials_csv/` (CSV de trials por modelo) y los `.pkl` de modelos/meta/OOF.
