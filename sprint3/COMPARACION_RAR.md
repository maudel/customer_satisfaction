# Comparación: nuestro proyecto vs. `customer-satisfaction-ml` (.rar)

Comparación del repo de trabajo (`customer_satisfaction`, sprint3 v3.0.0) contra el
repo subido por el equipo (`customer-satisfaction-ml`, autor MVladyOM), y las mejoras
que se integraron a partir de este último.

## Qué tenía cada uno (antes de integrar)

| Aspecto | Nuestro sprint3 (antes) | `.rar` (repo del equipo) |
|---|---|---|
| Métrica objetivo | F1(clase 0) ✓ | F1(clase 0) ✓ |
| Panel de modelos | 8 (LR, RF, ET, HGB, LGBM, CatBoost, linearSVM, SGD) | 8 (mismos) |
| Búsqueda | Optuna TPE ✓ | Optuna TPE ✓ |
| Split | 4 vías temporal ✓ | 4 vías temporal ✓ |
| Codificación categóricas | One-Hot (concentraba importancia) | **Target encoding** (`te_*`) |
| Validación de robustez | val + backtest | **CV 5-fold + OOF** |
| Ensamble | — | **Stacking** (meta-learner LogReg sobre OOF) |
| Calibración de probas | sigmoide manual (linearSVM) | **CalibratedClassifierCV** (método tuneado) |
| Desbalance | class_weight | class_weight + **scale_pos_weight** |
| Early stopping | — | **CatBoost** con `od_type`/`od_wait` |
| Persistencia | 1 .pkl + métricas | **modelos/estudios/meta/oof/val .pkl + JSONs** |
| Datos | muestra sintética 40K | **datos reales (~95K) ya particionados** |

## Mejoras integradas del `.rar` a nuestro proyecto

1. **Target encoding de la categoría** (`te_product_category_name_english`): elimina el
   artefacto de importancia que tenía el One-Hot (una categoría dominaba >30%).
2. **CV 5-fold + OOF**: probabilidades out-of-fold por modelo, base honesta del stacking.
3. **Stacking**: meta-learner LogReg sobre las OOF de los 8 modelos (`run_real.py --finalize`).
4. **Calibración**: `CalibratedClassifierCV` para linearSVM con método (`sigmoid`/`isotonic`)
   como hiperparámetro tuneado por Optuna.
5. **scale_pos_weight** y espacios de búsqueda más ricos (CatBoost con early stopping,
   SGD multi-loss, ExtraTrees con `bootstrap`/`criterion`).
6. **Persistencia estilo repo**: `modelos_optuna.pkl`, `meta_learner.pkl`, `oof_probas.pkl`,
   `val_probas.pkl`, `mejores_params.json`, trial CSVs por modelo y `metrics_sprint3.json`.
7. **Datos reales**: se copiaron los splits de `data/master/` (X/y train/val/backtest/live)
   y el pipeline ahora corre sobre ellos (`run_real.py`).

## Lo que aportó nuestro proyecto y se mantuvo

- **Estabilidad como factor de selección** (`score = F1_val − caída(val→backtest)`):
  con datos reales revela que TODOS los modelos sobreajustan (caída 0.14–0.21), lo que
  conecta con el right-censoring documentado del backtest.
- **Auditoría de la regla del 15%** (`importancia.py`): permutación + cap + mitigación.
- Notebook ejecutable y figuras de defensa regenerables.

## Resultado tras integrar (datos reales)

- F1(cls0) validación: Extra Trees 0.5775 · RF 0.5763 · CatBoost 0.5673 · LightGBM 0.5639 ·
  HGB 0.5633 · SGD 0.5587 · LR 0.5572 · linearSVM 0.5372. AUC val ≈ 0.765.
- Stacking: F1(cls0) val 0.5613 · AUC 0.7652.
- Ganador por estabilidad: el modelo con menor caída val→backtest (más robusto).
- Importancia: la logística domina (entrega/retraso); 3 variables superan el 15% → se
  documenta y mitiga vía regularización + target encoding.

## Cómo reproducir

```bash
cd sprint3
# requiere data/master/ con los splits reales (X/y train/val/backtest)
python run_real.py --trials 100      # procesa el próximo modelo pendiente (resumible)
#   repetir hasta que no queden pendientes
python run_real.py --finalize        # stacking + importancia + artefactos
```
