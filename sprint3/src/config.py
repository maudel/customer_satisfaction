"""
============================================================================
 config.py  ·  Sprint 3 · Stage 4: Hiperparametrización y Modelo Final
 Proyecto: Predicción de Satisfacción del Cliente (Olist)
============================================================================

Metodología (alineada a la defensa del grupo):
  · La métrica que define todo es F1 de la clase 0 (insatisfecho), NO el AUC
    global: el error caro para Olist es NO detectar a quien está por irse.
  · Optuna (sampler TPE) maximiza F1(clase 0) en VALIDACIÓN, 100 trials/modelo.
  · Split temporal de 4 vías. Regla de Oro: el BACKTEST/TEST no se toca hasta
    elegir el modelo final; se evalúa una sola vez.
  · Selección por F1(cls0) PERO penalizada por ESTABILIDAD (caída val->backtest):
    un modelo que sobreajusta validación no sirve aunque gane en validación.
  · Auditoría de importancia: ninguna variable debe superar el 15 % (concentrar
    el riesgo en una sola variable es frágil y suele esconder fuga).
"""
from pathlib import Path

VERSION_PIPELINE = "3.0.0"
SEMILLA          = 42

# Rutas (monorepo: data/ compartida en la raíz)
DIR_RAIZ      = Path(__file__).resolve().parents[1]
DIR_REPO      = Path(__file__).resolve().parents[2]
DIR_SPRINT2   = DIR_REPO / "sprint2"
DIR_DATOS     = DIR_REPO / "data"
DIR_PROCESADO = DIR_DATOS / "processed"
DIR_MODELOS   = DIR_RAIZ / "models"
DIR_REPORTES  = DIR_RAIZ / "reports"
DIR_FIGURAS   = DIR_REPORTES / "figures"
for _c in (DIR_MODELOS, DIR_REPORTES, DIR_FIGURAS):
    _c.mkdir(parents=True, exist_ok=True)

# Métrica y objetivo
OBJETIVO = "is_satisfied"   # 1 = satisfecho · 0 = insatisfecho (la clase cara)
CLASE_FOCO = 0              # F1 se mide sobre la clase 0 (insatisfecho)

# Optuna
N_TRIALS_OPTUNA = 100       # config del grupo: 100 trials por modelo
SAMPLER = "TPE"             # Tree-structured Parzen Estimator
METRICA_OPTUNA = "f1_clase0_val"

# Selección: estabilidad
# score_final = f1_val - PENALIZACION * max(0, f1_val - f1_backtest)
PENALIZACION_ESTABILIDAD = 1.0
UMBRAL_INESTABILIDAD = 0.05   # caida > 0.05 de F1(cls0) se marca como inestable

# Auditoría de importancia
CAP_IMPORTANCIA = 0.15        # ninguna variable deberia superar el 15 %

PERCENTILES_LIFT = [5, 10, 20]
