"""
============================================================================
 sensibilidad.py  ·  Análisis de sensibilidad de variables en revisión
============================================================================

Implementa la ablación prometida en la defensa para `has_comment`, la
variable más discutible del set final:

    El comentario nace en el mismo acto que la puntuación, así que está en
    la FRONTERA del leakage. La defensa de mantenerla en el baseline es que
    pasó toda la cascada y que, al momento del scoring, Olist sí sabe si
    existe texto. La cautela es ESTE módulo: entrenar con y sin ella y
    medir cuánto AUC aporta realmente.

Regla de decisión (documentada para el Sprint 3):
    ΔAUC_val < 0.005  : la variable no es determinante -> puede salir del
                        set final sin costo, cerrando la discusión
    ΔAUC_val ≥ 0.005  : aporta señal real -> se mantiene, dejando registrada
                        la justificación del momento de predicción

Las variables bajo revisión se declaran en config.VARIABLES_EN_REVISION,
de modo que sumar otra al análisis es editar UNA lista.

Salida:
    reports/sensibilidad_variables.csv
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import selection


# ════════════════════════════════════════════════════════════════════════
#  ABLACIÓN  (mismo evaluador RF de la cascada -> comparación justa)
# ════════════════════════════════════════════════════════════════════════
def comparar_sin_variables(
    train: pd.DataFrame,
    val: pd.DataFrame,
    variables_base: list[str],
    en_revision: list[str] | None = None,
) -> pd.DataFrame:
    """Entrena el RF evaluador con y sin cada variable en revisión.

    Usa `selection.evaluar_variables` (el MISMO evaluador de la cascada)
    para que el ΔAUC sea comparable con la tabla de selección.

    Devuelve un DataFrame con una fila por escenario:
        escenario | variables | auc_val | gini_val | delta_auc_val | decision
    """
    en_revision = en_revision if en_revision is not None else C.VARIABLES_EN_REVISION

    # ── Escenario base: el set completo seleccionado por la cascada ──────
    auc_tr_base, auc_val_base = selection.evaluar_variables(train, val, variables_base)
    filas = [{
        "escenario": "set_completo",
        "variables": len(variables_base),
        "auc_val": round(auc_val_base, 4),
        "gini_val": round(2 * auc_val_base - 1, 4),
        "delta_auc_val": 0.0,
        "decision": "(referencia)",
    }]

    # ── Un escenario por variable en revisión: el set SIN esa variable ───
    for variable in en_revision:
        if variable not in variables_base:
            continue  # la cascada ya la descartó: no hay nada que ablacionar
        sin_var = [v for v in variables_base if v != variable]
        _, auc_val = selection.evaluar_variables(train, val, sin_var)
        delta = auc_val_base - auc_val
        filas.append({
            "escenario": f"sin_{variable}",
            "variables": len(sin_var),
            "auc_val": round(auc_val, 4),
            "gini_val": round(2 * auc_val - 1, 4),
            "delta_auc_val": round(delta, 4),
            "decision": ("aporta señal real -> se mantiene"
                         if delta >= C.TOLERANCIA_ABLACION
                         else "prescindible -> candidata a salir en Sprint 3"),
        })

    tabla = pd.DataFrame(filas)
    tabla.to_csv(C.DIR_REPORTES / "sensibilidad_variables.csv", index=False)
    return tabla
