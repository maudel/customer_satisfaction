"""
importancia.py · Importancia de variables por permutación + auditoría del 15 %.

Se usa importancia por PERMUTACIÓN (sobre VAL) porque es comparable entre los 8
modelos del panel (lineales, árboles y boosting) y mide impacto real en F1(cls0),
no el criterio interno de cada modelo. La regla de negocio del grupo:

    ninguna variable debe superar el 15 % de la importancia total.

Si una variable lo supera, el modelo concentra el riesgo en una sola señal
(frágil ante drift y sospechoso de fuga). Se reporta y se ofrece una mitigación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C3, metricas, modelos


def _scorer_f1cls0(pipe, X, y):
    return metricas.f1_clase0(y, modelos.predice_proba(pipe, X))


def importancia_permutacion(pipe, datos, variables, va, n_repeats: int = 5) -> pd.DataFrame:
    """Importancia por permutación normalizada a % (suma 100)."""
    Xva, yva = va[variables].copy(), va[C3.OBJETIVO]
    r = permutation_importance(pipe, Xva, yva, scoring=_scorer_f1cls0,
                               n_repeats=n_repeats, random_state=C3.SEMILLA, n_jobs=1)
    imp = np.clip(r.importances_mean, 0, None)
    total = imp.sum() or 1.0
    return (pd.DataFrame({"variable": variables, "importancia_pct": 100 * imp / total})
            .sort_values("importancia_pct", ascending=False).reset_index(drop=True))


def auditar_cap(tabla: pd.DataFrame, cap: float = C3.CAP_IMPORTANCIA) -> dict:
    """Verifica la regla del 15 %. Devuelve infractoras y veredicto."""
    cap_pct = cap * 100
    infractoras = tabla[tabla["importancia_pct"] > cap_pct]
    return {
        "cap_pct": cap_pct,
        "cumple": infractoras.empty,
        "max_pct": round(float(tabla["importancia_pct"].max()), 2),
        "infractoras": infractoras[["variable", "importancia_pct"]].to_dict("records"),
    }


def mitigar_iterativo(nombre_modelo, datos, variables, tr, va, constructor, params,
                      cap: float = C3.CAP_IMPORTANCIA, min_vars: int = 8, max_pasos: int = 6):
    """
    Mitigación iterativa de la regla del 15 %: mientras una variable supere el cap,
    se quita la infractora más dominante, se re-entrena y se vuelve a medir, hasta
    que ninguna supere el cap (o se llegue al mínimo de variables). Concentrar el
    riesgo en una variable es frágil; redistribuir la importancia da un modelo más
    robusto a drift. Devuelve (variables_final, tabla_final, pasos).
    """
    cap_pct = cap * 100
    vars_actual = list(variables)
    pasos = []
    for _ in range(max_pasos):
        pipe = constructor(nombre_modelo, datos, vars_actual, params)
        pipe.fit(tr[vars_actual], tr[C3.OBJETIVO])
        tabla = importancia_permutacion(pipe, datos, vars_actual, va)
        top = tabla.iloc[0]
        pasos.append({"n_vars": len(vars_actual), "top_var": top["variable"],
                      "top_pct": round(float(top["importancia_pct"]), 2)})
        if top["importancia_pct"] <= cap_pct or len(vars_actual) <= min_vars:
            return vars_actual, tabla, pasos
        vars_actual = [v for v in vars_actual if v != top["variable"]]
    return vars_actual, tabla, pasos
