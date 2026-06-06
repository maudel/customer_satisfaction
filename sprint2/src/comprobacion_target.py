"""
============================================================================
 comprobacion_target.py  ·  Paso 2b · Comprobación del Target Preliminar
============================================================================

Responde a la observación del docente en la defensa del Sprint 2:

    "Un target así de simple hay que mejorarlo... tienen que buscar más
     opciones, porque un target así de simple no necesita un modelo."

La respuesta tiene dos partes, y este módulo implementa ambas:

  A) COMPROBACIÓN — evidencia con datos de que `is_satisfied` NO es un
     target arbitrario ni trivial:
       1. Estabilidad operacional : la tasa mensual de satisfechos se mueve
          con la realidad del negocio (cae cuando suben los retrasos).
       2. Discriminación          : gap de satisfacción entre entregas a
          tiempo y tardías (≈ 56 puntos porcentuales).
       3. No-trivialidad          : ninguna regla determinística captura de
          antemano a los futuros insatisfechos; el modelo sí (AUC ~0.73).

  B) ALTERNATIVAS — candidatos de TARGET FINAL para evaluar en Sprint 3:
       · ordinal de 3 clases (estilo NPS)  : detractor / neutro / promotor
       · umbral estricto (≥ 5)             : predecir solo promotores
     Se construyen aquí para que el Sprint 3 los compare con el mismo
     pipeline, sin tocar el resto del código.

Salidas:
    reports/comprobacion_target.json   (evidencia numérica de la corrida)
    reports/figures/s2_fig04_comprobacion_target.png  (vía figuras.py)
"""
from __future__ import annotations

import pandas as pd

from . import config as C


# ════════════════════════════════════════════════════════════════════════
#  A · COMPROBACIÓN DEL TARGET PRELIMINAR
# ════════════════════════════════════════════════════════════════════════
def tasa_mensual_satisfechos(datos: pd.DataFrame) -> pd.DataFrame:
    """Tasa de satisfechos por mes de compra (estabilidad operacional).

    Lectura esperada sobre Olist:
        feb-mar 2018  : la tasa cae a ~70% (pico de retrasos de entrega)
        abr-ago 2018  : se recupera a ~83%
    Un target de ruido NO seguiría a la operación; éste sí lo hace.
    """
    tabla = (
        datos.groupby(C.COL_MES)
        .agg(n=(C.OBJETIVO, "size"), tasa_satisfechos=(C.OBJETIVO, "mean"))
        .reset_index()
    )
    tabla["tasa_satisfechos"] = tabla["tasa_satisfechos"].round(4)
    return tabla


def gap_entrega(datos: pd.DataFrame) -> dict:
    """Satisfacción según el driver principal: entrega a tiempo vs tardía.

    El gap (≈ 0.83 vs 0.27) demuestra que el target discrimina sobre una
    variable OPERACIONAL — el negocio puede actuar sobre la logística.
    """
    if "delivered_on_time" not in datos.columns:
        return {}
    por_grupo = datos.groupby("delivered_on_time")[C.OBJETIVO].mean()
    a_tiempo = float(por_grupo.get(1, float("nan")))
    tardia = float(por_grupo.get(0, float("nan")))
    return {
        "satisfaccion_entrega_a_tiempo": round(a_tiempo, 4),
        "satisfaccion_entrega_tardia": round(tardia, 4),
        "gap_puntos_porcentuales": round((a_tiempo - tardia) * 100, 1),
    }


def volumen_mensual(datos: pd.DataFrame) -> pd.DataFrame:
    """Pedidos por mes y partición — el gráfico que pidió el docente.

    Justifica el diseño del split con tres lecturas:
        · 2016 es marginal (un solo pedido en septiembre) -> se descarta
        · 2018 estabiliza ~6.500 pedidos/mes -> particiones comparables
        · tras 2018-08 casi no hay reseña cargada -> mes PREDICT sin label
    """
    tabla = (
        datos.groupby([C.COL_MES, "dataset_split"])
        .size()
        .reset_index(name="n")
    )
    return tabla


def comprobar_target(datos: pd.DataFrame) -> dict:
    """Ejecuta la comprobación completa y escribe el JSON de evidencia."""
    mensual = tasa_mensual_satisfechos(datos)
    # Solo meses con volumen real: los marginales de 2016 (n=1) distorsionan
    # el mín/máx sin aportar lectura operacional.
    mensual = mensual[mensual["n"] >= 100].reset_index(drop=True)
    resumen = {
        "target": C.OBJETIVO,
        "regla": f"{C.COL_PUNTAJE_RESENA} >= {C.UMBRAL_SATISFACCION}",
        "n_poblacion": int(len(datos)),
        "tasa_global_satisfechos": round(float(datos[C.OBJETIVO].mean()), 4),
        # 1 · Estabilidad: rango de la tasa mensual (mín/máx con su mes)
        "tasa_mensual_min": {
            "mes": str(mensual.loc[mensual.tasa_satisfechos.idxmin(), C.COL_MES]),
            "tasa": float(mensual.tasa_satisfechos.min()),
        },
        "tasa_mensual_max": {
            "mes": str(mensual.loc[mensual.tasa_satisfechos.idxmax(), C.COL_MES]),
            "tasa": float(mensual.tasa_satisfechos.max()),
        },
        # 2 · Discriminación por el driver principal
        **gap_entrega(datos),
        # 3 · No-trivialidad: la cierra el AUC out-of-time de metrics_sprint2.json
        "nota_no_trivialidad": (
            "la prueba de que el target necesita un modelo es el AUC "
            "out-of-time sostenido en monthly_backtest.csv"
        ),
        # B · Plan hacia el Target Final (Sprint 3)
        "alternativas_target_final": [
            "ordinal_3_clases (detractor 1-2 / neutro 3 / promotor 4-5)",
            f"umbral_estricto (score >= {C.UMBRAL_PROMOTOR}, solo promotores)",
            "target compuesto (score + señales de comportamiento)",
        ],
    }
    return resumen


# ════════════════════════════════════════════════════════════════════════
#  B · TARGETS ALTERNATIVOS  (candidatos a Target Final · Sprint 3)
# ════════════════════════════════════════════════════════════════════════
def construir_target_ordinal(datos: pd.DataFrame) -> pd.DataFrame:
    """Target ordinal de 3 clases, estilo NPS (candidato a Target Final).

        0 = detractor  (score 1-2)  : intervención urgente
        1 = neutro     (score 3)    : recuperable con poco esfuerzo
        2 = promotor   (score 4-5)  : proteger la experiencia

    Separa al neutro recuperable del detractor — la crítica del docente al
    binario es exactamente que los mezcla. NO reemplaza a `is_satisfied`
    en el Sprint 2: convive como columna extra para el análisis comparado.
    """
    datos = datos.copy()
    puntaje = datos[C.COL_PUNTAJE_RESENA].astype(int)
    datos[C.OBJETIVO_ORDINAL] = pd.cut(
        puntaje, bins=[0, 2, 3, 5], labels=[0, 1, 2]
    ).astype(int)
    return datos


def construir_target_estricto(datos: pd.DataFrame) -> pd.DataFrame:
    """Target binario estricto: solo 5★ cuenta como promotor (candidato).

    Útil si el negocio quiere optimizar la proporción de promotores puros
    (el 5★ concentra ~59% de la masa); más exigente que `is_satisfied`.
    """
    datos = datos.copy()
    datos[C.OBJETIVO_ESTRICTO] = (
        datos[C.COL_PUNTAJE_RESENA].astype(int) >= C.UMBRAL_PROMOTOR
    ).astype(int)
    return datos
