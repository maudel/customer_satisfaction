"""
============================================================================
 run_pipeline.py  ·  Orquestador del pipeline completo (script modularizado)
============================================================================

Ejecuta el flujo de trabajo de extremo a extremo y produce los artefactos:
    - data/processed/master_table.csv          (compartida en la raíz del repo)
    - reports/feature_selection_report.csv     (cascada con AUC y Gini por paso)
    - reports/comprobacion_target.json         (Paso 2b · evidencia del target)
    - reports/justificacion_features.csv       (Paso 3b · Gini/IV por candidata)
    - reports/tabla_lift.csv                   (priorización por riesgo · LIVE)
    - reports/sensibilidad_variables.csv       (ablación · con --sensibilidad)
    - reports/figures/s2_fig04..07_*.png       (figuras de defensa)
    - reports/metrics_sprint2.json · monthly_backtest.csv
    - models/satisfaction_pipeline_v<X>.pkl

Uso:
    python run_pipeline.py                 # ejecuta todo
    python run_pipeline.py --rebuild       # reconstruye la master desde raw/
    python run_pipeline.py --sensibilidad  # añade la ablación de has_comment

Este es el mismo flujo que el notebook, pensado para ejecución mensual
automatizada (cron / orquestador). Cada corrida queda versionada.

El cuerpo está partido en funciones-etapa (`_etapa_*`) para que se lea como
una receta: cada paso recibe lo del anterior y devuelve lo que necesita el
siguiente. `main()` solo encadena esas etapas.

Cambios v2.1.0 (mejoras post-defensa del Sprint 2):
    · Paso 2b — comprobación del target + targets alternativos (Sprint 3)
    · Cascada — Gini re-calculado por paso y display como DataFrame
    · Negocio — tabla de lift por percentil de riesgo (palanca del umbral)
    · Defensa — figuras reproducibles (volumen mensual, cascada, estabilidad)
    · Riesgo  — ablación opcional de variables en revisión (has_comment)
    · Features — Paso 3b: Gini univariante + IV por candidata, la evidencia
      que justifica cada familia del feature engineering (fig08)
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config as C
from src import (cleaning, comprobacion_target, features as F, figuras,
                 justificacion_features, metrics, pipeline, selection,
                 sensibilidad, utils)


def _banner(titulo: str) -> None:
    """Encabezado legible por etapa (la demo en vivo se sigue desde acá)."""
    print("\n" + "─" * 70)
    print(f"  {titulo}")
    print("─" * 70)


# ── Paso 4-5 · Master Table ──────────────────────────────────────────────
def _etapa_master_table(reconstruir: bool) -> pd.DataFrame:
    """Carga la master table desde disco, o la reconstruye desde los CSV crudos."""
    if reconstruir or not C.RUTA_MASTER_TABLE.exists():
        tablas = utils.cargar_csvs_crudos()
        master = utils.construir_master_table(tablas)
        utils.guardar_master_table(master)
    else:
        master = utils.cargar_master_table()
        print(f"Master Table cargada: {master.shape}")
    return master


# ── Paso 2-3 · Población, target, features + split temporal ──────────────
def _etapa_features_y_split(master: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables y asigna cada pedido a su partición temporal."""
    datos = F.construir_variables(master)
    # Targets alternativos (candidatos a Target Final · Sprint 3): conviven
    # como columnas extra sin tocar el target preliminar del Sprint 2.
    datos = comprobacion_target.construir_target_ordinal(datos)
    datos = comprobacion_target.construir_target_estricto(datos)
    datos = utils.asignar_split_temporal(datos)
    print("\nSplit temporal:")
    print(utils.resumen_split(datos).to_string())
    return datos


# ── Paso 2b · Comprobación del target (respuesta al docente) ─────────────
def _etapa_comprobacion_target(datos: pd.DataFrame) -> None:
    """Evidencia de que el target preliminar no es arbitrario ni trivial.

    Escribe el JSON de comprobación y las dos figuras de defensa asociadas:
    la tasa mensual de satisfechos y el volumen mensual por partición.
    """
    resumen = comprobacion_target.comprobar_target(datos)
    utils.escribir_json(resumen, C.DIR_REPORTES / "comprobacion_target.json")

    mensual = comprobacion_target.tasa_mensual_satisfechos(datos)
    figuras.fig_comprobacion_target(mensual)
    figuras.fig_volumen_mensual(comprobacion_target.volumen_mensual(datos))

    print(f"Target {C.OBJETIVO}: tasa global "
          f"{resumen['tasa_global_satisfechos']:.1%} · gap entrega "
          f"{resumen.get('gap_puntos_porcentuales', '—')} pp "
          f"(a tiempo vs tardía)")
    print("Comprobación escrita en reports/comprobacion_target.json "
          "+ figuras s2_fig04 / s2_fig05")


# ── Paso 6-7 · Limpieza + selección de variables ─────────────────────────
def _etapa_limpieza_y_seleccion(datos: pd.DataFrame):
    """Ajusta el limpiador en train y corre la cascada de selección.

    Devuelve (limpiador, train_limpio, val_limpio, variables) para que la
    ablación posterior reuse las MISMAS particiones limpias (comparación
    justa, sin re-transformar).
    """
    train = datos[datos.dataset_split == "train"].copy()
    val   = datos[datos.dataset_split == "val"].copy()

    limpiador = cleaning.LimpiadorDatos().ajustar(train)
    train_limpio = limpiador.transformar(train)
    val_limpio   = limpiador.transformar(val)

    # Paso 3b · Evidencia del feature engineering (Gini/IV por candidata,
    # sobre el MISMO train limpio que verá la cascada -> mismas escalas)
    evidencia = justificacion_features.tabla_justificacion(train_limpio)
    figuras.fig_justificacion_features(evidencia)
    print("\nJustificación del feature engineering (campeona por familia):")
    print(justificacion_features.resumen_por_familia(evidencia).to_string(index=False))

    selector = selection.SelectorVariables().ajustar(train_limpio, val_limpio)
    reporte = selector.mostrar_reporte()          # display como DataFrame
    reporte.to_csv(C.DIR_REPORTES / "feature_selection_report.csv", index=False)
    figuras.fig_cascada_seleccion(reporte)        # figura de la cascada

    variables = selector.variables_seleccionadas_
    print(f"\nVariables seleccionadas ({len(variables)}): {variables}")
    return limpiador, train_limpio, val_limpio, variables


# ── Paso 7b · Sensibilidad de variables en revisión (opcional) ───────────
def _etapa_sensibilidad(train_limpio, val_limpio, variables) -> None:
    """Ablación de has_comment (y futuras variables en revisión).

    Cierra la discusión de leakage prometida en la defensa: entrena con y
    sin la variable y decide con datos si su aporte de AUC es real.
    """
    tabla = sensibilidad.comparar_sin_variables(train_limpio, val_limpio, variables)
    print("\nSensibilidad de variables en revisión:")
    print(tabla.to_string(index=False))


# ── Paso 5-6 · Entrenamiento del pipeline baseline ───────────────────────
def _etapa_entrenar(datos: pd.DataFrame, limpiador, variables: list[str]):
    """Entrena el pipeline (preprocesador + RF) sobre el train limpio."""
    train = datos[datos.dataset_split == "train"].copy()
    train_limpio = limpiador.transformar(train)
    pipe = pipeline.construir_pipeline(variables)
    pipe.fit(train_limpio, train_limpio[C.OBJETIVO])
    return pipe


# ── Métricas finales (técnicas + negocio + lift) por partición ───────────
def _etapa_metricas(datos, limpiador, pipe, variables) -> dict:
    """Evalúa el modelo en validación, backtest y producción.

    Además de las métricas técnicas y de negocio, escribe la tabla de lift
    sobre el mes LIVE: la herramienta operativa para que soporte elija el
    punto de corte según su capacidad de contacto.
    """
    todas_metricas = {"technical": {}, "business": {}}
    for nombre_split in ["val", "backtest", "live"]:
        particion = datos[datos.dataset_split == nombre_split].copy()
        if particion.empty:
            continue
        particion_limpia = limpiador.transformar(particion)
        proba = pipe.predict_proba(particion_limpia[variables])[:, 1]
        todas_metricas["technical"][nombre_split] = metrics.metricas_tecnicas(
            particion_limpia[C.OBJETIVO], proba)
        todas_metricas["business"][nombre_split] = metrics.metricas_negocio(
            particion_limpia, proba)

        if nombre_split == "live":   # priorización por riesgo en producción
            lift = metrics.tabla_lift(particion_limpia, proba)
            lift.to_csv(C.DIR_REPORTES / "tabla_lift.csv", index=False)
            print("\nPriorización por riesgo (mes LIVE):")
            print(lift.to_string(index=False))
    return todas_metricas


# ── Simulación mensual (backtest mes a mes, como en producción) ──────────
def _etapa_backtest_mensual(datos, limpiador, pipe, variables) -> list[dict]:
    """Recorre mes a mes los meses de backtest + producción y mide el AUC."""
    mensual = []
    for mes in C.MESES_BACKTEST + [C.MES_PRODUCCION]:
        particion = datos[datos[C.COL_MES] == mes].copy()
        if particion.empty:
            continue
        particion_limpia = limpiador.transformar(particion)
        proba = pipe.predict_proba(particion_limpia[variables])[:, 1]
        fila = {"month": mes, **metrics.metricas_tecnicas(particion_limpia[C.OBJETIVO], proba)}
        mensual.append(fila)
    pd.DataFrame(mensual).to_csv(C.DIR_REPORTES / "monthly_backtest.csv", index=False)
    figuras.fig_estabilidad_mensual(mensual)      # figura de estabilidad
    return mensual


# ── Persistencia de artefactos (pickle + métricas en JSON) ───────────────
def _etapa_persistencia(pipe, variables, todas_metricas, mensual) -> None:
    """Guarda el pipeline versionado y el JSON de métricas con sus metadatos."""
    todas_metricas["monthly_backtest"] = mensual
    meta = utils.metadatos_ejecucion({"n_features": len(variables), "features": variables,
                                      "stage": "sprint2_baseline"})
    pipeline.guardar_pipeline(pipe, variables, meta)
    utils.escribir_json({**meta, "metrics": todas_metricas},
                        C.DIR_REPORTES / "metrics_sprint2.json")


def main(reconstruir: bool = False, correr_sensibilidad: bool = False) -> None:
    print("=" * 70)
    print(f"  PIPELINE SATISFACCIÓN OLIST · v{C.VERSION_PIPELINE}")
    print("=" * 70)

    _banner("Paso 4-5 · Master Table")
    master = _etapa_master_table(reconstruir)

    _banner("Paso 2-3 · Población, target y features + split temporal")
    datos = _etapa_features_y_split(master)

    _banner("Paso 2b · Comprobación del target preliminar")
    _etapa_comprobacion_target(datos)

    _banner("Paso 6-7 · Limpieza + cascada de selección")
    limpiador, train_limpio, val_limpio, variables = _etapa_limpieza_y_seleccion(datos)

    if correr_sensibilidad:
        _banner("Paso 7b · Sensibilidad de variables en revisión")
        _etapa_sensibilidad(train_limpio, val_limpio, variables)

    _banner("Paso 5-6 · Entrenamiento del pipeline baseline")
    pipe = _etapa_entrenar(datos, limpiador, variables)

    _banner("Evaluación · métricas técnicas, de negocio y lift")
    todas_metricas = _etapa_metricas(datos, limpiador, pipe, variables)

    _banner("Backtest mensual out-of-time")
    mensual = _etapa_backtest_mensual(datos, limpiador, pipe, variables)
    _etapa_persistencia(pipe, variables, todas_metricas, mensual)

    print("\nMétricas backtest (mes a mes):")
    print(pd.DataFrame(mensual)[["month", "roc_auc", "f1", "gini"]].to_string(index=False))
    print("\n✔ Pipeline ejecutado correctamente. Artefactos en reports/ y models/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Reconstruir la master table desde data/raw/")
    ap.add_argument("--sensibilidad", action="store_true",
                    help="Correr la ablación de variables en revisión (has_comment)")
    args = ap.parse_args()
    main(reconstruir=args.rebuild, correr_sensibilidad=args.sensibilidad)
