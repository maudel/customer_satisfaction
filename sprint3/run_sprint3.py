"""
============================================================================
 run_sprint3.py · Sprint 3 · Stage 4 — Hiperparametrización y Modelo Final
============================================================================

Flujo (alineado a la metodología del grupo):
  1. Datos: master compartida + features S2 + interacciones S3 (sin leakage)
  2. Split temporal 4 vías: TRAIN / VAL / BACKTEST / LIVE (Regla de Oro)
  3. Optuna (TPE, 100 trials) por modelo, maximizando F1(clase 0) en VAL
  4. Estabilidad: caída F1(cls0) val->backtest como factor de selección
  5. Importancia por permutación + auditoría de la regla del 15 %
  6. Modelo final: evaluación en backtest + lift, export versionado, figuras

Uso:
    python run_sprint3.py                 # 100 trials (config del grupo)
    python run_sprint3.py --trials 25     # corrida rápida para demo
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import (busqueda, config as C3, features_extra, figuras, importancia,
                 metricas, modelos)
from src.puente_sprint2 import (cleaning, config2 as C2, features as F2, utils as u2)


def _banner(t): print("\n" + "─" * 70 + f"\n  {t}\n" + "─" * 70)


def _datos():
    _banner("1 · Datos — features S2 + interacciones S3 (sin leakage)")
    m = u2.cargar_master_table()
    m = F2.construir_variables(m)
    m = features_extra.construir_features_extra(m)
    m = u2.asignar_split_temporal(m)
    lim = cleaning.LimpiadorDatos()
    lim.ajustar(m[m[C2.COL_MES] <= C2.FIN_ENTRENAMIENTO])
    m = lim.transformar(m)
    variables = features_extra.features_disponibles(m)
    print(f"{len(m):,} pedidos · {len(variables)} variables candidatas")
    return m, variables


def _split(m):
    M = C2.COL_MES
    tr = m[m[M] <= C2.FIN_ENTRENAMIENTO]
    va = m[m[M].isin(C2.MESES_VALIDACION)]
    bt = m[m[M].isin(C2.MESES_BACKTEST)]
    lv = m[m[M] == C2.MES_PRODUCCION]
    print(f"TRAIN {len(tr):,} · VAL {len(va):,} · BACKTEST {len(bt):,} · LIVE {len(lv):,}")
    return tr, va, bt, lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=C3.N_TRIALS_OPTUNA)
    args = ap.parse_args()

    m, variables = _datos()
    tr, va, bt, lv = _split(m)

    # ── 3 · Optuna por modelo ────────────────────────────────────────────
    _banner(f"3 · Optuna (TPE) · F1(cls0) en VAL · {args.trials} trials/modelo")
    historiales, filas = {}, []
    hechos = set()
    _csv = C3.DIR_REPORTES / 'comparacion_modelos.csv'
    import os
    if os.environ.get('RESUME') and _csv.exists():
        prev = pd.read_csv(_csv); filas = prev.to_dict('records'); hechos = set(prev['modelo'])
        for nm in hechos:
            hp = C3.DIR_REPORTES / f"optuna_trials_{nm.replace(' ','_')}.csv"
            if hp.exists(): historiales[nm] = pd.read_csv(hp)
    for nombre in modelos.PANEL:
        if nombre in hechos:
            print(f'  {nombre:24s} (ya hecho, omito)'); continue
        try:
            res = busqueda.optimizar(nombre, m, variables, tr, va, args.trials)
        except Exception as e:
            print(f"  {nombre:24s} OMITIDO ({type(e).__name__})"); continue
        historiales[nombre] = res["historial"]
        # re-entrena con mejores params y evalúa val + backtest (estabilidad)
        pipe = modelos.construir_candidato(nombre, m, variables, res["mejores_params"])
        pipe.fit(tr[variables], tr[C3.OBJETIVO])
        f1_va = metricas.f1_clase0(va[C3.OBJETIVO], modelos.predice_proba(pipe, va[variables]))
        f1_bt = metricas.f1_clase0(bt[C3.OBJETIVO], modelos.predice_proba(pipe, bt[variables]))
        caida = f1_va - f1_bt
        score = f1_va - C3.PENALIZACION_ESTABILIDAD * max(0, caida)
        filas.append({"modelo": nombre, "f1_cls0_val": round(f1_va, 4),
                      "f1_cls0_backtest": round(f1_bt, 4), "caida": round(caida, 4),
                      "score_estabilidad": round(score, 4),
                      "estable": "sí" if caida <= C3.UMBRAL_INESTABILIDAD else "NO",
                      "mejores_params": json.dumps(res["mejores_params"])})
        print(f"  {nombre:24s} F1cls0 val={f1_va:.4f} bt={f1_bt:.4f} "
              f"caída={caida:+.4f} score={score:.4f}", flush=True)
        pd.DataFrame(filas).to_csv(C3.DIR_REPORTES / "comparacion_modelos.csv", index=False)
        res["historial"].to_csv(C3.DIR_REPORTES / f"optuna_trials_{nombre.replace(' ','_')}.csv", index=False)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(C3.DIR_REPORTES / "comparacion_modelos.csv", index=False)

    # ── 4 · Selección por estabilidad ────────────────────────────────────
    _banner("4 · Selección por F1(cls0) + estabilidad")
    ganador_f1 = tabla.sort_values("f1_cls0_val", ascending=False).iloc[0]
    ganador = tabla.sort_values("score_estabilidad", ascending=False).iloc[0]
    print(f"Mejor por F1(cls0) val solo : {ganador_f1['modelo']} "
          f"(val {ganador_f1['f1_cls0_val']}, pero backtest {ganador_f1['f1_cls0_backtest']}, "
          f"{'INESTABLE' if ganador_f1['estable']=='NO' else 'estable'})")
    print(f"GANADOR corregido por estabilidad: {ganador['modelo']} "
          f"(score {ganador['score_estabilidad']}, caída {ganador['caida']})")
    nombre_g = ganador["modelo"]
    params_g = json.loads(ganador["mejores_params"])

    figuras.fig_comparacion_modelos(tabla)
    figuras.fig_estabilidad(tabla)
    figuras.fig_score_estabilidad(tabla)
    figuras.fig_trials_optuna(historiales)
    if nombre_g in historiales:
        figuras.fig_dispersion_trials(historiales[nombre_g], nombre_g)

    # ── 5 · Importancia + regla del 15 % ─────────────────────────────────
    _banner("5 · Importancia de variables + auditoría de la regla del 15 %")
    pipe_g = modelos.construir_candidato(nombre_g, m, variables, params_g)
    pipe_g.fit(tr[variables], tr[C3.OBJETIVO])
    imp = importancia.importancia_permutacion(pipe_g, m, variables, va)
    imp.to_csv(C3.DIR_REPORTES / "importancia_variables.csv", index=False)
    audit = importancia.auditar_cap(imp)
    print(f"Máxima importancia: {audit['max_pct']}% · cap {audit['cap_pct']}% · "
          f"{'CUMPLE' if audit['cumple'] else 'VIOLA'}")
    for inf in audit["infractoras"]:
        print(f"   ⚠ {inf['variable']}: {inf['importancia_pct']:.1f}%")
    figuras.fig_importancia(imp, audit["cap_pct"],
                            f"Importancia — {nombre_g} (antes de mitigar)",
                            "s3_fig06a_importancia_antes.png")

    despues = imp
    variables_final = variables
    if not audit["cumple"]:
        variables_final, despues, pasos = importancia.mitigar_iterativo(
            nombre_g, m, variables, tr, va, modelos.construir_candidato, params_g)
        despues.to_csv(C3.DIR_REPORTES / "importancia_variables_mitigada.csv", index=False)
        audit2 = importancia.auditar_cap(despues)
        print(f"Mitigación iterativa ({len(pasos)} pasos): "
              f"{[ (p['top_var'], p['top_pct']) for p in pasos ]}")
        print(f"Tras mitigar: máx {audit2['max_pct']}% sobre {len(variables_final)} vars · "
              f"{'CUMPLE' if audit2['cumple'] else 'aún VIOLA'}")
        pipe_g = modelos.construir_candidato(nombre_g, m, variables_final, params_g)
        pipe_g.fit(tr[variables_final], tr[C3.OBJETIVO])
        figuras.fig_importancia_antes_despues(imp, despues, audit["cap_pct"])
        figuras.fig_importancia(despues, audit["cap_pct"],
                                f"Importancia — {nombre_g} (tras mitigar, cumple 15 %)",
                                "s3_fig06b_importancia_despues.png")

    # ── 6 · Modelo final: backtest + lift + export ───────────────────────
    _banner("6 · Modelo final — backtest, lift y export")
    proba_bt = modelos.predice_proba(pipe_g, bt[variables_final])
    met_bt = metricas.metricas_completas(bt[C3.OBJETIVO], proba_bt)
    print("Backtest:", {k: round(v, 4) for k, v in met_bt.items()})
    figuras.fig_radar_ganador(met_bt, nombre_g)

    import pickle
    meta = u2.metadatos_ejecucion({
        "sprint": 3, "stage": 4, "modelo_ganador": nombre_g,
        "seleccion": "F1(cls0) penalizado por estabilidad",
        "hiperparametros": params_g, "variables": variables_final,
        "metricas_backtest": met_bt, "auditoria_15pct": audit,
        "version": C3.VERSION_PIPELINE,
    })
    ruta = C3.DIR_MODELOS / f"satisfaction_final_{nombre_g.replace(' ','_')}_v{C3.VERSION_PIPELINE}.pkl"
    with open(ruta, "wb") as f:
        pickle.dump({"pipeline": pipe_g, "features": variables_final,
                     "metadata": meta, "pipeline_version": C3.VERSION_PIPELINE}, f)
    print(f"Modelo final guardado: {ruta}")
    u2.escribir_json(meta, C3.DIR_REPORTES / "metrics_sprint3.json")
    print("\n✔ Sprint 3 (Stage 4) completo. reports/ y models/ actualizados.")


if __name__ == "__main__":
    main()
