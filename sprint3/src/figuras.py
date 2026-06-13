"""figuras.py · Gráficos de defensa del Sprint 3 (paleta del deck verde/azul)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C3

VERDE = "#34D399"; VERDE_OSC = "#047857"; TEAL = "#14B8A6"
AZUL = "#2563EB"; ROJO = "#DC2626"; GRIS = "#64748B"; AMBAR = "#D97706"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _save(fig, nombre):
    ruta = C3.DIR_FIGURAS / nombre
    fig.tight_layout(); fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig); return str(ruta)


def fig_comparacion_modelos(tabla: pd.DataFrame) -> str:
    """Barras horizontales de F1(cls0) en validación, ordenadas."""
    t = tabla.sort_values("f1_cls0_val")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colores = [VERDE if m == tabla.iloc[0]["modelo"] else "#A7F3D0" for m in t["modelo"]]
    ax.barh(t["modelo"], t["f1_cls0_val"], color=colores)
    for y, v in enumerate(t["f1_cls0_val"]):
        ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("F1(clase 0 · insatisfecho) en VALIDACIÓN")
    ax.set_title("Comparación de modelos post-Optuna — la métrica que define todo")
    ax.set_xlim(0, max(t["f1_cls0_val"]) * 1.12)
    return _save(fig, "s3_fig01_comparacion_f1cls0.png")


def fig_estabilidad(tabla: pd.DataFrame) -> str:
    """val vs backtest F1(cls0): la caída revela sobreajuste (estabilidad)."""
    t = tabla.sort_values("f1_cls0_val", ascending=False)
    x = np.arange(len(t)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w/2, t["f1_cls0_val"], w, label="VAL", color=TEAL)
    ax.bar(x + w/2, t["f1_cls0_backtest"], w, label="BACKTEST (out-of-time)", color=AZUL)
    for i, (_, r) in enumerate(t.iterrows()):
        caida = r["f1_cls0_val"] - r["f1_cls0_backtest"]
        col = ROJO if caida > C3.UMBRAL_INESTABILIDAD else VERDE_OSC
        ax.text(i, max(r["f1_cls0_val"], r["f1_cls0_backtest"]) + 0.01,
                f"−{caida:.3f}", ha="center", fontsize=8.5, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(t["modelo"], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("F1(clase 0)")
    ax.set_title("Estabilidad: caída F1(cls0) de validación a backtest (rojo = inestable)")
    ax.legend(frameon=False)
    return _save(fig, "s3_fig02_estabilidad.png")


def fig_score_estabilidad(tabla: pd.DataFrame) -> str:
    """Score final = F1 val penalizado por inestabilidad. El verdadero ranking."""
    t = tabla.sort_values("score_estabilidad")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colores = [VERDE if m == t.iloc[-1]["modelo"] else "#A7F3D0" for m in t["modelo"]]
    ax.barh(t["modelo"], t["score_estabilidad"], color=colores)
    for y, v in enumerate(t["score_estabilidad"]):
        ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Score = F1(cls0) val − penalización por caída a backtest")
    ax.set_title("Ranking corregido por estabilidad — el ganador real")
    ax.set_xlim(min(0, t["score_estabilidad"].min()), max(t["score_estabilidad"]) * 1.12)
    return _save(fig, "s3_fig03_score_estabilidad.png")


def fig_trials_optuna(historiales: dict[str, pd.DataFrame]) -> str:
    """Convergencia de Optuna: mejor F1(cls0) acumulado por modelo a lo largo de los trials."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(historiales)))
    for (nombre, h), col in zip(historiales.items(), cmap):
        h = h.sort_values("trial")
        ax.plot(h["trial"], h["f1_cls0_val"].cummax(), lw=2, label=nombre, color=col)
    ax.set_xlabel("trial de Optuna"); ax.set_ylabel("mejor F1(cls0) val acumulado")
    ax.set_title("Cómo Optuna llega al óptimo — convergencia por modelo (TPE)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, "s3_fig04_trials_optuna.png")


def fig_dispersion_trials(historial: pd.DataFrame, nombre: str) -> str:
    """Dispersión de los 100 trials del modelo ganador + frontera de mejora."""
    h = historial.sort_values("trial")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.scatter(h["trial"], h["f1_cls0_val"], s=20, color="#A7F3D0", alpha=0.8, label="trial")
    ax.plot(h["trial"], h["f1_cls0_val"].cummax(), color=VERDE_OSC, lw=2.2, label="mejor acumulado")
    ax.set_xlabel("trial"); ax.set_ylabel("F1(cls0) val")
    ax.set_title(f"Los {len(h)} trials de Optuna — {nombre}")
    ax.legend(frameon=False)
    return _save(fig, "s3_fig05_dispersion_trials.png")


def fig_importancia(tabla: pd.DataFrame, cap_pct: float, titulo: str, nombre_archivo: str) -> str:
    """Importancia por variable con la línea del 15 %; infractoras en rojo."""
    t = tabla.sort_values("importancia_pct").tail(18)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colores = [ROJO if v > cap_pct else TEAL for v in t["importancia_pct"]]
    ax.barh(t["variable"], t["importancia_pct"], color=colores)
    ax.axvline(cap_pct, color=AMBAR, ls="--", lw=1.8)
    ax.text(cap_pct + 0.2, 0.5, f"cap {cap_pct:.0f}%", color=AMBAR, fontsize=9, fontweight="bold")
    for y, v in enumerate(t["importancia_pct"]):
        ax.text(v + 0.15, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("importancia (%)"); ax.set_title(titulo)
    return _save(fig, nombre_archivo)


def fig_importancia_antes_despues(antes: pd.DataFrame, despues: pd.DataFrame, cap_pct: float) -> str:
    """Antes vs después de la mitigación del cap, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, tab, tit in [(axes[0], antes, "Antes de mitigar (viola 15 %)"),
                         (axes[1], despues, "Después de mitigar (cumple 15 %)")]:
        t = tab.sort_values("importancia_pct").tail(14)
        colores = [ROJO if v > cap_pct else TEAL for v in t["importancia_pct"]]
        ax.barh(t["variable"], t["importancia_pct"], color=colores)
        ax.axvline(cap_pct, color=AMBAR, ls="--", lw=1.6)
        for y, v in enumerate(t["importancia_pct"]):
            ax.text(v + 0.15, y, f"{v:.1f}%", va="center", fontsize=7.5)
        ax.set_title(tit); ax.set_xlabel("importancia (%)")
    fig.suptitle("Auditoría de la regla del 15 % — concentración de riesgo", fontsize=13, y=1.02)
    return _save(fig, "s3_fig06_importancia_cap.png")


def fig_radar_ganador(metricas_dict: dict, nombre: str) -> str:
    """Radar del modelo ganador en backtest (varias métricas a la vez)."""
    ejes = ["f1_cls0", "rec_cls0", "pre_cls0", "f1_cls1", "auc_roc", "accuracy"]
    etiquetas = ["F1 cls0", "Recall cls0", "Prec cls0", "F1 cls1", "AUC", "Accuracy"]
    vals = [metricas_dict.get(e, 0) for e in ejes]
    ang = np.linspace(0, 2*np.pi, len(ejes), endpoint=False).tolist()
    vals += vals[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(ang, vals, color=VERDE_OSC, lw=2.2)
    ax.fill(ang, vals, color=VERDE, alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_ylim(0, 1)
    for a, v in zip(ang[:-1], vals[:-1]):
        ax.text(a, v + 0.06, f"{v:.3f}", ha="center", fontsize=8.5, color=VERDE_OSC)
    ax.set_title(f"Modelo ganador en BACKTEST — {nombre}", y=1.08, fontsize=12)
    return _save(fig, "s3_fig07_radar_ganador.png")
