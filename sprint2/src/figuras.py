"""
============================================================================
 figuras.py  ·  Figuras de defensa del Sprint 2
============================================================================

Genera, desde el propio pipeline, los gráficos que el docente pidió ver en
la sesión de defensas (en lugar de explicarlos verbalmente):

    s2_fig04_comprobacion_target.png : tasa mensual de satisfechos
                                       ("comprobación del target")
    s2_fig05_volumen_mensual.png     : pedidos por mes coloreados por
                                       partición ("muestren el gráfico")
    s2_fig06_cascada_seleccion.png   : features vs AUC/Gini por paso
                                       (re-cálculo tras cada eliminación)
    s2_fig07_estabilidad_mensual.png : AUC/Gini/F1 mes a mes (backtest)
    s2_fig08_justificacion_features.png : Gini univariante por variable,
                                       coloreado por familia (Paso 3b)

Todas se escriben en reports/figures/ con dpi 150 y se regeneran en cada
corrida — son artefactos reproducibles, igual que el resto del pipeline.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla: apto para cron / CI
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

# Paleta de la presentación (un color por partición del split temporal)
COLORES_SPLIT = {
    "train":    "#4C1D95",   # violeta oscuro
    "val":      "#6D28D9",   # violeta
    "backtest": "#9333EA",   # púrpura
    "live":     "#DB2777",   # magenta
    "predict":  "#F9A8D4",   # rosa (mes sin label)
    "fuera_de_rango": "#D8CCEE",
}


def _guardar(fig, nombre: str) -> str:
    """Guarda la figura en reports/figures/ y devuelve la ruta como str."""
    ruta = C.DIR_FIGURAS / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(ruta)


# ════════════════════════════════════════════════════════════════════════
#  FIG 04 · Comprobación del target (estabilidad mensual de la tasa)
# ════════════════════════════════════════════════════════════════════════
def fig_comprobacion_target(tasa_mensual: pd.DataFrame) -> str:
    """Tasa mensual de satisfechos: el target sigue a la operación.

    La caída a ~70% en feb-mar 2018 (pico de retrasos) y la recuperación
    posterior son la evidencia visual de que el target NO es ruido.
    """
    tabla = tasa_mensual[tasa_mensual["n"] >= 100]  # meses con volumen real
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(tabla[C.COL_MES], tabla["tasa_satisfechos"] * 100,
            "o-", color="#DB2777", lw=2)
    ax.axhline(tabla["tasa_satisfechos"].mean() * 100,
               color="#6D28D9", ls="--", lw=1, label="media del período")
    ax.set_title("Comprobación del target · tasa mensual de satisfechos",
                 fontweight="bold")
    ax.set_xlabel("Mes de compra"); ax.set_ylabel("% satisfechos")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return _guardar(fig, "s2_fig04_comprobacion_target.png")


# ════════════════════════════════════════════════════════════════════════
#  FIG 05 · Volumen mensual por partición (el gráfico que pidió el docente)
# ════════════════════════════════════════════════════════════════════════
def fig_volumen_mensual(volumen: pd.DataFrame) -> str:
    """Pedidos por mes, coloreados según la partición del split temporal.

    Justifica con datos (no verbalmente) tres decisiones del split:
        2016 marginal · régimen estable ~6.5K/mes · sin labels tras 2018-08
    """
    pivote = (volumen.pivot(index=C.COL_MES, columns="dataset_split", values="n")
              .fillna(0))
    orden = [p for p in COLORES_SPLIT if p in pivote.columns]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    base = pd.Series(0.0, index=pivote.index)
    for particion in orden:
        ax.bar(pivote.index.astype(str), pivote[particion], bottom=base,
               label=particion, color=COLORES_SPLIT[particion],
               edgecolor="white", linewidth=0.4)
        base = base + pivote[particion]
    ax.set_title("Volumen mensual de pedidos por partición del split",
                 fontweight="bold")
    ax.set_xlabel("Mes de compra"); ax.set_ylabel("Nº de pedidos")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="partición")
    return _guardar(fig, "s2_fig05_volumen_mensual.png")


# ════════════════════════════════════════════════════════════════════════
#  FIG 06 · Cascada de selección (re-cálculo tras cada eliminación)
# ════════════════════════════════════════════════════════════════════════
def fig_cascada_seleccion(reporte: pd.DataFrame) -> str:
    """Features que sobreviven vs AUC/Gini re-calculados en cada paso.

    Visualiza el método que el docente pidió en la defensa: "primero
    calculás con toda la población; eliminás algo, volvés a calcular".
    """
    fig, ax1 = plt.subplots(figsize=(11, 4.6))
    ax1.bar(reporte["metodo"], reporte["features"],
            color="#C9B8E8", edgecolor="white")
    for i, v in enumerate(reporte["features"]):
        ax1.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    ax1.set_ylabel("Nº de features"); ax1.set_xlabel("Paso de la cascada")
    ax1.tick_params(axis="x", rotation=20)

    ax2 = ax1.twinx()
    ax2.plot(reporte["metodo"], reporte["auc_roc_val"], "o-",
             color="#6D28D9", lw=2, label="AUC val")
    ax2.plot(reporte["metodo"], reporte["gini_val"], "s--",
             color="#DB2777", lw=2, label="Gini val")
    ax2.set_ylabel("AUC / Gini (val)"); ax2.legend(loc="lower left")
    ax1.set_title("Cascada de selección · re-cálculo tras cada eliminación",
                  fontweight="bold")
    return _guardar(fig, "s2_fig06_cascada_seleccion.png")


# ════════════════════════════════════════════════════════════════════════
#  FIG 07 · Estabilidad mensual del modelo (backtest out-of-time)
# ════════════════════════════════════════════════════════════════════════
def fig_estabilidad_mensual(mensual: list[dict] | pd.DataFrame) -> str:
    """AUC, Gini y F1 mes a mes: la banda natural del modelo sin re-entrenar.

    La banda observada (~±2 pp de AUC) es el umbral de alarma del monitoreo
    de drift que se formaliza en Sprint 4.
    """
    tabla = pd.DataFrame(mensual)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(tabla["month"], tabla["roc_auc"], "o-", color="#6D28D9", lw=2, label="ROC-AUC")
    ax.plot(tabla["month"], tabla["gini"], "s--", color="#DB2777", lw=2, label="Gini")
    ax.plot(tabla["month"], tabla["f1"], "^-", color="#9333EA", lw=2, label="F1")
    ax.set_title("Estabilidad del modelo mes a mes (backtest out-of-time)",
                 fontweight="bold")
    ax.set_xlabel("Mes"); ax.set_ylabel("Métrica"); ax.set_ylim(0, 1)
    ax.legend()
    return _guardar(fig, "s2_fig07_estabilidad_mensual.png")


# ════════════════════════════════════════════════════════════════════════
#  FIG 08 · Justificación del feature engineering (Gini univariante)
# ════════════════════════════════════════════════════════════════════════
COLORES_FAMILIA = {
    "A · Tiempos de entrega":      "#4C1D95",
    "B · Banderas de entrega":     "#6D28D9",
    "C · Valor del pedido":        "#9333EA",
    "D · Pago":                    "#C084FC",
    "E · Producto":                "#DB2777",
    "F · Señal de reseña":         "#F472B6",
    "G · Calendario y geografía":  "#A78BFA",
}


def fig_justificacion_features(tabla: "pd.DataFrame") -> str:
    """Gini univariante de cada candidata, coloreado por familia.

    Hace visible en una sola imagen qué hipótesis del Sprint 1 sobrevive
    a los datos: las familias de entrega dominan, calendario es ruido.
    La línea punteada marca el umbral del método univariante (0.05).
    """
    datos = (tabla.dropna(subset=["gini_univariante"])
             .sort_values("gini_univariante", ascending=True))
    colores = [COLORES_FAMILIA.get(f, "#D8CCEE") for f in datos["familia"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(datos["variable"], datos["gini_univariante"], color=colores)
    ax.axvline(C.UMBRAL_UNIVARIANTE, color="#DB2777", ls="--", lw=1.5,
               label=f"umbral univariante ({C.UMBRAL_UNIVARIANTE})")
    ax.set_title("Justificación del feature engineering · Gini univariante en TRAIN",
                 fontweight="bold")
    ax.set_xlabel("|Gini| univariante (2·AUC − 1)")
    # Leyenda manual por familia (los colores cuentan la historia)
    from matplotlib.patches import Patch
    parches = [Patch(color=c, label=f) for f, c in COLORES_FAMILIA.items()]
    parches.append(plt.Line2D([0], [0], color="#DB2777", ls="--",
                              label=f"umbral ({C.UMBRAL_UNIVARIANTE})"))
    ax.legend(handles=parches, fontsize=8, loc="lower right")
    return _guardar(fig, "s2_fig08_justificacion_features.png")
