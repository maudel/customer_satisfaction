"""
============================================================================
 selection.py  ·  Paso 7 · Selección de Variables
============================================================================

Reproduce la cascada de selección del "flujo de trabajo de Modelos de
Negocio", en el orden de ejecución del Excel:

    0. Estado Inicial
    1. metodo_faltantes      (descarta features con muchos nulos)
    2. metodo_psi            (descarta features inestables en el tiempo)
    3. metodo_correlacion    (descarta features redundantes entre sí)
    4. metodo_univariante    (descarta features sin señal vs target)
    5. WOE / Information Value (ranking complementario de poder predictivo)

Cada paso se EVALÚA entrenando un Random Forest y midiendo AUC-ROC en
train y val, produciendo una tabla idéntica a la del flujo de trabajo:

    orden | método | features | AUC-ROC train | AUC-ROC val

Los parámetros (umbrales) viven en config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from . import features as F


# ════════════════════════════════════════════════════════════════════════
#  MÉTRICAS AUXILIARES (una por criterio de la cascada)
# ════════════════════════════════════════════════════════════════════════
def indice_estabilidad_poblacional(
    esperado: pd.Series, observado: pd.Series, num_bins: int = C.NUM_BINS_PSI
) -> float:
    """PSI entre la distribución de TRAIN (esperado) y otra partición (observado).

    Interpretación habitual:
        PSI < 0.10  : estable
        0.10–0.25   : cambio moderado
        > 0.25      : inestable (la feature 'se mueve' con el tiempo) → descartar
    """
    esperado = pd.to_numeric(esperado, errors="coerce").dropna()
    observado = pd.to_numeric(observado, errors="coerce").dropna()
    if esperado.nunique() <= 1 or len(esperado) == 0 or len(observado) == 0:
        return 0.0

    # Bins por cuantiles de la población esperada (la de referencia)
    cuantiles = np.linspace(0, 1, num_bins + 1)
    bordes = np.unique(esperado.quantile(cuantiles).values)
    if len(bordes) < 3:
        return 0.0
    bordes[0], bordes[-1] = -np.inf, np.inf

    pct_esperado = np.histogram(esperado, bins=bordes)[0] / len(esperado)
    pct_observado = np.histogram(observado, bins=bordes)[0] / len(observado)
    eps = 1e-6  # evita log(0) / división por cero en bins vacíos
    pct_esperado = np.clip(pct_esperado, eps, None)
    pct_observado = np.clip(pct_observado, eps, None)
    return float(np.sum((pct_observado - pct_esperado) * np.log(pct_observado / pct_esperado)))


def fuerza_univariante(x: pd.Series, y: pd.Series) -> float:
    """Poder univariante de una feature vs target binario.

    Usa el |Gini| univariante (2·AUC − 1) tratando la feature como score.
    Escala natural para los umbrales del flujo de trabajo [0.1, 0.2, 0.3]:
        Gini 0.1 ≈ AUC 0.55  ·  Gini 0.3 ≈ AUC 0.65
    Toma valor absoluto: da igual si la feature correlaciona en + o en −.
    """
    x = pd.to_numeric(x, errors="coerce")
    valido = x.notna() & y.notna()
    # Hacen falta datos suficientes y variación tanto en x como en y
    if valido.sum() < 30 or x[valido].nunique() <= 1 or y[valido].nunique() < 2:
        return 0.0
    try:
        auc = roc_auc_score(y[valido].astype(int), x[valido])
    except ValueError:
        return 0.0
    return float(abs(2 * auc - 1))


def valor_informacion(x: pd.Series, y: pd.Series, num_bins: int = 10) -> float:
    """Information Value (WOE) de una feature vs target binario.

    Para variables con baja cardinalidad (binarias/categóricas) agrupa por
    valor; para continuas usa bins por cuantiles.
    """
    x = pd.to_numeric(x, errors="coerce")
    tabla = pd.DataFrame({"x": x, "y": y}).dropna()
    if tabla["x"].nunique() <= 1 or tabla["y"].nunique() < 2:
        return 0.0
    if tabla["x"].nunique() <= num_bins:
        tabla["bucket"] = tabla["x"]
    else:
        try:
            tabla["bucket"] = pd.qcut(tabla["x"], q=num_bins, duplicates="drop")
        except ValueError:
            return 0.0
    grupos = tabla.groupby("bucket", observed=True)["y"].agg(["sum", "count"])
    buenos = grupos["sum"]                    # eventos (y=1)
    malos = grupos["count"] - grupos["sum"]   # no-eventos (y=0)
    pct_buenos = buenos / max(buenos.sum(), 1)
    pct_malos = malos / max(malos.sum(), 1)
    eps = 1e-6
    woe = np.log((pct_buenos + eps) / (pct_malos + eps))
    iv = float(((pct_buenos - pct_malos) * woe).sum())
    return iv if np.isfinite(iv) else 0.0


# ════════════════════════════════════════════════════════════════════════
#  EVALUADOR  (Random Forest · AUC train/val)  — como en el flujo de trabajo
# ════════════════════════════════════════════════════════════════════════
def _matriz_diseno(datos: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Matriz numérica para el RF evaluador: numéricas tal cual + one-hot ligero."""
    numericas = [c for c in variables if c in F.VARIABLES_NUMERICAS + F.VARIABLES_BINARIAS]
    categoricas = [c for c in variables if c in F.VARIABLES_CATEGORICAS]
    X = datos[numericas].apply(pd.to_numeric, errors="coerce").fillna(0).copy()
    if categoricas:
        dummies = pd.get_dummies(datos[categoricas].astype(str), dummy_na=False)
        X = pd.concat([X, dummies], axis=1)
    return X


def evaluar_variables(
    train: pd.DataFrame, val: pd.DataFrame, variables: list[str], objetivo: str = C.OBJETIVO
) -> tuple[float, float]:
    """Entrena un RF con `variables` y devuelve (AUC train, AUC val)."""
    if not variables:
        return 0.5, 0.5
    X_train = _matriz_diseno(train, variables)
    X_val = _matriz_diseno(val, variables).reindex(columns=X_train.columns, fill_value=0)
    y_train, y_val = train[objetivo].astype(int), val[objetivo].astype(int)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return 0.5, 0.5

    rf = RandomForestClassifier(**C.PARAMS_RF_SELECCION)
    rf.fit(X_train, y_train)
    auc_train = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    auc_val = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    return float(auc_train), float(auc_val)


# ════════════════════════════════════════════════════════════════════════
#  CASCADA DE SELECCIÓN
# ════════════════════════════════════════════════════════════════════════
class SelectorVariables:
    """Ejecuta la cascada completa y guarda el reporte paso a paso."""

    def __init__(
        self,
        umbral_faltantes: float = C.UMBRAL_FALTANTES,
        umbral_psi: float = C.UMBRAL_PSI,
        umbral_correlacion: float = C.UMBRAL_CORRELACION,
        umbral_univariante: float = C.UMBRAL_UNIVARIANTE,
        num_bins_psi: int = C.NUM_BINS_PSI,
    ):
        self.umbral_faltantes = umbral_faltantes
        self.umbral_psi = umbral_psi
        self.umbral_correlacion = umbral_correlacion
        self.umbral_univariante = umbral_univariante
        self.num_bins_psi = num_bins_psi
        # Resultados que se rellenan al ajustar
        self.reporte_: list[dict] = []
        self.variables_seleccionadas_: list[str] = []
        self.detalle_: dict[str, dict] = {}

    def _registrar(self, orden, metodo, variables, train, val):
        """Anota una fila del reporte RE-EVALUANDO el RF con las variables vivas.

        Método pedido explícitamente por el docente en la defensa: "primero
        calculás con toda la población; eliminás algo, volvés a calcular".
        Cada fila re-entrena el evaluador y re-mide AUC y Gini (= 2·AUC − 1,
        la métrica del caso) en train y val.
        """
        auc_train, auc_val = evaluar_variables(train, val, variables)
        self.reporte_.append({
            "orden": orden,
            "metodo": metodo,
            "features": len(variables),
            "auc_roc_train": round(auc_train, 4),
            "auc_roc_val": round(auc_val, 4),
            "gini_train": round(2 * auc_train - 1, 4),
            "gini_val": round(2 * auc_val - 1, 4),
        })

    def ajustar(self, train: pd.DataFrame, val: pd.DataFrame) -> "SelectorVariables":
        objetivo = C.OBJETIVO
        y = train[objetivo].astype(int)
        candidatas = [c for c in F.todas_las_columnas() if c in train.columns]

        # ── 0 · Estado inicial (todas las features sobre la mesa) ─────────
        self._registrar(0, "Estado Inicial", candidatas, train, val)

        # ── 1 · método de faltantes (descarta features muy incompletas) ──
        faltantes = {c: train[c].isna().mean() for c in candidatas}
        candidatas = [c for c in candidatas if faltantes[c] <= self.umbral_faltantes]
        self.detalle_["missing"] = {k: round(v, 4) for k, v in faltantes.items()}
        self._registrar(1, "metodo_faltantes", candidatas, train, val)

        # ── 2 · método PSI (estabilidad temporal train vs val) ───────────
        psi = {}
        numericas = [c for c in candidatas if c in F.VARIABLES_NUMERICAS]
        for c in numericas:
            psi[c] = indice_estabilidad_poblacional(train[c], val[c], self.num_bins_psi)
        descartar_psi = {c for c, v in psi.items() if v > self.umbral_psi}
        candidatas = [c for c in candidatas if c not in descartar_psi]
        self.detalle_["psi"] = {k: round(v, 4) for k, v in psi.items()}
        self._registrar(2, "metodo_psi", candidatas, train, val)

        # ── 3 · método de correlación (descarta redundancia entre features) ─
        numericas = [c for c in candidatas if c in F.VARIABLES_NUMERICAS]
        if len(numericas) > 1:
            corr = train[numericas].apply(pd.to_numeric, errors="coerce").corr().abs()
            triangulo_sup = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            # Para cada par muy correlado, descarta el de menor señal univariante
            a_descartar = set()
            for col in triangulo_sup.columns:
                for fila in triangulo_sup.index:
                    if triangulo_sup.loc[fila, col] > self.umbral_correlacion:
                        fuerza_fila = fuerza_univariante(train[fila], y)
                        fuerza_col = fuerza_univariante(train[col], y)
                        a_descartar.add(col if fuerza_col <= fuerza_fila else fila)
            candidatas = [c for c in candidatas if c not in a_descartar]
            self.detalle_["correlation_dropped"] = sorted(a_descartar)
        self._registrar(3, f"metodo_correlacion: {self.umbral_correlacion}",
                        candidatas, train, val)

        # ── 4 · método univariante (descarta features sin señal vs target) ─
        univariante = {}
        for c in candidatas:
            if c in F.VARIABLES_NUMERICAS + F.VARIABLES_BINARIAS:
                univariante[c] = fuerza_univariante(train[c], y)
            else:
                univariante[c] = 1.0  # categóricas: se conservan, se evalúan por IV aparte
        debiles = {c for c, v in univariante.items() if v < self.umbral_univariante}
        candidatas = [c for c in candidatas if c not in debiles]
        self.detalle_["univariate_strength"] = {k: round(v, 4) for k, v in univariante.items()}
        self._registrar(4, f"metodo_univariante: {self.umbral_univariante}",
                        candidatas, train, val)

        # ── 5 · WOE / Information Value (ranking complementario) ─────────
        iv = {}
        for c in candidatas:
            if c in F.VARIABLES_NUMERICAS + F.VARIABLES_BINARIAS:
                iv[c] = valor_informacion(train[c], y, num_bins=10)
        self.detalle_["information_value"] = {
            k: round(v, 4) for k, v in sorted(iv.items(), key=lambda par: -par[1])
        }

        self.variables_seleccionadas_ = candidatas
        return self

    def tabla_reporte(self) -> pd.DataFrame:
        """Tabla orden × método × AUC/Gini (la del flujo de trabajo)."""
        return pd.DataFrame(self.reporte_)

    def mostrar_reporte(self) -> pd.DataFrame:
        """Imprime el reporte como DataFrame legible y lo devuelve.

        Formato de display pedido por el docente en la defensa ("prefiero
        un DataFrame, para que tengamos más visión"): una sola tabla con
        el efecto de cada eliminación, en lugar de prints sueltos.
        """
        tabla = self.tabla_reporte()
        print("\nCascada de selección (re-cálculo tras cada eliminación):")
        print(tabla.to_string(index=False))
        return tabla

    def tabla_iv(self) -> pd.DataFrame:
        """Ranking de variables por Information Value, con su etiqueta de fuerza."""
        iv = self.detalle_.get("information_value", {})
        filas = [{"variable": k, "information_value": v, "fuerza": _etiqueta_iv(v)}
                 for k, v in iv.items()]
        return pd.DataFrame(filas)


def _etiqueta_iv(iv: float) -> str:
    """Traduce un valor de IV a su interpretación cualitativa."""
    if iv < 0.02:  return "sin poder"
    if iv < 0.1:   return "débil"
    if iv < 0.3:   return "medio"
    if iv < 0.5:   return "fuerte"
    return "sospechoso (revisar fuga)"
