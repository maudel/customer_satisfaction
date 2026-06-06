"""
============================================================================
 cleaning.py  ·  Paso 6 · Limpieza de variables
============================================================================

Implementa el paso "Limpieza de variables" del flujo de trabajo:
    - Clipado de outliers   (winsorización por cuantiles)
    - Tratamiento de NaN    (imputación documentada)
    - Agrupamiento de categorías raras  ("OTHER")

Diseñado con interfaz ajustar/transformar (estilo sklearn): los parámetros
(cortes de clipado, medianas de imputación, categorías frecuentes) se
APRENDEN solo del set de ENTRENAMIENTO y se APLICAN igual a validación /
backtest / producción / predicción. Esto evita fuga de información y
garantiza reproducibilidad mes a mes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F


class LimpiadorDatos:
    """Limpiador reproducible: se ajusta en train y se aplica a cualquier partición."""

    def __init__(
        self,
        cuantiles_clip: tuple[float, float] = C.CUANTILES_CLIP,
        pct_min_rara: float = C.PCT_MIN_CATEGORIA_RARA,
    ):
        self.cuantiles_clip = cuantiles_clip
        self.pct_min_rara = pct_min_rara
        # Parámetros aprendidos en .ajustar() (el sufijo "_" marca que son aprendidos)
        self.limites_clip_: dict[str, tuple[float, float]] = {}
        self.valores_imputacion_: dict[str, float] = {}
        self.categorias_frecuentes_: dict[str, set] = {}
        self.ajustado_ = False

    # ── AJUSTAR (aprende los parámetros solo de train) ────────────────────
    def ajustar(self, datos: pd.DataFrame) -> "LimpiadorDatos":
        # Clipado: aprendemos los cortes (cuantiles) de cada variable continua
        q_inf, q_sup = self.cuantiles_clip
        for col in F.VARIABLES_NUMERICAS:
            if col in datos.columns:
                serie = pd.to_numeric(datos[col], errors="coerce")
                self.limites_clip_[col] = (serie.quantile(q_inf), serie.quantile(q_sup))
                self.valores_imputacion_[col] = serie.median()

        # Categorías frecuentes (las raras se mandarán luego a "OTHER")
        n_filas = len(datos)
        for col in F.VARIABLES_CATEGORICAS:
            if col in datos.columns:
                frecuencia = datos[col].astype(str).value_counts(dropna=False) / n_filas
                self.categorias_frecuentes_[col] = set(
                    frecuencia[frecuencia >= self.pct_min_rara].index
                )
        self.ajustado_ = True
        return self

    # ── TRANSFORMAR (aplica lo aprendido, sin volver a aprender nada) ─────
    def transformar(self, datos: pd.DataFrame) -> pd.DataFrame:
        if not self.ajustado_:
            raise RuntimeError("LimpiadorDatos no ajustado: llama a .ajustar() primero.")
        salida = datos.copy()

        # 1) Numéricas: clipado de outliers + imputación con la mediana de train
        for col, (inf, sup) in self.limites_clip_.items():
            if col in salida.columns:
                salida[col] = pd.to_numeric(salida[col], errors="coerce")
                salida[col] = salida[col].clip(lower=inf, upper=sup)
                salida[col] = salida[col].fillna(self.valores_imputacion_[col])

        # 2) Binarias: un NaN equivale a "no ocurrió" → 0
        for col in F.VARIABLES_BINARIAS:
            if col in salida.columns:
                salida[col] = pd.to_numeric(salida[col], errors="coerce").fillna(0).astype(int)

        # 3) Categóricas: lo que no esté entre las frecuentes (o sea NaN) → "OTHER"
        for col, frecuentes in self.categorias_frecuentes_.items():
            if col in salida.columns:
                serie = salida[col].astype(str)
                salida[col] = np.where(serie.isin(frecuentes), serie, C.ETIQUETA_CATEGORIA_RARA)

        return salida

    def ajustar_transformar(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Atajo: ajusta y transforma en un solo paso (solo para train)."""
        return self.ajustar(datos).transformar(datos)

    # ── Reporte de limpieza (para documentar qué se hizo a cada variable) ─
    def reporte(self) -> pd.DataFrame:
        filas = []
        for col, (inf, sup) in self.limites_clip_.items():
            filas.append({
                "variable": col, "tipo": "numérica",
                "clip_low": round(float(inf), 3), "clip_high": round(float(sup), 3),
                "fill_value": round(float(self.valores_imputacion_[col]), 3),
                "n_categorias_frecuentes": "-",
            })
        for col, categorias in self.categorias_frecuentes_.items():
            filas.append({
                "variable": col, "tipo": "categórica",
                "clip_low": "-", "clip_high": "-", "fill_value": "-",
                "n_categorias_frecuentes": len(categorias),
            })
        return pd.DataFrame(filas)
