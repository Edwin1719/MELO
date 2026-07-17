"""
profiler.py — Perfilado automático de datasets.

Analiza un DataFrame y devuelve un perfil estructurado que el
NPC estratega usa para decidir el plan de preprocesamiento.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .config import BALANCE_THRESHOLDS, BALANCE_STRATEGIES


def profile_dataset(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    """
    Perfila un DataFrame completo: columnas, nulos, tipos, distribuciones,
    correlaciones y (si hay target) análisis de balanceo.

    Returns
    -------
    dict con toda la metadata estructurada.
    """
    columns = []
    numeric_cols = []
    categorical_cols = []
    high_cardinality_cols = []
    high_null_cols = []

    for col in df.columns:
        col_info = _profile_column(df, col)
        columns.append(col_info)

        if col == target:
            continue

        # Incluir categorical_numeric como numérica para imputación/escalado
        if col_info["type"] in ("numeric", "categorical_numeric"):
            numeric_cols.append(col)
        if col_info["cardinality_ratio"] > 0.5:
            high_cardinality_cols.append(col)
        if col_info["type"] == "categorical":
            categorical_cols.append(col)

        if col_info["null_pct"] > 50:
            high_null_cols.append(col)

    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": columns,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "high_null_cols": high_null_cols,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "duplicated_rows": int(df.duplicated().sum()),
    }

    # Correlaciones si hay al menos 2 numéricas
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        high_corr = _find_high_correlations(corr_df)
        profile["correlations"] = {
            "matrix": corr_df.to_dict(),
            "high_pairs": high_corr,
        }
    else:
        profile["correlations"] = {"matrix": {}, "high_pairs": []}

    # Análisis del target si se especificó
    if target and target in df.columns:
        target_profile = _profile_target(df, target)
        profile["target"] = target_profile
        profile["task_type_candidates"] = target_profile["task_candidates"]
    else:
        profile["target"] = None
        profile["task_type_candidates"] = ["clustering", "dimensionality_reduction"]

    return profile


def _profile_column(df: pd.DataFrame, col: str) -> dict:
    """Perfila una columna individual."""
    series = df[col]
    n_total = len(series)
    n_null = int(series.isna().sum())
    n_unique = int(series.nunique())
    dtype = str(series.dtype)

    info = {
        "name": col,
        "dtype": dtype,
        "nulls": n_null,
        "null_pct": round(n_null / n_total * 100, 2) if n_total else 0,
        "unique": n_unique,
        "cardinality_ratio": round(n_unique / n_total, 4) if n_total else 0,
    }

    # Detectar tipo semántico
    if pd.api.types.is_numeric_dtype(series):
        info["type"] = "numeric"
        info["min"] = float(series.min()) if n_total > 0 else None
        info["max"] = float(series.max()) if n_total > 0 else None
        info["mean"] = float(series.mean()) if n_total > 0 else None
        info["std"] = float(series.std()) if n_total > 0 else None
        info["skew"] = float(series.skew()) if n_total > 0 else None
        # Detectar si es categórica numérica (pocos valores únicos)
        if n_unique <= 10:
            info["type"] = "categorical_numeric"
            info["values"] = series.value_counts().head(20).to_dict()
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        info["type"] = "categorical"
        if n_unique <= 20:
            info["values"] = series.value_counts().head(20).to_dict()
    elif pd.api.types.is_bool_dtype(series):
        info["type"] = "boolean"
        info["values"] = series.value_counts().to_dict()
    elif pd.api.types.is_datetime64_any_dtype(series):
        info["type"] = "datetime"
        info["min"] = str(series.min()) if n_total > 0 else None
        info["max"] = str(series.max()) if n_total > 0 else None
    else:
        info["type"] = "other"

    return info


def _profile_target(df: pd.DataFrame, target: str) -> dict:
    """Analiza la variable objetivo: tipo de tarea, desbalanceo y concentración."""
    series = df[target]
    n_unique = series.nunique()
    n_total = len(series)

    vc = series.value_counts()
    majority_count = int(vc.iloc[0])
    majority_value = vc.index[0]
    majority_pct = round(majority_count / n_total * 100, 2)

    result = {
        "name": target,
        "dtype": str(series.dtype),
        "unique": n_unique,
        "majority_value": majority_value,
        "majority_pct": majority_pct,
        "top_values": {str(k): int(v) for k, v in vc.head(5).items()},
    }

    # Nivel de concentración (aplica a cualquier target)
    if majority_pct > 80:
        result["concentration_level"] = "extreme"
    elif majority_pct > 60:
        result["concentration_level"] = "severe"
    elif majority_pct > 40:
        result["concentration_level"] = "moderate"
    else:
        result["concentration_level"] = "normal"

    # ── Tipo de tarea ─────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series) and n_unique > 15:
        result["task_type"] = "regression"
        result["task_candidates"] = ["regression"]
        result["min"] = float(series.min())
        result["max"] = float(series.max())
        result["mean"] = float(series.mean())
        result["std"] = float(series.std())
        result["distribution"] = {
            "p25": float(series.quantile(0.25)),
            "p50": float(series.median()),
            "p75": float(series.quantile(0.75)),
        }
    else:
        if n_unique == 2:
            result["task_type"] = "classification"
            result["task_candidates"] = ["classification"]
        elif n_unique < 15:
            result["task_type"] = "classification"
            result["task_candidates"] = ["classification"]
        else:
            result["task_type"] = "classification"
            result["task_candidates"] = ["classification"]
            result["high_cardinality_warning"] = True

        counts = series.value_counts()
        result["distribution"] = counts.to_dict()
        result["distribution_pct"] = (counts / n_total * 100).round(2).to_dict()

        majority = counts.max()
        minority = counts.min()
        ratio = minority / majority if majority > 0 else 1.0
        result["balance_ratio"] = round(ratio, 4)

        levels = sorted(BALANCE_THRESHOLDS.items(), key=lambda x: -x[1])
        level = "extreme"
        for name, threshold in levels:
            if ratio >= threshold:
                level = name
                break
        result["balance_level"] = level
        result["balance_strategy"] = BALANCE_STRATEGIES[level]
        result["use_weighted_metrics"] = level != "balanced"

    return result


def _find_high_correlations(corr_df: pd.DataFrame, threshold: float = 0.8) -> list:
    """Encuentra pares de columnas con alta correlación."""
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = abs(corr_df.iloc[i, j])
            if val > threshold:
                pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": round(val, 3),
                })
    return sorted(pairs, key=lambda x: -x["correlation"])


def profile_summary_text(profile: dict) -> str:
    """Genera un resumen legible del perfil para el NPC."""
    lines = []
    lines.append(f"Dataset: {profile['n_rows']} filas x {profile['n_columns']} columnas")
    lines.append(f"Memoria: {profile['memory_mb']:.1f} MB")
    lines.append(f"Filas duplicadas: {profile['duplicated_rows']}")

    if profile["target"]:
        t = profile["target"]
        lines.append(f"")
        lines.append(f"Target: {t['name']} ({t['task_type']})")
        lines.append(f"  Tipo: {t['dtype']}")
        if t["task_type"] == "classification":
            lines.append(f"  Clases: {t['unique']}")
            if t["balance_level"] != "balanced":
                lines.append(
                    f"  Balanceo: {t['balance_level']} "
                    f"(ratio {t['balance_ratio']})"
                )
                lines.append(f"  Estrategia sugerida: {t['balance_strategy']}")
            lines.append(f"  Distribución: {t['distribution']}")
        elif t["task_type"] == "regression":
            lines.append(f"  Rango: [{t['min']:.2f} - {t['max']:.2f}]")
            lines.append(f"  Media ± std: {t['mean']:.2f} ± {t['std']:.2f}")

    lines.append(f"")
    lines.append(f"Columnas numéricas ({len(profile['numeric_cols'])}):")
    for c in profile["numeric_cols"]:
        ci = next(col for col in profile["columns"] if col["name"] == c)
        lines.append(f"  • {c}: nulls={ci['null_pct']}%, "
                      f"rango=[{ci.get('min', '?'):.1f}, {ci.get('max', '?'):.1f}]")

    lines.append(f"")
    lines.append(f"Columnas categóricas ({len(profile['categorical_cols'])}):")
    for c in profile["categorical_cols"]:
        ci = next(col for col in profile["columns"] if col["name"] == c)
        lines.append(f"  • {c}: {ci['unique']} valores únicos")

    if profile["high_cardinality_cols"]:
        lines.append(f"")
        lines.append(f"⚠️ Alta cardinalidad ({len(profile['high_cardinality_cols'])}):")
        for c in profile["high_cardinality_cols"]:
            lines.append(f"  • {c}")

    if profile["high_null_cols"]:
        lines.append(f"")
        lines.append(f"⚠️ Altos nulos (>50%): {profile['high_null_cols']}")

    if profile["correlations"]["high_pairs"]:
        lines.append(f"")
        lines.append(f"Altas correlaciones ({len(profile['correlations']['high_pairs'])}):")
        for p in profile["correlations"]["high_pairs"][:5]:
            lines.append(
                f"  • {p['col1']} ↔ {p['col2']}: {p['correlation']}"
            )

    return "\n".join(lines)
