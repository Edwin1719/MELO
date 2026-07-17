"""
pipeline.py — Ejecuta el plan de preprocesamiento sobre el DataFrame.

Incluye ETL real:
- Parseo automático de formatos sucios ($1,234.56, 45%, fechas mezcladas)
- Normalización de strings categóricos (Male→male, Bogotá→bogota)
- Detección y tratamiento de outliers (IQR, capping)
- Manejo de valores infinitos, negativos fuera de rango, IDs
- Imputación, encoding, escalado, balanceo, split
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
warnings.filterwarnings("ignore")

def _is_text_col(df: pd.DataFrame, col: str) -> bool:
    """True si la columna es texto (object o StringDtype de pandas 3.x)."""
    dtype = df[col].dtype
    return dtype == "object" or isinstance(dtype, pd.StringDtype)


def _parse_dates_mixed(series: pd.Series) -> pd.Series:
    """
    Prueba múltiples formatos de fecha para columnas con formatos mezclados.

    Returns pd.Series de datetime (NaT para los no parseables).
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    str_vals = series.dropna().astype(str)

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%m/%d/%Y",
        "%Y%m%d",
    ]

    remaining = str_vals.copy()
    for fmt in formats:
        if len(remaining) == 0:
            break
        parsed = pd.to_datetime(remaining, format=fmt, errors="coerce")
        valid = parsed.notna()
        result.loc[remaining.index[valid]] = parsed[valid]
        remaining = remaining[~valid]

    # Fallback: inferir formato automáticamente
    if len(remaining) > 0:
        auto_parsed = pd.to_datetime(remaining, infer_datetime_format=True, errors="coerce")
        valid = auto_parsed.notna()
        result.loc[remaining.index[valid]] = auto_parsed[valid]

    return result


def preprocess_raw(df: pd.DataFrame, plan: dict, target: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocesamiento ETL ANTES de ejecutar el plan del estratega.

    Maneja:
    - Parseo de moneda: "$1,234.56" → 1234.56
    - Parseo de porcentajes: "45.2%" → 0.452
    - Fechas en múltiples formatos → features numéricas (año, mes, día, día_semana)
    - Valores infinitos → NaN (para imputación posterior)
    - Normalización de strings categóricos (strip, lowercase)
    - Detección de IDs de alta cardinalidad (>90% únicos)
    - Negativos fuera de rango
    """
    df = df.copy()
    etl_log = {"parsed_currency": [], "parsed_pct": [], "parsed_dates": [],
               "normalized_cats": [], "inf_handled": [], "ids_detected": [],
               "nulls_normalized": [], "bools_converted": [], "coerced_numeric": []}

    for col in df.columns:
        if col == target:
            continue

        # ── 0a. Normalizar nulos textuales ANTES de cualquier parseo ──
        if _is_text_col(df, col):
            null_aliases = {"n/a", "na", "null", "none", "?", "-", "--", "nan", ""}
            n_before = df[col].isna().sum()
            df[col] = df[col].replace(null_aliases, np.nan)
            if df[col].isna().sum() > n_before:
                etl_log.setdefault("nulls_normalized", []).append(col)

        # ── 0b. Convertir booleanos a 0/1 para sklearn ──────────────
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
            etl_log.setdefault("bools_converted", []).append(col)
            continue

        # ── 1. Detectar y parsear columnas de moneda ──────────────
        if _is_text_col(df, col):
            sample = df[col].dropna().iloc[:20]
            # Detectar si tiene formato moneda ($ o , como separador miles)
            has_currency = (
                sample.astype(str).str.contains(r"^\$", na=False).any()
                or (
                    sample.astype(str).str.contains(r"^\d+,\d{3}", na=False).any()
                    and not sample.astype(str).str.contains(r"^[A-Za-z]", na=False).any()
                )
            )
            if has_currency:
                try:
                    df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    etl_log["parsed_currency"].append(col)
                except Exception:
                    warnings.warn(f"Error al parsear moneda en columna '{col}'", stacklevel=2)
                continue

        # ── 2. Detectar y parsear porcentajes ─────────────────────
        if _is_text_col(df, col):
            pct_sample = df[col].dropna().iloc[:20]
            has_pct = pct_sample.astype(str).str.contains(r"%$", na=False).any()
            if has_pct and not has_currency:
                try:
                    df[col] = df[col].astype(str).str.replace("%", "", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
                    etl_log["parsed_pct"].append(col)
                except Exception:
                    warnings.warn(f"Error al parsear porcentaje en columna '{col}'", stacklevel=2)
                continue

        # ── 2b. Coerción genérica: columnas que parecen numéricas ──
        if _is_text_col(df, col):
            num_sample = df[col].dropna().iloc[:50]
            if len(num_sample) >= 5:
                parsed = pd.to_numeric(num_sample, errors="coerce")
                ratio = parsed.notna().sum() / len(num_sample)
                if ratio >= 0.7:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    etl_log.setdefault("coerced_numeric", []).append(col)
                    continue

        # ── 3. Detectar y parsear fechas ──────────────────────────
        if _is_text_col(df, col):
            sample = df[col].dropna()
            if len(sample) > 10:
                sample_strs = sample.iloc[:30].astype(str)
                date_patterns = [
                    r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}",
                    r"\d{4}/\d{2}/\d{2}", r"\d{2}-[A-Za-z]{3}-\d{4}",
                ]
                has_date = any(
                    sample_strs.str.contains(p, na=False).any()
                    for p in date_patterns
                )
                if has_date:
                    parsed = _parse_dates_mixed(df[col])
                    if parsed.notna().sum() > len(df) * 0.4:
                        year = parsed.dt.year.fillna(parsed.dt.year.median())
                        month = parsed.dt.month.fillna(parsed.dt.month.median())
                        day = parsed.dt.day.fillna(parsed.dt.day.median())
                        dow = parsed.dt.dayofweek.fillna(2)
                        df[f"{col}_año"] = year
                        df[f"{col}_mes"] = month
                        df[f"{col}_dia"] = day
                        df[f"{col}_diasemana"] = dow
                        df = df.drop(columns=[col])
                        etl_log["parsed_dates"].append(col)
                        continue  # Columna procesada como fecha
                    else:
                        # No se pudo parsear → tratar como ID/no-categórica
                        etl_log["ids_detected"].append(col)
                        continue
                # Si no parece fecha, continúa a pasos 4 y 5
            # Si hay pocas muestras, continúa a pasos 4 y 5

        # ── 4. Normalizar strings categóricos ────────────────────
        if _is_text_col(df, col):
            n_unique = df[col].nunique()
            if n_unique < len(df) * 0.5:
                # Unificar variantes comunes
                replacements = {
                    "masculino": "masculino", "male": "masculino", "m": "masculino", "h": "masculino",
                    "femenino": "femenino", "female": "femenino", "f": "femenino", "mujer": "femenino",
                    "no binario": "otro", "nb": "otro", "prefiero no decirlo": "otro",
                }
                df[col] = df[col].map(lambda x: replacements.get(x, x))
                etl_log["normalized_cats"].append(col)

        # ── 5. Detectar IDs de alta cardinalidad (>90% únicos) ───
        if _is_text_col(df, col):
            n_unique = df[col].nunique()
            if n_unique > len(df) * 0.9:
                etl_log["ids_detected"].append(col)

        # ── 6. Manejar valores infinitos → NaN ────────────────────
        if pd.api.types.is_numeric_dtype(df[col]):
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                etl_log["inf_handled"].append(col)

        # ── 7. Manejar negativos en columnas que no deberían tenerlos ───
        if pd.api.types.is_numeric_dtype(df[col]) and col not in (target or ""):
            min_val = df[col].min()
            if min_val is not None and min_val < 0 and not pd.isna(min_val):
                # Si tiene pocos negativos (<5%) → convertirlos a NaN
                neg_count = (df[col] < 0).sum()
                if neg_count > 0 and neg_count < len(df) * 0.05:
                    df.loc[df[col] < 0, col] = np.nan

    return df, etl_log


def detect_and_cap_outliers(
    df: pd.DataFrame, plan: dict, target: Optional[str] = None
) -> pd.DataFrame:
    """
    Detecta y trata outliers usando IQR.
    Solo aplica a columnas numéricas.
    """
    df = df.copy()
    treated = []

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0 or pd.isna(IQR):
            continue

        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_ratio = n_outliers / len(df)

        # Solo tratar si hay outliers significativos (>1% y <10%)
        if 0.01 < outlier_ratio < 0.10:
            df[col] = df[col].clip(lower, upper)
            treated.append(col)

    return df, treated


def execute_pipeline(
    df: pd.DataFrame,
    plan: dict,
    target: Optional[str] = None,
    config=None,
) -> dict:
    """
    Ejecuta el plan de preprocesamiento completo con ETL real.
    """
    df = df.copy()
    preprocessor = {
        "dropped_cols": [],
        "imputed_cols": {},
        "encoded_cols": {},
        "scaled_cols": {},
        "outliers_capped": [],
        "etl": {},
        "target_encoders": {},
    }

    # ── 00. Eliminar filas duplicadas (antes de ETL para evitar data leakage) ──
    dup_before = len(df)
    df = df.drop_duplicates()
    if len(df) < dup_before:
        preprocessor["duplicates_removed"] = dup_before - len(df)

    # ── 0. ETL pre-process ─────────────────────────────────────────
    df, etl_log = preprocess_raw(df, plan, target)
    preprocessor["etl"] = etl_log

    # ── 0b. Outlier capping (antes de cualquier otra cosa) ─────────
    df, capped_cols = detect_and_cap_outliers(df, plan, target)
    preprocessor["outliers_capped"] = capped_cols

    # ── 0c. Eliminar IDs de alta cardinalidad ──────────────────────
    for col in etl_log.get("ids_detected", []):
        if col in df.columns and col != target:
            df = df.drop(columns=[col])
            preprocessor["dropped_cols"].append(f"{col} (ID)")
    # Si el plan ya incluye drop para estas, evitar duplicados
    plan_drops = {a["col"] for a in plan.get("preprocessing", [])
                  if a["action"] == "drop"}

    # ── 1. Separar target ──────────────────────────────────────────
    y = None
    if target and target in df.columns:
        y = df[target].copy()
        df = df.drop(columns=[target])

    # ── 2. Aplicar acciones del plan ───────────────────────────────
    # Filtrar acciones que ya no aplican (columnas ya eliminadas o parseadas)
    dropped_in_etl = set(etl_log.get("ids_detected", []))
    parsed_dates = set(etl_log.get("parsed_dates", []))
    skipped_actions = dropped_in_etl | parsed_dates

    for action in plan.get("preprocessing", []):
        col = action["col"]
        action_type = action["action"]

        if col in skipped_actions:
            continue
        # Transform actions crean nuevas columnas; verificar source en vez de col
        if action_type == "transform":
            source_cols = action.get("source", [col])
            if not all(c in df.columns for c in source_cols):
                continue
        elif col not in df.columns:
            continue

        if action_type == "drop":
            df = df.drop(columns=[col])
            if col not in preprocessor["dropped_cols"]:
                preprocessor["dropped_cols"].append(col)
            continue

        if action_type == "impute":
            # Indicador de que el valor faltaba (antes de imputar)
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                df[f"{col}_is_missing"] = df[col].isna().astype(int)
            method = action.get("method", "mean")
            if method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "zero":
                df[col] = df[col].fillna(0)
            elif method == "mode":
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
            preprocessor["imputed_cols"][col] = method
            if n_nulls > 0:
                preprocessor.setdefault("missing_indicators", []).append(col)

        elif action_type == "encode":
            method = action.get("method", "onehot")
            if method == "label":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                preprocessor["encoded_cols"][col] = {
                    "method": "label", "classes": le.classes_.tolist(),
                }
            elif method == "onehot":
                ohe = OneHotEncoder(sparse_output=False, drop="first",
                                    handle_unknown="ignore")
                encoded = ohe.fit_transform(df[[col]])
                col_names = [
                    f"{col}_{cat}"
                    for cat in ohe.categories_[0][1:]
                ]
                encoded_df = pd.DataFrame(
                    encoded, columns=col_names, index=df.index
                )
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                preprocessor["encoded_cols"][col] = {
                    "method": "onehot",
                    "categories": ohe.categories_[0].tolist(),
                    "new_columns": col_names,
                }
            elif method == "target":
                target_mean = y.groupby(df[col]).mean() if y is not None else None
                if target_mean is not None:
                    df[col] = df[col].map(target_mean).fillna(y.mean())
                    preprocessor["encoded_cols"][col] = {
                        "method": "target",
                        "mapping": target_mean.to_dict(),
                    }
                else:
                    freq = df[col].value_counts()
                    df[col] = df[col].map(freq).fillna(0)
                    preprocessor["encoded_cols"][col] = {"method": "frequency"}

        elif action_type == "scale":
            method = action.get("method", "standard")
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                continue

            df[col] = scaler.fit_transform(df[[col]]).ravel()
            preprocessor["scaled_cols"][col] = {
                "method": method,
                "params": str(scaler.get_params()),
            }

        elif action_type == "transform":
            method = action.get("method", "log1p")
            source = action.get("source", [col])
            params = action.get("params", {})
            new_col = col
            transformed = {}

            if method == "log1p":
                log_col = f"{col}_log"
                df[log_col] = np.log1p(df[col].clip(lower=0))
                transformed = {"method": "log1p", "new_column": log_col}

            elif method == "ratio" and len(source) >= 1:
                group_col = params.get("group_col")
                src = source[0]
                if group_col and group_col in df.columns:
                    df[new_col] = df[src] / df.groupby(group_col)[src].transform("mean").replace(0, np.nan)
                    df[new_col] = df[new_col].fillna(1)
                else:
                    df[new_col] = df[src]
                transformed = {"method": "ratio", "group_col": group_col}

            elif method == "interact" and len(source) >= 2:
                df[new_col] = df[source[0]] * df[source[1]]
                transformed = {"method": "interact", "sources": source[:2]}

            elif method == "str_len" and len(source) >= 1:
                df[new_col] = df[source[0]].astype(str).str.len()
                transformed = {"method": "str_len", "source": source[0]}

            elif method == "keyword" and len(source) >= 1:
                pattern = params.get("pattern", "")
                if pattern:
                    df[new_col] = df[source[0]].astype(str).str.contains(pattern, case=False, na=False).astype(int)
                    transformed = {"method": "keyword", "pattern": pattern}

            elif method == "date_diff" and len(source) >= 2:
                diff = (pd.to_datetime(df[source[0]]) - pd.to_datetime(df[source[1]]))
                df[new_col] = diff.dt.days.fillna(0).astype(int)
                transformed = {"method": "date_diff", "sources": source[:2]}

            elif method == "bin" and len(source) >= 1:
                n_bins = params.get("n_bins", 4)
                labels = params.get("labels", None)
                df[new_col] = pd.cut(df[source[0]], bins=n_bins, labels=labels)
                transformed = {"method": "bin", "n_bins": n_bins}

            if transformed:
                preprocessor.setdefault("transformed_cols", {})[col] = transformed

    # ── 3. Feature selection (varianza casi cero) ─────────────────
    if plan.get("feature_selection", False) and len(df.columns) > 2:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 2:
            variances = numeric_df.var()
            near_zero = variances[variances < 1e-10].index.tolist()
            if near_zero:
                df = df.drop(columns=near_zero)
                preprocessor["zero_variance_dropped"] = near_zero

    # ── 3b. Eliminar columnas constantes (incluyendo object) ──────
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        df = df.drop(columns=constant_cols)
        preprocessor["constant_dropped"] = constant_cols

    # ── 4. Train/Test split ───────────────────────────────────────
    test_size = getattr(config, "test_size", 0.2) if config else 0.2
    random_state = getattr(config, "random_state", 42) if config else 42

    target_encoder = None
    task_type = plan.get("task_type", "regression")
    y_encoded = None

    if y is not None:
        if task_type == "classification" or isinstance(y.dtype, pd.StringDtype) or y.dtype == "object":
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
        else:
            y_encoded = y

    balance_strategy = plan.get("balance", {}).get("strategy", "none")
    stratify_param = None
    if (
        y_encoded is not None
        and plan.get("task_type") == "classification"
        and balance_strategy not in ("none",)
    ):
        # Solo stratify si todas las clases tienen >= 2 miembros
        _class_counts = pd.Series(y_encoded).value_counts()
        if _class_counts.min() >= 2:
            stratify_param = y_encoded

    if y_encoded is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            df, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )
    else:
        X_train, X_test = df, None
        y_train = y_test = None

    # ── 5. Balanceo SOLO en train ─────────────────────────────────
    if balance_strategy not in ("none", "class_weight") and y_train is not None:
        try:
            X_train, y_train = _apply_balancing(
                X_train, y_train, balance_strategy, config
            )
            preprocessor["balance_applied"] = {
                "strategy": balance_strategy,
                "original_size": len(df),
                "balanced_size": len(X_train),
            }
        except ImportError:
            preprocessor["balance_applied"] = {
                "strategy": "none (imbalanced-learn no instalado)", "error": True,
            }
        except Exception as e:
            preprocessor["balance_applied"] = {
                "strategy": f"none (error: {e})", "error": True,
            }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X_train.columns.tolist() if X_train is not None else [],
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "target_classes": (
            target_encoder.classes_.tolist() if target_encoder else None
        ),
    }


def _apply_balancing(
    X: pd.DataFrame, y: np.ndarray, strategy: str, config,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Aplica técnicas de balanceo vía imbalanced-learn."""
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN, SMOTETomek

    sampler_map = {
        "smote": SMOTE(random_state=42, k_neighbors=getattr(config, "smote_k_neighbors", 5)),
        "adasyn": ADASYN(random_state=42),
        "smote_enn": SMOTEENN(random_state=42),
        "smote_tomek": SMOTETomek(random_state=42),
    }
    sampler = sampler_map.get(strategy)
    if sampler is None:
        return X, y
    return sampler.fit_resample(X, y)
