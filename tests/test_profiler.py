"""
Tests para src/profiler.py

Cubre:
1. profile_dataset() detecta tipos correctamente
2. _profile_target() con target continuo (precio) → regression
3. _profile_target() con target concentrado (calificacion) → extreme
4. _profile_target() con target binario (churn) → severe
5. Alta cardinalidad en columnas tipo id
6. Columnas numéricas: min/max/mean/skew
"""

import os
import math
import pandas as pd
import numpy as np
import pytest

from src.profiler import profile_dataset, _profile_target, _profile_column

# ── Paths a datasets ──────────────────────────────────────────────
DATASETS = os.path.join(os.path.dirname(__file__), "..", "datasets")
CHURN_PATH = os.path.join(DATASETS, "churn_real.csv")
PRECIOS_PATH = os.path.join(DATASETS, "precios_viviendas.csv")
PRODUCTOS_PATH = os.path.join(DATASETS, "Productos_ML.csv")
SEGMENTACION_REAL_PATH = os.path.join(DATASETS, "segmentacion_real.csv")


# ====================================================================
# profile_dataset — detección de tipos de columna
# ====================================================================

class TestProfileDataset:
    """profile_dataset() — estructura general y tipos de columna."""

    def test_profile_dataset_detects_column_types(self):
        """Numeric, categorical y alta cardinalidad se clasifican correctamente."""
        df = pd.read_csv(CHURN_PATH)
        profile = profile_dataset(df, target="churn")

        assert profile["n_rows"] == len(df)
        assert profile["n_columns"] == len(df.columns)

        # Columnas numéricas esperadas
        assert "ingreso_mensual" in profile["numeric_cols"]
        assert "gasto_total_anual" in profile["numeric_cols"]
        assert "edad" in profile["numeric_cols"]
        assert "antigüedad_meses" in profile["numeric_cols"]

        # Columnas categóricas esperadas
        assert "region" in profile["categorical_cols"]
        assert "genero" in profile["categorical_cols"]

        # id_cliente tiene ~2000 únicos → alta cardinalidad
        assert "id_cliente" in profile["high_cardinality_cols"]

        # El target NO aparece en las listas de features
        assert "churn" not in profile["numeric_cols"]
        assert "churn" not in profile["categorical_cols"]
        assert "churn" not in profile["high_cardinality_cols"]

    def test_profile_dataset_no_target(self):
        """Sin target, task_type_candidates es clustering."""
        df = pd.read_csv(PRECIOS_PATH)
        profile = profile_dataset(df)

        assert profile["target"] is None
        assert profile["task_type_candidates"] == [
            "clustering", "dimensionality_reduction"
        ]

    def test_profile_dataset_high_cardinality_from_segmentation(self):
        """id (100% único) va a high_cardinality_cols."""
        df = pd.read_csv(SEGMENTACION_REAL_PATH)
        profile = profile_dataset(df)

        assert "id" in profile["high_cardinality_cols"]
        # Buscar el column info de id
        col_info = next(c for c in profile["columns"] if c["name"] == "id")
        assert col_info["cardinality_ratio"] == 1.0
        assert col_info["type"] == "categorical"

    def test_profile_dataset_correlations_exist_with_two_or_more_numeric(self):
        """Dos o más numéricas → correlations.matrix no vacío."""
        df = pd.read_csv(CHURN_PATH)
        profile = profile_dataset(df, target="churn")

        assert profile["correlations"]["matrix"]
        assert isinstance(profile["correlations"]["high_pairs"], list)

    def test_profile_dataset_memory_and_duplicates(self):
        """Campos de metadata general se incluyen."""
        df = pd.read_csv(CHURN_PATH)
        profile = profile_dataset(df, target="churn")

        assert profile["memory_mb"] > 0
        assert profile["duplicated_rows"] >= 0


# ====================================================================
# _profile_target — tipos y concentración
# ====================================================================

class TestProfileTarget:
    """_profile_target() — tipo de tarea, concentración y balanceo."""

    def test_target_continuous_precio_is_regression(self):
        """precio: ~1000 valores únicos → regression, concentración normal."""
        df = pd.read_csv(PRECIOS_PATH)
        target = _profile_target(df, "precio")

        assert target["task_type"] == "regression"
        assert target["task_candidates"] == ["regression"]
        assert target["concentration_level"] == "normal"
        assert target["unique"] > 15
        assert "min" in target
        assert "max" in target
        assert "mean" in target
        assert "std" in target
        assert "distribution" in target
        assert "p25" in target["distribution"]
        assert "p50" in target["distribution"]
        assert "p75" in target["distribution"]
        # Precio es positivo
        assert target["min"] > 0

    def test_target_extreme_concentration_calificacion(self):
        """calificacion: ~95% en 0.0 → classification, extreme concentration."""
        df = pd.read_csv(PRODUCTOS_PATH)
        target = _profile_target(df, "calificacion")

        assert target["concentration_level"] == "extreme"
        # majority_pct ~ 95%
        assert target["majority_pct"] == pytest.approx(94.99, abs=0.5)
        assert target["majority_value"] == 0.0
        assert target["unique"] == 10
        # Tarea es clasificación
        assert target["task_type"] == "classification"
        assert "balance_ratio" in target
        assert "balance_level" in target
        assert "balance_strategy" in target
        assert target["use_weighted_metrics"] is True

    def test_target_binary_churn_is_severe(self):
        """churn: target binario 0/1 con ~64% → severe concentration."""
        df = pd.read_csv(CHURN_PATH)
        target = _profile_target(df, "churn")

        assert target["task_type"] == "classification"
        assert target["task_candidates"] == ["classification"]
        assert target["unique"] == 2
        # majority > 60% → severe (puede variar ligeramente)
        assert target["majority_pct"] > 60
        assert target["concentration_level"] == "severe"
        # distribución binaria
        assert isinstance(target["distribution"], dict)
        assert isinstance(target["distribution_pct"], dict)
        assert target["balance_ratio"] > 0
        assert target["balance_ratio"] < 1.0

    def test_target_no_target_key_error(self):
        """Llamar _profile_target con columna inexistente levanta KeyError."""
        df = pd.read_csv(CHURN_PATH)
        with pytest.raises(KeyError):
            _profile_target(df, "columna_que_no_existe")


# ====================================================================
# _profile_column — estadísticas numéricas y detección de tipos
# ====================================================================

class TestProfileColumn:
    """_profile_column() — tipo, nulls, cardinalidad y estadísticas."""

    def test_numeric_column_has_stats(self):
        """Columna numérica reporta min/max/mean/skew correctamente."""
        df = pd.read_csv(CHURN_PATH)
        info = _profile_column(df, "edad")

        assert info["type"] == "numeric"
        assert info["dtype"].startswith("int") or info["dtype"].startswith("float")
        assert isinstance(info["min"], float)
        assert isinstance(info["max"], float)
        assert isinstance(info["mean"], float)
        assert isinstance(info["std"], float)
        assert isinstance(info["skew"], float)
        # Edad tiene valores sensibles
        assert 0 <= info["min"] <= info["mean"] <= info["max"]

    def test_numeric_column_values_smoke_check(self):
        """Verificar valores específicos de columna numérica."""
        df = pd.read_csv(CHURN_PATH)
        info = _profile_column(df, "gasto_total_anual")

        assert info["type"] == "numeric"
        assert info["mean"] > 0
        assert info["min"] >= 0  # puede haber 0 o inf, pero no negativo
        assert info["unique"] > 100  # muchas filas, valores continuos

    def test_categorical_column_detection(self):
        """Columna categórica de string se detecta."""
        df = pd.read_csv(CHURN_PATH)
        info = _profile_column(df, "region")

        assert info["type"] == "categorical"
        # region tiene > 20 únicos → NO debe tener values (supera el límite)
        # En churn_real las regiones son ~5-10 ciudades + variantes → revisar

    def test_categorical_numeric_detection(self):
        """Columna numérica con pocos únicos → categorical_numeric."""
        # Crear un DataFrame pequeño con columna numérica de pocos valores
        df = pd.DataFrame({"score": [1, 2, 3, 1, 2, 3, 1, 2, 3]})
        info = _profile_column(df, "score")

        assert info["type"] == "categorical_numeric"
        assert info["unique"] <= 10
        assert "values" in info

    def test_high_cardinality_detection(self):
        """Columna con muchos valores únicos → cardinality_ratio alto."""
        df = pd.read_csv(CHURN_PATH)
        info = _profile_column(df, "id_cliente")

        assert info["type"] == "categorical"
        assert info["cardinality_ratio"] > 0.9  # casi todas únicas
        assert info["unique"] > 1000

    def test_column_with_nulls(self):
        """Columnas con nulos reportan null_pct correcto."""
        df = pd.read_csv(CHURN_PATH)
        # Varias columnas tienen nulos (ingreso_mensual, etc.)
        info = _profile_column(df, "ingreso_mensual")

        assert info["nulls"] >= 0
        assert info["null_pct"] >= 0
        assert info["null_pct"] <= 100

    def test_high_cardinality_column_in_profile_dataset(self):
        """Los IDs con cardinality_ratio > 0.5 van a high_cardinality_cols."""
        # Segmentación real tiene id 100% único
        df = pd.read_csv(SEGMENTACION_REAL_PATH)
        profile = profile_dataset(df, target=None)

        # Buscar info de id
        col_infos = [c for c in profile["columns"] if c["name"] == "id"]
        assert len(col_infos) == 1
        info = col_infos[0]
        assert info["cardinality_ratio"] == 1.0
        assert info["unique"] == len(df)

    def test_categorical_numeric_boolean(self):
        """Booleano con pocos únicos se detecta como categorical_numeric
        (porque is_numeric_dtype antecede a is_bool_dtype en el profiler)."""
        arr = pd.array([True, False, True, True], dtype="boolean")
        df = pd.DataFrame({"flag": arr})
        info = _profile_column(df, "flag")

        assert info["type"] == "categorical_numeric"
        assert info["unique"] == 2
# ====================================================================
# profile_summary_text — (smoke)
# ====================================================================

class TestProfileSummary:
    """profile_summary_text() genera texto sin errores."""

    def test_summary_text_generation(self):
        """profile_summary_text produce al menos 10 líneas de texto."""
        from src.profiler import profile_summary_text

        df = pd.read_csv(CHURN_PATH)
        profile = profile_dataset(df, target="churn")
        summary = profile_summary_text(profile)

        assert isinstance(summary, str)
        assert len(summary.splitlines()) >= 10
        # Debe mencionar el target
        assert "churn" in summary
