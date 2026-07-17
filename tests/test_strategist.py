"""
Tests para strategist.py — build_plan() y _default_plan().

Cubren: auto-switch regression→classification por concentración,
target continuo normal→regresión, clasificación con desbalanceo,
modo no-supervisado (sin target), estructura de preprocessing,
y correspondencia modelos-task_type.
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, TASK_MODEL_MAP
from src.profiler import profile_dataset
from src.strategist import build_plan


# ── Helpers ──────────────────────────────────────────────────────────

_CLS_CATALOG = set(TASK_MODEL_MAP["classification"].keys())
_REG_CATALOG = set(TASK_MODEL_MAP["regression"].keys())
_CLS_CATALOG = set(TASK_MODEL_MAP["classification"].keys())
_CLU_CATALOG = set(TASK_MODEL_MAP["clustering"].keys())
_WEIGHT_MODELS = {
    "LogisticRegression", "RandomForestClassifier",
    "GradientBoostingClassifier", "SVC",
}


def _model_names(plan: dict) -> set:
    return {m["name"] for m in plan["models"]}


def _assert_models_from_catalog(plan: dict, expected_catalog: set):
    names = _model_names(plan)
    assert names.issubset(expected_catalog), (
        f"Models {names} not in {expected_catalog}"
    )


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return PipelineConfig(npc_model="none", random_state=42, feature_selection=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. Auto-switch: regression forzada sobre target discreto+concentrado
# ═══════════════════════════════════════════════════════════════════════

class TestRegressionForceAutoSwitch:
    """force_task='regression' sobre target ≤15 valores y concentración extrema."""

    @pytest.fixture
    def df_discrete_concentrated(self):
        """Target calificacion: 3 valores (0,1,2), 99.6% en 0."""
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.uniform(0, 100, n),
            "feature3": rng.choice(["cat_a", "cat_b", "cat_c"], n),
            "calificacion": np.r_[np.full(n - 2, 0), [1, 2]],
        }).sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def test_auto_switches_to_classification(self, df_discrete_concentrated):
        """El plan cambia task_type de regression a classification."""
        profile = profile_dataset(df_discrete_concentrated, target="calificacion")
        plan = build_plan(profile, target="calificacion", force_task="regression")

        assert plan["task_type"] == "classification", (
            f"Esperaba classification, obtuve '{plan['task_type']}'"
        )

    def test_balance_strategy_class_weight(self, df_discrete_concentrated):
        """Clase minoritaria con <2 muestras → class_weight."""
        profile = profile_dataset(df_discrete_concentrated, target="calificacion")
        plan = build_plan(profile, target="calificacion", force_task="regression")

        assert plan["balance"]["strategy"] == "class_weight", (
            f"Esperaba class_weight, obtuve '{plan['balance'].get('strategy')}'"
        )

    def test_target_weighting_empty(self, df_discrete_concentrated):
        """Al cambiar a classification, target_weighting debe ser {}."""
        profile = profile_dataset(df_discrete_concentrated, target="calificacion")
        plan = build_plan(profile, target="calificacion", force_task="regression")

        assert plan.get("target_weighting") == {}, (
            "target_weighting debe estar vacío en classification"
        )

    def test_models_are_classification(self, df_discrete_concentrated):
        """Modelos del catálogo de clasificación."""
        profile = profile_dataset(df_discrete_concentrated, target="calificacion")
        plan = build_plan(profile, target="calificacion", force_task="regression")

        _assert_models_from_catalog(plan, _CLS_CATALOG)

    def test_supported_models_get_class_weight(self, df_discrete_concentrated):
        """Modelos que soportan weight reciben class_weight='balanced'."""
        profile = profile_dataset(df_discrete_concentrated, target="calificacion")
        plan = build_plan(profile, target="calificacion", force_task="regression")

        for m in plan["models"]:
            if m["name"] in _WEIGHT_MODELS:
                assert m["params"].get("class_weight") == "balanced", (
                    f"{m['name']} debería tener class_weight='balanced'"
                )

    def test_metrics_exclude_accuracy(self, df_discrete_concentrated):
        """Plan imbalanceado → sin accuracy en métricas."""
        profile = profile_dataset(df_discrete_concentrated, target="calificacion")
        plan = build_plan(profile, target="calificacion", force_task="regression")

        assert "accuracy" not in plan["metrics"], (
            "accuracy debe excluirse para clasificación imbalanceada"
        )
        assert "roc_auc" in plan["metrics"]
        assert "f1" in plan["metrics"]


# ═══════════════════════════════════════════════════════════════════════
# 2. Regression forzada sobre target continuo normal
# ═══════════════════════════════════════════════════════════════════════

class TestRegressionContinuousNormal:
    """force_task='regression' sobre target continuo (>15 únicos, concentración normal)."""

    @pytest.fixture
    def df_continuous(self):
        rng = np.random.default_rng(42)
        n = 500
        return pd.DataFrame({
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.uniform(0, 100, n),
            "feature3": rng.choice(["A", "B", "C"], n),
            "precio": rng.normal(250_000, 50_000, n).clip(50_000, 500_000),
        })

    def test_stays_regression(self, df_continuous):
        """force_task='regression' con target continuo → se mantiene regression."""
        profile = profile_dataset(df_continuous, target="precio")
        plan = build_plan(profile, target="precio", force_task="regression")

        assert plan["task_type"] == "regression"

    def test_target_weighting_empty(self, df_continuous):
        """Concentración normal → target_weighting = {}."""
        profile = profile_dataset(df_continuous, target="precio")
        plan = build_plan(profile, target="precio", force_task="regression")

        assert plan["target_weighting"] == {}, (
            "target_weighting debe estar vacío para concentración normal"
        )

    def test_balance_empty(self, df_continuous):
        """Regresión → sin balance strategy."""
        profile = profile_dataset(df_continuous, target="precio")
        plan = build_plan(profile, target="precio", force_task="regression")

        assert not plan.get("balance"), (
            "No debe haber balance strategy en regresión"
        )

    def test_models_are_regression(self, df_continuous):
        """Modelos del catálogo de regresión."""
        profile = profile_dataset(df_continuous, target="precio")
        plan = build_plan(profile, target="precio", force_task="regression")

        _assert_models_from_catalog(plan, _REG_CATALOG)

    def test_metrics_are_regression(self, df_continuous):
        """Métricas de regresión (r2, mse, mae, rmse)."""
        profile = profile_dataset(df_continuous, target="precio")
        plan = build_plan(profile, target="precio", force_task="regression")

        assert "r2" in plan["metrics"]
        assert "mse" in plan["metrics"]
        assert "mae" in plan["metrics"]

    def test_no_class_weight_in_params(self, df_continuous):
        """Modelos de regresión no tienen class_weight."""
        profile = profile_dataset(df_continuous, target="precio")
        plan = build_plan(profile, target="precio", force_task="regression")

        for m in plan["models"]:
            assert m["params"] == {}, (
                f"{m['name']} no debería tener params en regresión"
            )


# ═══════════════════════════════════════════════════════════════════════
# 3. Clasificación binaria con desbalanceo (churn)
# ═══════════════════════════════════════════════════════════════════════

class TestClassificationChurn:
    """build_plan() sobre target binario con imbalance moderado."""

    @pytest.fixture
    def df_churn(self):
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "antiguedad": rng.exponential(24, n),
            "gasto": rng.lognormal(4.2, 0.9, n),
            "plan": rng.choice(["basico", "premium", "vip"], n),
            "churn": np.r_[np.full(400, 0), np.full(100, 1)],
        }).sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def test_task_type_classification(self, df_churn):
        """Target binario → task_type = classification."""
        profile = profile_dataset(df_churn, target="churn")
        plan = build_plan(profile, target="churn")

        assert plan["task_type"] == "classification"

    def test_balance_strategy_smote(self, df_churn):
        """Imbalance moderado (80/20) → smote."""
        profile = profile_dataset(df_churn, target="churn")
        plan = build_plan(profile, target="churn")

        assert plan["balance"]["strategy"] == "smote", (
            f"Esperaba smote, obtuve '{plan['balance'].get('strategy')}'"
        )

    def test_balance_models_with_weight_present(self, df_churn):
        """Plan de balance incluye lista de modelos con weight."""
        profile = profile_dataset(df_churn, target="churn")
        plan = build_plan(profile, target="churn")

        assert "models_with_weight" in plan["balance"]
        assert len(plan["balance"]["models_with_weight"]) > 0

    def test_models_are_classification(self, df_churn):
        """Modelos del catálogo de clasificación."""
        profile = profile_dataset(df_churn, target="churn")
        plan = build_plan(profile, target="churn")

        _assert_models_from_catalog(plan, _CLS_CATALOG)

    def test_supported_models_get_weight(self, df_churn):
        """Modelos weight-supporting reciben class_weight='balanced'."""
        profile = profile_dataset(df_churn, target="churn")
        plan = build_plan(profile, target="churn")

        for m in plan["models"]:
            if m["name"] in _WEIGHT_MODELS:
                assert m["params"].get("class_weight") == "balanced"

    def test_balance_justification_present(self, df_churn):
        """El plan de balance incluye justificación."""
        profile = profile_dataset(df_churn, target="churn")
        plan = build_plan(profile, target="churn")

        assert "justification" in plan["balance"]
        assert len(plan["balance"]["justification"]) > 10


# ═══════════════════════════════════════════════════════════════════════
# 4. Sin target → no supervisado
# ═══════════════════════════════════════════════════════════════════════

class TestNoTarget:
    """build_plan(target=None) → clustering / anomaly_detection."""

    @pytest.fixture
    def df_no_target(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "feature1": rng.normal(0, 1, 200),
            "feature2": rng.uniform(0, 100, 200),
            "feature3": rng.choice(["A", "B", "C"], 200),
        })

    def test_task_type_unsupervised(self, df_no_target):
        """Sin target → clustering."""
        profile = profile_dataset(df_no_target, target=None)
        plan = build_plan(profile, target=None)

        assert plan["task_type"] in ("clustering", "anomaly_detection", "dimensionality_reduction")

    def test_target_empty_string(self, df_no_target):
        """target se guarda como string vacío."""
        profile = profile_dataset(df_no_target, target=None)
        plan = build_plan(profile, target=None)

        assert plan["target"] == ""

    def test_no_balance(self, df_no_target):
        """No supervisado → sin balance strategy."""
        profile = profile_dataset(df_no_target, target=None)
        plan = build_plan(profile, target=None)

        assert not plan.get("balance"), (
            "No debe haber balance en tarea no supervisada"
        )

    def test_no_target_weighting(self, df_no_target):
        """No supervisado → sin target_weighting."""
        profile = profile_dataset(df_no_target, target=None)
        plan = build_plan(profile, target=None)

        assert plan.get("target_weighting") == {}

    def test_models_are_clustering(self, df_no_target):
        """Modelos del catálogo de clustering."""
        profile = profile_dataset(df_no_target, target=None)
        plan = build_plan(profile, target=None)

        _assert_models_from_catalog(plan, _CLU_CATALOG)

    def test_metrics_are_clustering(self, df_no_target):
        """Métricas de clustering."""
        profile = profile_dataset(df_no_target, target=None)
        plan = build_plan(profile, target=None)

        for metric in ("silhouette", "davies_bouldin"):
            assert metric in plan["metrics"], (
                f"Esperaba métrica {metric} en plan no supervisado"
            )


# ═══════════════════════════════════════════════════════════════════════
# 5. Preprocessing: drop por alta cardinalidad, encode, scale
# ═══════════════════════════════════════════════════════════════════════

class TestPreprocessingSteps:
    """El plan contiene acciones de drop, encode y scale."""

    @pytest.fixture
    def df_with_high_cardinality(self):
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame({
            "numeric_a": rng.normal(0, 1, n),
            "numeric_b": rng.uniform(0, 100, n),
            "high_card_cat": [f"id_{i}" for i in range(n)],
            "low_card_cat": rng.choice(["X", "Y", "Z"], n),
            "binary_cat": rng.choice(["M", "F"], n),
            "target": rng.choice([0, 1], n),
        })

    def test_contains_drop_for_high_cardinality(self, df_with_high_cardinality):
        """Columna con >50 únicos y ratio>0.5 → drop."""
        profile = profile_dataset(df_with_high_cardinality, target="target")
        plan = build_plan(profile, target="target")
        drops = [p for p in plan["preprocessing"] if p["action"] == "drop"]

        assert any("high_card_cat" in d["col"] for d in drops), (
            "high_card_cat debería tener acción drop"
        )

    def test_contains_encode_for_categorical(self, df_with_high_cardinality):
        """Columnas categóricas reciben encode."""
        profile = profile_dataset(df_with_high_cardinality, target="target")
        plan = build_plan(profile, target="target")
        encodes = [p for p in plan["preprocessing"] if p["action"] == "encode"]

        col_names = {e["col"] for e in encodes}
        assert "low_card_cat" in col_names, "low_card_cat debería tener encode"
        assert "binary_cat" in col_names, "binary_cat debería tener encode"

    def test_encode_method_onehot_for_low_card(self, df_with_high_cardinality):
        """Categórica ≤10 valores → encode method='onehot'."""
        profile = profile_dataset(df_with_high_cardinality, target="target")
        plan = build_plan(profile, target="target")
        encodes = {e["col"]: e["method"] for e in plan["preprocessing"]
                   if e["action"] == "encode"}

        assert encodes.get("low_card_cat") == "onehot"
        assert encodes.get("binary_cat") == "label"  # ≤2 → label

    def test_contains_scale_for_numeric(self, df_with_high_cardinality):
        """Columnas numéricas (no-target) reciben scale."""
        profile = profile_dataset(df_with_high_cardinality, target="target")
        plan = build_plan(profile, target="target")
        scales = [p for p in plan["preprocessing"] if p["action"] == "scale"]

        col_names = {s["col"] for s in scales}
        assert "numeric_a" in col_names
        assert "numeric_b" in col_names

    def test_scale_method_standard_for_low_skew(self, df_with_high_cardinality):
        """Skew normal → scale method='standard'."""
        profile = profile_dataset(df_with_high_cardinality, target="target")
        plan = build_plan(profile, target="target")
        scales = {s["col"]: s["method"] for s in plan["preprocessing"]
                  if s["action"] == "scale"}

        assert scales.get("numeric_a") in ("standard", "robust")
        assert scales.get("numeric_b") in ("standard", "robust")

    def test_no_drop_for_target(self, df_with_high_cardinality):
        """El target nunca aparece en preprocessing."""
        profile = profile_dataset(df_with_high_cardinality, target="target")
        plan = build_plan(profile, target="target")

        target_in_preproc = any(
            p["col"] == "target" for p in plan["preprocessing"]
        )
        assert not target_in_preproc, "El target no debe estar en preprocessing"


# ═══════════════════════════════════════════════════════════════════════
# 6. Modelos corresponden al task_type final
# ═══════════════════════════════════════════════════════════════════════

class TestModelsMatchTaskType:
    """Verificación cross-case de que task_type y modelos coinciden."""

    @pytest.mark.parametrize("force,desc", [
        ("classification", "clasificación explícita"),
        ("regression", "regresión explícita"),
        ("clustering", "clustering explícito"),
    ])
    def test_forced_task_models_match(self, force, desc):
        """force_task directo → modelos coinciden con el task_type."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.choice([0, 1, 2], n),
        })
        profile = profile_dataset(df, target="y")
        plan = build_plan(profile, target="y", force_task=force)

        assert plan["task_type"] == force, (
            f"{desc}: task_type debería ser '{force}', es '{plan['task_type']}'"
        )
        _assert_models_from_catalog(plan, set(TASK_MODEL_MAP[force].keys()))

    def test_inferred_classification_models_match(self):
        """Inferencia de clasificación → modelos de clasificación."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 200),
            "y": rng.choice(["yes", "no"], 200),
        })
        profile = profile_dataset(df, target="y")
        plan = build_plan(profile, target="y")

        assert plan["task_type"] == "classification"
        _assert_models_from_catalog(plan, _CLS_CATALOG)

    def test_model_names_are_non_empty(self):
        """Siempre hay al menos un modelo seleccionado."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 200),
            "y": rng.choice([0, 1], 200),
        })
        profile = profile_dataset(df, target="y")
        plan = build_plan(profile, target="y")

        assert len(plan["models"]) >= 1
        for m in plan["models"]:
            assert "name" in m
            assert isinstance(m["name"], str)
            assert len(m["name"]) > 0

    def test_no_regression_model_in_classification_plan(self):
        """Plan de clasificación nunca contiene modelo de regresión."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 200),
            "y": rng.choice([0, 1], 200),
        })
        profile = profile_dataset(df, target="y")
        plan = build_plan(profile, target="y")

        model_names = _model_names(plan)
        assert model_names.isdisjoint(_REG_CATALOG), (
            f"Plan de clasificación contiene modelos de regresión: "
            f"{model_names & _REG_CATALOG}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 7. Invariantes generales del plan
# ═══════════════════════════════════════════════════════════════════════

class TestPlanInvariants:
    """Propiedades que todo plan debe cumplir."""

    @pytest.fixture
    def df_any(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "x1": rng.normal(0, 1, 200),
            "x2": rng.choice(["a", "b"], 200),
            "y": rng.choice([0, 1], 200),
        })

    def test_plan_has_required_keys(self, df_any):
        """Estructura base del plan."""
        profile = profile_dataset(df_any, target="y")
        plan = build_plan(profile, target="y")

        for key in ("task_type", "target", "justification",
                     "preprocessing", "balance", "models",
                     "feature_selection", "metrics"):
            assert key in plan, f"Falta clave '{key}' en el plan"

    def test_preprocessing_is_list(self, df_any):
        """preprocessing es una lista."""
        profile = profile_dataset(df_any, target="y")
        plan = build_plan(profile, target="y")

        assert isinstance(plan["preprocessing"], list)

    def test_each_preprocessing_step_has_col_and_action(self, df_any):
        """Cada step de preprocessing tiene col y action."""
        profile = profile_dataset(df_any, target="y")
        plan = build_plan(profile, target="y")

        for step in plan["preprocessing"]:
            assert "col" in step
            assert "action" in step
            assert step["action"] in ("impute", "encode", "scale", "drop")

    def test_feature_selection_from_config(self, df_any):
        """feature_selection se hereda de PipelineConfig."""
        cfg = PipelineConfig(npc_model="none", feature_selection=False)
        profile = profile_dataset(df_any, target="y")
        plan = build_plan(profile, target="y", config=cfg)

        assert plan["feature_selection"] is False

    def test_justification_is_non_empty(self, df_any):
        """justification es texto no vacío."""
        profile = profile_dataset(df_any, target="y")
        plan = build_plan(profile, target="y")

        assert isinstance(plan["justification"], str)
        assert len(plan["justification"]) > 10
