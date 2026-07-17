"""
tests/test_trainer.py — Tests unitarios y de integración para src/trainer.py

Cubre:
  1. train_models() con clasificación + class_weight  → RF CV > 0.9
  2. train_models() con regresión (precio viviendas)   → RFR CV > 0.5
  3. ensemble_predictions() produce scores con r2, mse, mae, rmse
  4. ranked_models ordenados por CV descendente
  5. LinearRegression puede fallar (CV≈0) sin romper el pipeline
  6. Integración: pipeline CSV → reporte con secciones esperadas
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

from src.trainer import train_models, ensemble_predictions
from cli import run_pipeline
from src.config import PipelineConfig



# ── Fixtures compartidos ─────────────────────────────────────────────


@pytest.fixture
def cfg():
    """PipelineConfig rápido para tests unitarios: 3 folds, sin LLM, sin guardar."""
    c = PipelineConfig(
        cv_folds=3,
        random_state=42,
        save_report=False,
    )
    c.feature_selection = False  # no perder features sintéticas
    return c


# ── Helpers ──────────────────────────────────────────────────────────


def _clf_data(n=300, seed=42):
    """Datos sintéticos de clasificación binaria con señal fuerte."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    # Señal logística limpia con poca superposición → CV alto
    logit = X["x1"] * 1.5 + X["x2"] * 1.2 - X["x3"] * 0.5
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    return X, y


def _reg_data(n=200, seed=42):
    """Datos sintéticos de regresión con señal moderada."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "sqft": rng.uniform(500, 5000, n),
        "bedrooms": rng.integers(1, 6, n).astype(float),
        "age": rng.uniform(0, 50, n),
    })
    y = (100000
         + X["sqft"] * 80
         + X["bedrooms"] * 20000
         - X["age"] * 500
         + rng.normal(0, 5000, n))
    return X, y


def _clf_plan():
    """Plan mínimo para clasificación binaria (class_weight activo)."""
    return {
        "task_type": "classification",
        "metrics": ["accuracy", "f1"],
        "target_weighting": {},
        "models": [
            {"name": "RandomForestClassifier", "params": {"class_weight": "balanced"}},
            {"name": "LogisticRegression", "params": {"class_weight": "balanced", "max_iter": 1000}},
        ],
    }


def _reg_plan():
    """Plan mínimo para regresión."""
    return {
        "task_type": "regression",
        "metrics": ["r2", "mse", "mae", "rmse"],
        "target_weighting": {},
        "models": [
            {"name": "RandomForestRegressor", "params": {}},
            {"name": "LinearRegression", "params": {}},
            {"name": "Ridge", "params": {}},
        ],
    }


# ═════════════════════════════════════════════════════════════════════
#  1. Clasificación + class_weight → RF CV > 0.9
# ═════════════════════════════════════════════════════════════════════


class TestClassification:
    """train_models con tarea de clasificación y class_weight."""

    def test_random_forest_achieves_high_cv(self, cfg):
        """RandomForestClassifier con class_weight obtiene CV > 0.9."""
        X, y = _clf_data(n=300)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        results = train_models(X_train, y_train, _clf_plan(), config=cfg, verbose=False)

        assert "ranked_models" in results
        assert len(results["ranked_models"]) > 0

        rf = next(r for r in results["ranked_models"] if r["name"] == "RandomForestClassifier")
        assert rf["cv_mean"] > 0.9, (
            f"RandomForestClassifier CV={rf['cv_mean']:.4f}, esperado > 0.9"
        )


# ═════════════════════════════════════════════════════════════════════
#  2. Regresión (precio viviendas) → RFR CV > 0.5
# ═════════════════════════════════════════════════════════════════════


class TestRegression:
    """train_models con tarea de regresión."""

    def test_random_forest_regressor_achieves_high_cv(self):
        """RandomForestRegressor con datos de viviendas obtiene CV > 0.5."""
        df = pd.read_csv(os.path.join(DATASETS_DIR, "precios_viviendas.csv"))
        plan = _reg_plan()

        # Usar solo columnas numéricas
        X = df[["metros2", "habitaciones", "antigüedad", "distancia_centro",
                 "numero_pisos", "calidad_acabados"]]
        y = df["precio"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        cfg = PipelineConfig(cv_folds=3, random_state=42, feature_selection=False)
        results = train_models(X_train, y_train, plan, config=cfg, verbose=False)

        rfr = next(r for r in results["ranked_models"] if r["name"] == "RandomForestRegressor")
        assert rfr["cv_mean"] > 0.5, (
            f"RandomForestRegressor CV={rfr['cv_mean']:.4f}, esperado > 0.5"
        )


# ═════════════════════════════════════════════════════════════════════
#  3. ensemble_predictions → scores r2, mse, mae, rmse
# ═════════════════════════════════════════════════════════════════════


class TestEnsemble:
    """ensemble_predictions produce scores completos para regresión."""

    def test_ensemble_scores_include_all_regression_metrics(self, cfg):
        """Scores del ensemble contienen r2, mse, mae, rmse."""
        X, y = _reg_data(n=200)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        train_results = train_models(
            X_train, y_train, _reg_plan(), config=cfg, verbose=False,
        )
        ensemble_result = ensemble_predictions(
            X_test, y_test, train_results, _reg_plan(),
            config=cfg, verbose=False,
        )

        assert "scores" in ensemble_result, "ensemble_result debe tener 'scores'"
        scores = ensemble_result["scores"]
        for metric in ("r2", "mse", "mae", "rmse"):
            assert metric in scores, (
                f"Falta métrica '{metric}' en ensemble scores: {list(scores.keys())}"
            )
            assert isinstance(scores[metric], (int, float)), (
                f"score['{metric}'] = {scores[metric]!r} no es numérico"
            )


# ═════════════════════════════════════════════════════════════════════
#  4. ranked_models ordenados por CV descendente
# ═════════════════════════════════════════════════════════════════════


class TestRanking:
    """Los modelos rankeados deben estar ordenados por CV descendente."""

    def test_ranked_models_sorted_by_cv_descending(self, cfg):
        """ranked_models está ordenado de mayor a menor cv_mean."""
        X, y = _clf_data(n=200)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        results = train_models(X_train, y_train, _clf_plan(), config=cfg, verbose=False)

        ranked = results["ranked_models"]
        assert len(ranked) >= 2, "Se necesitan al menos 2 modelos rankeados"

        cv_values = [r["cv_mean"] for r in ranked]
        assert cv_values == sorted(cv_values, reverse=True), (
            f"ranked_models no está ordenado descendente: CVs = {cv_values}"
        )


# ═════════════════════════════════════════════════════════════════════
#  5. LinearRegression falla (CV≈0) sin romper el pipeline
# ═════════════════════════════════════════════════════════════════════


class TestLinearRegressionFailsGracefully:
    """LinearRegression con CV≈0 no debe romper el pipeline."""

    def test_low_cv_model_does_not_break_pipeline(self, cfg):
        """Cuando LR da CV≈0, otros modelos y ensemble aún funcionan."""
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        })
        # Target = ruido puro → R² ≈ 0 para LR
        y = rng.normal(0, 1, n)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        plan = _reg_plan()
        results = train_models(X_train, y_train, plan, config=cfg, verbose=False)

        # LR debe haber sido incluido en los resultados
        lr_result = next(
            (r for r in results["ranked_models"] if r["name"] == "LinearRegression"),
            None,
        )
        if lr_result is not None:
            assert lr_result["cv_mean"] < 0.1, (
                f"LinearRegression CV={lr_result['cv_mean']:.4f} debe ser ≈ 0 "
                "en target ruidoso"
            )

        # El pipeline NO se rompió: otros modelos deben existir y tener CV válido
        assert len(results["ranked_models"]) > 1, (
            f"Solo {len(results['ranked_models'])} modelos; "
            "LR con fallo no debe detener a los demás"
        )
        other_cvs = [
            r["cv_mean"] for r in results["ranked_models"]
            if r["name"] != "LinearRegression"
        ]
        assert all(cv > -1 for cv in other_cvs), (
            "Modelos no-LR deben tener CV > -1"
        )

        # Ensemble aún debe producir predicciones
        ensemble_result = ensemble_predictions(
            X_test, y_test, results, plan,
            config=cfg, verbose=False,
        )
        assert "predictions" in ensemble_result
        if ensemble_result["predictions"] is not None:
            assert len(ensemble_result["predictions"]) == len(y_test)


# ═════════════════════════════════════════════════════════════════════
#  6. Integración: pipeline CSV → reporte
# ═════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Pipeline completo desde CSV hasta reporte final."""

    def test_full_pipeline_produces_report_with_expected_sections(self):
        """run_pipeline retorna un reporte con todas las secciones esperadas."""
        csv_path = os.path.join(DATASETS_DIR, "precios_viviendas.csv")
        df = pd.read_csv(csv_path)

        cfg = PipelineConfig(
            cv_folds=3,
            random_state=42,
            save_report=False,
            feature_selection=False,
        )

        result = run_pipeline(
            df,
            target="precio",
            force_task="regression",
            config=cfg,
            use_npc=False,
            verbose=False,
        )

        # Estructura del resultado
        for key in ("profile", "plan", "pipeline_result", "train_results",
                     "ensemble_result", "report", "elapsed_seconds"):
            assert key in result, f"Falta '{key}' en resultado del pipeline"

        # Reporte: string no vacío
        report = result["report"]
        assert isinstance(report, str) and len(report) > 100, (
            "Reporte debe ser un string sustancial"
        )

        # Secciones obligatorias del reporte
        expected_sections = [
            "REPORTE DE ANÁLISIS",
            "DATOS",
            "PREPROCESAMIENTO",
            "MODELOS ENTRENADOS",
            "ENSEMBLE",
            "RECOMENDACIONES",
            "MELO",
        ]
        for section in expected_sections:
            assert section in report, (
                f"Sección '{section}' no encontrada en el reporte"
            )

        # Modelos entrenados: al menos un nombre de modelo aparece
        assert "RandomForestRegressor" in report or "Ridge" in report or "LinearRegression" in report, (
            "Debe mencionar al menos un modelo entrenado en el reporte"
        )

        # CV scores visibles en el reporte
        assert "CV=" in report, "Reporte debe contener scores de CV"

        # Métricas del ensemble
        assert "r2" in report or "Ensemble" in report, (
            "Reporte debe contener métricas del ensemble"
        )

        # Train results contiene modelos rankeados
        train_results = result["train_results"]
        assert "ranked_models" in train_results
        assert len(train_results["ranked_models"]) > 0
        assert train_results["best_score"] > 0, (
            f"best_score debe ser > 0, got {train_results['best_score']}"
        )
