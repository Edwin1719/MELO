"""
trainer.py — Auto-ML: entrena múltiples modelos con grid search,
cross-validation, y combina los mejores en un ensemble.

Reutiliza la lógica de npcpy ml_funcs cuando es posible.
"""

import importlib
import itertools
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── Métricas disponibles ────────────────────────────────────────────

CLASSIFICATION_METRICS = {
    "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    "f1": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
    "precision": lambda yt, yp: precision_score(yt, yp, average="weighted", zero_division=0),
    "recall": lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0),
}

REGRESSION_METRICS = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
}


def _import_model(module_path: str):
    """Importa dinámicamente una clase de modelo."""
    parts = module_path.rsplit(".", 1)
    mod = importlib.import_module(parts[0])
    return getattr(mod, parts[1])


def train_models(
    X_train,
    y_train,
    plan: dict,
    config=None,
    verbose: bool = True,
    X_test=None,
    y_test=None,
) -> dict:
    """
    Entrena múltiples modelos según el plan del estratega.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : array-like
    plan : dict
        Plan con lista de modelos a entrenar.
    config : PipelineConfig or None
    verbose : bool

    Returns
    -------
    dict con resultados de cada modelo + ranking.
    """
    cv_folds = getattr(config, "cv_folds", 5) if config else 5
    random_state = getattr(config, "random_state", 42) if config else 42
    task_type = plan.get("task_type", "classification")
    metrics = plan.get("metrics", ["accuracy"])

    results = []
    metric_funcs = _get_metric_funcs(task_type)

    for model_spec in plan.get("models", []):
        model_name = model_spec["name"]
        model_params = model_spec.get("params", {})

        if verbose:
            print(f"  🏋️  {model_name}...", end=" ", flush=True)

        start = time.time()

        # Buscar en catálogo de config.py
        catalog = _get_catalog_for_task(task_type)
        entry = catalog.get(model_name)
        if not entry:
            if verbose:
                print(f"⏭️  (no disponible)")
            continue

        try:
            model_class = _import_model(entry["module"])
        except (ImportError, AttributeError):
            if verbose:
                print(f"⏭️  (import error)")
            continue

        # Grid search: probar todas las combinaciones
        grid_params = entry.get("grid", {})
        param_keys = list(grid_params.keys())
        param_values = list(grid_params.values())

        best_score = -np.inf
        best_model = None
        best_params = {}
        all_cv_runs = []

        n_combos = max(
            1,
            len(list(itertools.product(*param_values))) if param_values else 1,
        )

        for combo_values in itertools.product(*param_values):
            params = dict(zip(param_keys, combo_values))
            merged_params = {**params, **model_params}
            merged_params["random_state"] = random_state

            try:
                model = model_class(**merged_params)

                # Calcular sample_weights para target concentrado
                _fit_params = {}
                tweight = plan.get("target_weighting", {})
                if tweight.get("method") == "inverse_frequency":
                    _vals = np.asarray(y_train)
                    _unique, _counts = np.unique(_vals, return_counts=True)
                    _freq = {u: c / len(_vals) for u, c in zip(_unique, _counts)}
                    _sw = np.array([1.0 / _freq[v] for v in _vals])
                    _fit_params["sample_weight"] = _sw / _sw.mean()

                # Cross-validation
                scorer = _get_default_scorer(task_type)
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_folds,
                    scoring=scorer,
                    error_score="raise",
                    params=_fit_params or None,
                )
                mean_cv = cv_scores.mean()

                all_cv_runs.append({
                    "params": merged_params,
                    "cv_mean": mean_cv,
                    "cv_std": cv_scores.std(),
                    "cv_scores": cv_scores.tolist(),
                })

                if mean_cv > best_score:
                    best_score = mean_cv
                    best_params = merged_params
            except Exception as e:
                if verbose:
                    print(f"(error: {type(e).__name__})", end=" ")
                continue

        test_scores = {}
        if best_model is None and all_cv_runs:
            best_params = all_cv_runs[0]["params"]
            try:
                best_model = model_class(**best_params)
                best_model.fit(X_train, y_train, **_fit_params)

                # Evaluar en test set
                if X_test is not None and y_test is not None:
                    y_pred = best_model.predict(X_test)
                    test_scores = _compute_metrics(y_test, y_pred, task_type)
                    if verbose:
                        top_metric = next(iter(test_scores.values()))
                        print(f" test={top_metric:.4f}", end=" ")

                # Guardar modelo si configurado
                if config and getattr(config, "save_model", False):
                    import joblib
                    model_name = model_spec["name"].replace(" ", "_")
                    out_dir = getattr(config, "output_dir", "output")
                    os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, f"{model_name}_{task_type}.pkl")
                    joblib.dump(best_model, path)

            except Exception:
                continue

        elapsed = time.time() - start

        all_cv_runs.sort(key=lambda x: -x["cv_mean"])

        result = {
            "name": model_name,
            "best_params": best_params,
            "cv_mean": all_cv_runs[0]["cv_mean"] if all_cv_runs else 0,
            "cv_std": all_cv_runs[0]["cv_std"] if all_cv_runs else 0,
            "n_combos_tried": len(all_cv_runs),
            "model": best_model,
            "time_seconds": round(elapsed, 2),
            "all_cv": all_cv_runs,
            "test_scores": test_scores if test_scores else None,
        }

        results.append(result)

        if verbose:
            print(
                f"✓ CV={result['cv_mean']:.4f} "
                f"±{result['cv_std']:.4f} "
                f"({elapsed:.1f}s)"
            )

    # Ranking por CV score
    results.sort(key=lambda r: -r.get("cv_mean", 0))
    # Evaluar en test si existe
    top = results[0] if results else None
    return {
        "ranked_models": results,
        "top_model": top,
        "best_score": top["cv_mean"] if top else 0,
        "best_test_scores": top.get("test_scores") if top else None,
    }


def ensemble_predictions(
    X_test,
    y_test,
    train_results: dict,
    plan: dict,
    config=None,
    verbose: bool = True,
) -> dict:
    """
    Combina los mejores modelos en un ensemble por votación ponderada.

    Parameters
    ----------
    X_test : pd.DataFrame
    y_test : array-like
    train_results : dict
        Resultados de train_models().
    plan : dict
    config : PipelineConfig
    verbose : bool

    Returns
    -------
    dict con predicciones del ensemble + métricas.
    """
    n_top = getattr(config, "n_top_ensemble", 3) if config else 3
    method = getattr(config, "ensemble_method", "weighted") if config else "weighted"
    task_type = plan.get("task_type", "classification")

    top_models = train_results.get("ranked_models", [])[:n_top]

    if len(top_models) < 2:
        if verbose:
            print("  ⚠️  Muy pocos modelos para ensemble")
        model = top_models[0]["model"] if top_models else None
        if model and X_test is not None:
            preds = model.predict(X_test)
            return {
                "predictions": preds,
                "scores": _compute_metrics(y_test, preds, task_type),
                "models_used": [top_models[0]["name"]] if top_models else [],
                "method": "single",
            }
        return {"predictions": None, "scores": {}, "models_used": [], "method": "none"}

    all_preds = []
    weights = []
    model_names = []

    for r in top_models:
        model = r["model"]
        if model is None or X_test is None:
            continue
        preds = model.predict(X_test)
        all_preds.append(preds)
        weights.append(r["cv_mean"])
        model_names.append(r["name"])

    if not all_preds:
        return {"predictions": None, "scores": {}, "models_used": [], "method": "none"}

    all_preds = np.array(all_preds)

    if task_type in ("classification",):
        if method == "weighted":
            weights = np.array(weights) / sum(weights)
            # Voto ponderado por fila
            ensemble_pred = np.zeros(len(X_test))
            for i, preds in enumerate(all_preds):
                ensemble_pred += weights[i] * preds
            ensemble_pred = np.round(ensemble_pred).astype(int)
        else:
            # Voto duro
            ensemble_pred, _ = sp_stats.mode(all_preds, axis=0)
            ensemble_pred = ensemble_pred.flatten()
    else:
        # Regresión: promedio ponderado
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(all_preds, axis=0, weights=weights)

    scores = {}
    if y_test is not None:
        scores = _compute_metrics(y_test, ensemble_pred, task_type)

    if verbose:
        print(f"\n  🧬 Ensemble ({method}): {model_names}")

    return {
        "predictions": ensemble_pred,
        "scores": scores,
        "models_used": model_names,
        "method": method,
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _get_metric_funcs(task_type: str) -> dict:
    if task_type == "regression":
        return REGRESSION_METRICS
    return CLASSIFICATION_METRICS


def _get_default_scorer(task_type: str):
    if task_type == "regression":
        return "r2"
    return "accuracy"


def _get_catalog_for_task(task_type: str) -> dict:
    from .config import TASK_MODEL_MAP
    return TASK_MODEL_MAP.get(task_type, {})


def _compute_metrics(y_true, y_pred, task_type: str) -> dict:
    scores = {}
    if task_type == "regression":
        scores["r2"] = round(r2_score(y_true, y_pred), 4)
        scores["mse"] = round(float(mean_squared_error(y_true, y_pred)), 4)
        scores["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 4)
        scores["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    else:
        scores["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
        scores["f1_weighted"] = round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        )
        scores["precision_weighted"] = round(
            precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
        )
        scores["recall_weighted"] = round(
            recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
        )
        # ROC AUC solo para binaria
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            try:
                scores["roc_auc"] = round(roc_auc_score(y_true, y_pred), 4)
            except Exception:
                pass
    return scores


def get_regression_metrics(y_true, y_pred):
    return _compute_metrics(y_true, y_pred, "regression")
