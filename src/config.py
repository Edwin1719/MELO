"""
config.py — Configuraciones, tipos de tarea y catálogo de modelos
para el pipeline MELO.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal

# ── Tipos de tarea soportados ──────────────────────────────────────

TaskType = Literal[
    "classification",
    "regression",
    "clustering",
    "dimensionality_reduction",
    "anomaly_detection",
]

# ── Catálogo de modelos por tipo de tarea ──────────────────────────

CLASSIFICATION_MODELS = {
    "LogisticRegression": {
        "module": "sklearn.linear_model.LogisticRegression",
        "grid": {"C": [0.1, 1.0, 10.0], "max_iter": [1000]},
        "supports_weight": True,
    },
    "RandomForestClassifier": {
        "module": "sklearn.ensemble.RandomForestClassifier",
        "grid": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
        "supports_weight": True,
    },
    "GradientBoostingClassifier": {
        "module": "sklearn.ensemble.GradientBoostingClassifier",
        "grid": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
        "supports_weight": True,
    },
    "SVC": {
        "module": "sklearn.svm.SVC",
        "grid": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"], "probability": [True]},
        "supports_weight": True,
    },
    "KNeighborsClassifier": {
        "module": "sklearn.neighbors.KNeighborsClassifier",
        "grid": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        "supports_weight": False,
    },
}

REGRESSION_MODELS = {
    "LinearRegression": {
        "module": "sklearn.linear_model.LinearRegression",
        "grid": {},
        "supports_weight": False,
    },
    "Ridge": {
        "module": "sklearn.linear_model.Ridge",
        "grid": {"alpha": [0.1, 1.0, 10.0]},
        "supports_weight": True,
    },
    "RandomForestRegressor": {
        "module": "sklearn.ensemble.RandomForestRegressor",
        "grid": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
        "supports_weight": True,
    },
    "GradientBoostingRegressor": {
        "module": "sklearn.ensemble.GradientBoostingRegressor",
        "grid": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
        "supports_weight": True,
    },
    "SVR": {
        "module": "sklearn.svm.SVR",
        "grid": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
        "supports_weight": False,
    },
}

CLUSTERING_MODELS = {
    "KMeans": {
        "module": "sklearn.cluster.KMeans",
        "grid": {"n_clusters": [3, 4, 5, 6, 7, 8], "n_init": [10]},
        "params_no_target": True,
    },
    "DBSCAN": {
        "module": "sklearn.cluster.DBSCAN",
        "grid": {"eps": [0.3, 0.5, 0.7, 1.0], "min_samples": [3, 5, 10]},
        "params_no_target": True,
    },
    "AgglomerativeClustering": {
        "module": "sklearn.cluster.AgglomerativeClustering",
        "grid": {"n_clusters": [3, 4, 5, 6, 7, 8], "linkage": ["ward", "complete"]},
        "params_no_target": True,
    },
}

DIM_REDUCTION_MODELS = {
    "PCA": {
        "module": "sklearn.decomposition.PCA",
        "grid": {"n_components": [0.85, 0.90, 0.95], "svd_solver": ["auto"]},
        "params_no_target": True,
    },
    "TSNE": {
        "module": "sklearn.manifold.TSNE",
        "grid": {"perplexity": [5, 15, 30, 50], "n_iter": [1000]},
        "params_no_target": True,
    },
}

ANOMALY_MODELS = {
    "IsolationForest": {
        "module": "sklearn.ensemble.IsolationForest",
        "grid": {
            "contamination": ["auto", 0.05, 0.1],
            "n_estimators": [100],
        },
        "supports_weight": False,
    },
    "LocalOutlierFactor": {
        "module": "sklearn.neighbors.LocalOutlierFactor",
        "grid": {
            "contamination": ["auto", 0.05, 0.1],
            "n_neighbors": [10, 20, 30],
        },
        "supports_weight": False,
    },
}

# ── Estrategias de balanceo ─────────────────────────────────────────

BALANCE_THRESHOLDS = {
    "balanced": 0.8,
    "mild": 0.5,
    "moderate": 0.2,
    "severe": 0.1,
    "extreme": 0.0,
}

BALANCE_STRATEGIES = {
    "balanced": "none",
    "mild": "class_weight",
    "moderate": "smote",
    "severe": "smote_enn",
    "extreme": "adasyn",
}

# ── Métricas por tipo de tarea ─────────────────────────────────────

METRICS_MAP = {
    "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "classification_imbalanced": ["f1", "precision", "recall", "roc_auc"],
    "regression": ["r2", "mse", "mae", "rmse"],
    "clustering": ["silhouette", "davies_bouldin", "calinski_harabasz"],
    "dimensionality_reduction": ["explained_variance_ratio"],
    "anomaly_detection": ["silhouette"],
}


# ── Configuración del pipeline ─────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuración global del pipeline AutoML."""

    # Procesamiento
    test_size: float = 0.2
    val_size: float = 0.0
    random_state: int = 42
    scale_method: str = "standard"  # standard, minmax, robust, maxabs
    categorical_strategy: str = "auto"  # auto, onehot, ordinal, target

    # Balanceo
    balance_method: Optional[str] = None  # auto-detect si None
    smote_k_neighbors: int = 5
    smote_enn_k_neighbors: int = 3

    # Feature selection
    feature_selection: bool = True
    feature_selection_method: str = "importance"  # importance, mutual_info, variance
    max_features_ratio: float = 0.8
    min_features: int = 2

    # Entrenamiento
    cv_folds: int = 5
    n_top_ensemble: int = 3
    ensemble_method: str = "weighted"  # vote, weighted

    # NPC
    npc_model: str = os.getenv("NPC_MODEL", "qwen2.5-coder:7b")
    npc_provider: str = "ollama"
    npc_temperature: float = 0.3

    # Output
    output_dir: str = "output"
    save_report: bool = True
    save_model: bool = False


# ── Helpers ─────────────────────────────────────────────────────────

TASK_MODEL_MAP = {
    "classification": CLASSIFICATION_MODELS,
    "regression": REGRESSION_MODELS,
    "clustering": CLUSTERING_MODELS,
    "dimensionality_reduction": DIM_REDUCTION_MODELS,
    "anomaly_detection": ANOMALY_MODELS,
}

TASK_DESCRIPTIONS = {
    "classification": "Clasificación supervisada (target categórico)",
    "regression": "Regresión supervisada (target numérico continuo)",
    "clustering": "Clustering no supervisado (sin target, segmentación)",
    "dimensionality_reduction": "Reducción de dimensionalidad no supervisada",
    "anomaly_detection": "Detección de anomalías no supervisada",
}


def get_model_catalog(task_type: str) -> dict:
    """Retorna el catálogo de modelos para un tipo de tarea."""
    return TASK_MODEL_MAP.get(task_type, {})


def get_metrics_for_task(task_type: str, is_imbalanced: bool = False) -> list:
    """Retorna las métricas relevantes para el tipo de tarea."""
    if task_type == "classification" and is_imbalanced:
        return METRICS_MAP["classification_imbalanced"]
    return METRICS_MAP.get(task_type, ["accuracy"])
