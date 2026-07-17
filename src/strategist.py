"""
strategist.py — NPC estratega que analiza el perfil del dataset y
decide el plan completo de preprocesamiento y modelos.

Usa un LLM local (Ollama) como "científico de datos" que adapta
las decisiones a cada dataset.
"""

import json
from typing import Optional

from .config import (
    PipelineConfig,
    TASK_MODEL_MAP,
    get_metrics_for_task,
    BALANCE_THRESHOLDS,
)

# ── Template del prompt para el NPC ─────────────────────────────────

STRATEGIST_PROMPT_TEMPLATE = """Eres un experto científico de datos. Tu trabajo es analizar el perfil de un dataset y generar un plan de preprocesamiento y modelado.

## Perfil del dataset:
{profile_text}

## Instrucciones:
1. Identifica el tipo de tarea: classification, regression, clustering, anomaly_detection
2. Para cada columna numérica, decide:
   - impute: "mean", "median", "zero", "drop", o "none" si no tiene nulos
3. Para cada columna categórica, decide:
   - encode: "onehot" (si tiene ≤10 valores), "label" (si es ordinal), "target" (si >10 valores), o "drop"
4. Decide el método de escalado: "standard", "minmax", "robust", o "none"
5. Si es clasificación y hay desbalanceo, incluye balance_strategy
6. Selecciona 3-5 modelos del catálogo que mejor se adapten al problema
7. Define feature_selection: True/False

Responde ÚNICAMENTE con un JSON válido, sin texto adicional, con esta estructura exacta:

{{
    "task_type": "...",
    "target": "{target_name}",
    "justification": "Breve explicación de las decisiones clave",
    "preprocessing": [
        {{"col": "nombre_col", "action": "impute", "method": "median"}},
        {{"col": "nombre_col", "action": "encode", "method": "onehot"}},
        {{"col": "nombre_col", "action": "scale", "method": "standard"}},
        {{"col": "nombre_col", "action": "drop", "reason": "..."}}
    ],
    "balance": {{
        "strategy": "smote",
        "justification": "...",
        "models_with_weight": ["RandomForestClassifier"]
    }},
    "models": [
        {{"name": "RandomForestClassifier", "params": {{"class_weight": "balanced"}}}},
        {{"name": "GradientBoostingClassifier", "params": {{}}}}
    ],
    "feature_selection": true,
    "metrics": ["f1", "precision", "recall", "roc_auc"]
}}"""


def build_plan(
    profile: dict,
    target: Optional[str] = None,
    force_task: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> dict:
    """
    Construye el plan de preprocesamiento usando un NPC vía Ollama.

    Parameters
    ----------
    profile : dict
        Perfil del dataset generado por profiler.profile_dataset().
    target : str or None
        Nombre de la columna objetivo (None si es no supervisado).
    force_task : str or None
        Forzar tipo de tarea (classification, regression, etc.).
    config : PipelineConfig or None
        Configuración del pipeline.

    Returns
    -------
    dict con el plan estructurado.
    """
    cfg = config or PipelineConfig()

    # Generar plan base
    if force_task:
        plan = _default_plan(profile, target, force_task, cfg)
    elif target is None or target not in [c["name"] for c in profile["columns"]]:
        plan = _default_plan(profile, None, "clustering", cfg)
    else:
        target_profile = profile.get("target")
        if target_profile:
            suggested = target_profile.get("task_type", "classification")
        else:
            suggested = _infer_task_from_profile(profile, target)
        plan = _default_plan(profile, target, suggested, cfg)

    # Enhancement opcional: feature engineering con LLM
    if cfg.npc_model != "none" and cfg.npc_provider != "none":
        plan = _enhance_with_features(profile, plan, cfg)

    return plan


def _infer_task_from_profile(profile: dict, target: str) -> str:
    """Infiere el tipo de tarea desde el perfil."""
    target_col = None
    for col in profile["columns"]:
        if col["name"] == target:
            target_col = col
            break
    if not target_col:
        return "clustering"

    if target_col["type"] == "numeric" and target_col.get("unique", 0) > 15:
        return "regression"
    return "classification"


def _default_plan(
    profile: dict,
    target: Optional[str],
    task_type: str,
    config: PipelineConfig,
) -> dict:
    """Genera un plan por defecto basado en reglas + perfil."""
    target_info = profile.get("target") or {}
    is_imbalanced = (
        target_info.get("balance_level", "balanced") not in ("balanced", "mild")
        if target_info
        else False
    )

    preprocessing = []
    numeric_cols = profile.get("numeric_cols", [])
    categorical_cols = profile.get("categorical_cols", [])
    high_cardinality = profile.get("high_cardinality_cols", [])
    high_null = profile.get("high_null_cols", [])

    # Decidir imputación por columna numérica
    for col in numeric_cols:
        if col == target:
            continue
        col_info = next((c for c in profile["columns"] if c["name"] == col), {})
        null_pct = col_info.get("null_pct", 0)

        if null_pct == 0:
            continue  # no necesita imputación
        elif null_pct > 50 and col in high_null:
            preprocessing.append({
                "col": col,
                "action": "drop",
                "reason": f"{null_pct}% nulos",
            })
        else:
            # skew > 1 → mediana, si no → media
            skew = col_info.get("skew", 0) or 0
            method = "median" if abs(skew) > 1 else "mean"
            preprocessing.append({
                "col": col,
                "action": "impute",
                "method": method,
            })

    # Decidir encoding para categóricas
    for col in categorical_cols:
        if col == target:
            continue
        col_info = next((c for c in profile["columns"] if c["name"] == col), {})
        n_unique = col_info.get("unique", 0)

        if col in high_cardinality and n_unique > 50:
            preprocessing.append({
                "col": col,
                "action": "drop",
                "reason": f"alta cardinalidad ({n_unique})",
            })
        elif n_unique <= 2:
            preprocessing.append({
                "col": col,
                "action": "encode",
                "method": "label",
            })
        elif n_unique <= 10:
            preprocessing.append({
                "col": col,
                "action": "encode",
                "method": "onehot",
            })
        else:
            preprocessing.append({
                "col": col,
                "action": "encode",
                "method": "target",
            })

    # Escalado: siempre standard para modelos basados en distancia
    needs_scale = "yes"
    # Si todos son categóricas encodeadas, puede que no necesite
    if not numeric_cols or len(numeric_cols) == 0:
        needs_scale = "none"
    elif all(c == target for c in numeric_cols):
        needs_scale = "none"


    # ── Transformación log1p para features con skew alto y no-negativas ──
    # Aplica ANTES del escalado para que la transformación normalize la
    # distribución y el escalado opere sobre datos más simétricos.
    for col in numeric_cols:
        if col == target:
            continue
        col_info = next((c for c in profile["columns"] if c["name"] == col), {})
        skew = abs(col_info.get("skew", 0) or 0)
        col_min = col_info.get("min")
        if skew > 2 and col_min is not None and col_min >= 0:
            if col not in [p["col"] for p in preprocessing if "drop" in p.values()]:
                preprocessing.append({
                    "col": col,
                    "action": "transform",
                    "method": "log1p",
                })
    if needs_scale != "none":
        for col in numeric_cols:
            if col == target:
                continue
            # Robust si hay outliers (skew alto)
            col_info = next((c for c in profile["columns"] if c["name"] == col), {})
            skew = abs(col_info.get("skew", 0) or 0)
            method = "robust" if skew > 2 else "standard"
            if col not in [p["col"] for p in preprocessing if "drop" in p.values()]:
                preprocessing.append({
                    "col": col,
                    "action": "scale",
                    "method": method,
                })

    # ── Auto-detección: regression forzada sobre target discreto con concentración ──
    # Si el target tiene ≤15 valores únicos (discreto) y concentración extrema,
    # conviene tratarlo como clasificación ordinal para aplicar balanceo real (SMOTE/ADASYN)
    # en vez de solo weighting. Caso típico: credit scoring, calificaciones con 95% en 0.
    n_unique = target_info.get("unique", 0)
    if task_type == "regression" and n_unique <= 15 and is_imbalanced:
        task_type = "classification"  # override interno
        target_weighting = {}  # no aplica weighting si vamos a clasificar

    # ── Estrategia de balanceo ──
    balance = {}
    if task_type == "classification" and is_imbalanced:
        level = target_info.get("balance_level", "moderate")

        # Elegir estrategia según el tamaño de la clase minoritaria
        # ADASYN/SMOTE requieren n_neighbors (default=5) samples mínimos
        dist = target_info.get("distribution", {})
        min_class_count = min(dist.values()) if dist else 0

        if min_class_count < 2:
            effective_strategy = "class_weight"  # Solo peso, sin oversampling
        elif min_class_count < 6:
            effective_strategy = "smote"  # SMOTE tolera menos vecinos
        else:
            strategy_map = {
                "extreme": "adasyn",
                "severe": "smote_enn",
                "moderate": "smote",
            }
            effective_strategy = strategy_map.get(level, "smote")

        balance = {
            "strategy": effective_strategy,
            "justification": (
                f"Ratio de balanceo {target_info.get('balance_ratio', 0.5)} "
                f"({level}, clase menor={min_class_count}). "
                f"Se aplica {effective_strategy} para balancear clases "
                f"+ class_weight en modelos que lo soportan."
            ),
            "models_with_weight": [
                "LogisticRegression",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "SVC",
            ],
        }

    # ── Pesado para regresión con target concentrado (>60% en un valor) ──
    target_weighting = {}
    if task_type == "regression":
        conc_level = target_info.get("concentration_level", "normal")
        if conc_level in ("severe", "extreme"):
            target_weighting = {
                "method": "inverse_frequency",
                "concentration_level": conc_level,
                "justification": (
                    f"Target concentrado: {target_info.get('majority_pct', 0)}% "
                    f"en {target_info.get('majority_value', '?')}. "
                    f"Se aplica peso inverso a frecuencia para que el modelo "
                    f"aprenda de los valores minoritarios."
                ),
            }

    # Selección de modelos según tipo de tarea
    catalog = TASK_MODEL_MAP.get(task_type, {})
    if not catalog:
        catalog = TASK_MODEL_MAP.get("classification", {})

    # Elegir modelos default del catálogo
    model_names = list(catalog.keys())[:4]
    models = []
    for name in model_names:
        model_entry = catalog[name]
        params = {}
        # Agregar class_weight si aplica
        if (
            task_type == "classification"
            and is_imbalanced
            and model_entry.get("supports_weight", False)
        ):
            params["class_weight"] = "balanced"
        models.append({"name": name, "params": params})

    # Métricas
    metrics = get_metrics_for_task(
        task_type, is_imbalanced=is_imbalanced
    )
    return {
        "target_weighting": target_weighting,
        "task_type": task_type,
        "target": target or "",
        "justification": (
            f"Plan generado automáticamente para {task_type}. "
            f"{len(profile.get('columns', []))} columnas, "
            f"{profile.get('n_rows', 0)} filas."
        ),
        "preprocessing": preprocessing,
        "balance": balance,
        "models": models,
        "feature_selection": config.feature_selection,
        "metrics": metrics,
    }


def _enhance_with_features(profile: dict, plan: dict, config: PipelineConfig) -> dict:
    """Usa el LLM para sugerir nuevas features derivadas y las agrega al plan."""
    available_cols = [c for c in profile.get("columns", [])
                      if c["name"] != plan.get("target")]

    # Solo columnas con nombre semántico (no IDs numéricos)
    candidates = []
    for c in available_cols:
        if c["cardinality_ratio"] > 0.9:
            continue  # probable ID
        if c["type"] in ("numeric", "categorical_numeric"):
            candidates.append(f"  • {c['name']} (numérico, min={c.get('min','?')}, max={c.get('max','?')}, media={c.get('mean','?')})")
        elif c["type"] == "categorical" and c.get("values"):
            vals = list(c["values"].keys())[:5]
            candidates.append(f"  • {c['name']} (categórica, {c['unique']} valores: {', '.join(str(v) for v in vals)}...)")
        elif c["type"] == "categorical":
            candidates.append(f"  • {c['name']} (categórica, {c['unique']} valores)")
        elif c["type"] == "datetime":
            candidates.append(f"  • {c['name']} (fecha, {c.get('min','?')} a {c.get('max','?')})")

    if len(candidates) < 2:
        return plan  # muy pocas columnas para feature engineering

    cols_text = "\n".join(candidates)
    task_type = plan.get("task_type", "classification")
    target_name = plan.get("target", "ninguno")

    prompt = f"""Eres un ingeniero de features experto. Tu tarea es sugerir nuevas columnas derivadas para mejorar un modelo de Machine Learning.

Columnas disponibles:
{cols_text}

Tipo de tarea: {task_type}
Target: {target_name}

Operaciones disponibles:
  ratio(col, grupo) — valor de col dividido por el promedio de su grupo (el grupo es una columna categórica)
  interact(a, b) — multiplicación a × b (dos columnas numéricas)
  str_len(col) — cantidad de caracteres de un campo de texto
  keyword(col, patrón) — 1 si el texto contiene el patrón, 0 si no
  date_diff(a, b) — días entre dos fechas
  bin(col, n) — discretizar col en n rangos

IMPORTANTE:
- NO uses bloques markdown ``` ni ```json
- NO agregues texto antes ni después del JSON
- Usa EXACTAMENTE este formato para cada sugerencia:
  {{"method": "ratio", "source": ["precio"], "params": {{"group_col": "categoria"}}, "col": "precio_relativo_categoria", "justification": "explicación"}}

Sugiere 2 o 3 transformaciones que aporten valor predictivo.
Responde ÚNICAMENTE con un JSON array válido."""

    try:
        from .llm_client import get_llm_response
        raw = get_llm_response(
            prompt,
            model=config.npc_model,
            provider=config.npc_provider,
            temperature=0.1,
            max_tokens=800,
        )
        if not raw or not raw.strip():
            raise ValueError("modelo no responde")
        # Extraer JSON tolerante
        import re
        clean = raw.strip()
        try:
            suggestions = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
            if m:
                clean = m.group(1).strip()
            else:
                start = clean.find("[")
                if start == -1:
                    start = clean.find("{")
                end = clean.rfind("]")
                if end == -1:
                    end = clean.rfind("}")
                if start != -1 and end > start:
                    clean = clean[start:end+1]
                else:
                    raise
            suggestions = json.loads(clean)
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
    except Exception as e:
        reason = str(e) if str(e) else type(e).__name__
        plan["justification"] += f" Feature engineering por IA no disponible ({reason})."
        return plan

    # Validar y agregar sugerencias
    col_names = {c["name"] for c in profile.get("columns", [])}
    valid_methods = {"ratio", "interact", "str_len", "keyword", "date_diff", "bin"}
    added = 0
    for s in suggestions:
        if added >= 3:
            break
        if not isinstance(s, dict):
            continue
        method = s.get("method")
        source = s.get("source", [])
        new_col = s.get("col", "")
        if method not in valid_methods or not new_col or new_col in col_names:
            continue
        if not all(src in col_names for src in source):
            continue
        # Normalizar params según método
        params = s.get("params", {}) or {}
        if method == "ratio":
            if not params.get("group_col") or params["group_col"] not in col_names:
                continue
        if method == "keyword" and not params.get("pattern"):
            continue
        if method in ("interact", "date_diff") and len(source) < 2:
            continue
        # Aceptar "n" como alias de "n_bins" para bin
        if method == "bin" and "n" in params and "n_bins" not in params:
            params["n_bins"] = params.pop("n")

        plan.setdefault("preprocessing", []).append({
            "col": new_col, "action": "transform", "method": method,
            "source": source, "params": params,
        })
        added += 1
    return plan
