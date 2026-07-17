"""
reporter.py — NPC reportero que genera un reporte ejecutivo en español
con los resultados del pipeline AutoML.

Usa Ollama (o cualquier LLM configurado) para dar insights cualitativos.
"""

import json
from typing import Optional

from .config import PipelineConfig
from .llm_client import get_llm_response


def generate_report(
    profile: dict,
    plan: dict,
    pipeline_result: dict,
    train_results: dict,
    ensemble_result: dict,
    config: Optional[PipelineConfig] = None,
    use_npc: bool = True,
) -> str:
    """
    Genera el reporte ejecutivo completo.

    Parameters
    ----------
    profile : dict
        Perfil del dataset.
    plan : dict
        Plan de preprocesamiento.
    pipeline_result : dict
        Resultado del pipeline (preprocessor metadata, shapes).
    train_results : dict
        Resultados de train_models().
    ensemble_result : dict
        Resultados de ensemble_predictions().
    config : PipelineConfig or None
    use_npc : bool
        Si True, usa LLM para generar insights cualitativos.

    Returns
    -------
    str con el reporte formateado.
    """
    report_parts = []

    # ── Encabezado ──────────────────────────────────────────────────
    report_parts.append("=" * 60)
    report_parts.append("📋  REPORTE DE ANÁLISIS — MELO")
    report_parts.append("=" * 60)
    report_parts.append("")

    # ── Resumen del dataset ─────────────────────────────────────────
    report_parts.append("📁  DATOS")
    report_parts.append("-" * 40)
    report_parts.append(f"  Filas: {profile['n_rows']:,}")
    report_parts.append(f"  Columnas: {profile['n_columns']}")
    report_parts.append(f"  Memoria: {profile['memory_mb']:.1f} MB")

    target_info = profile.get("target")
    if target_info:
        report_parts.append(f"  Target: {target_info['name']}")
        report_parts.append(f"  Tipo: {target_info['task_type']}")
        if target_info.get("unique"):
            report_parts.append(f"  Clases: {target_info['unique']}")
    report_parts.append("")

    # ── Plan ejecutado ──────────────────────────────────────────────
    report_parts.append("🔧  PREPROCESAMIENTO")
    report_parts.append("-" * 40)
    report_parts.append(f"  Tarea: {plan.get('task_type', 'auto-detectada')}")
    report_parts.append(f"  Justificación: {plan.get('justification', '')}")

    pp = pipeline_result.get("preprocessor", {})
    if pp.get("dropped_cols"):
        report_parts.append(f"  Columnas eliminadas: {pp['dropped_cols']}")
    if pp.get("imputed_cols"):
        report_parts.append(
            f"  Imputaciones: {len(pp['imputed_cols'])} columnas"
        )
    if pp.get("encoded_cols"):
        report_parts.append(
            f"  Codificaciones: {len(pp['encoded_cols'])} columnas"
        )
    if pp.get("scaled_cols"):
        report_parts.append(
            f"  Escalados: {len(pp['scaled_cols'])} columnas"
        )

    balance = plan.get("balance", {})
    if balance and balance.get("strategy") not in ("none",):
        report_parts.append(
            f"  Balanceo: {balance['strategy']} — {balance.get('justification', '')}"
        )
    if pp.get("balance_applied"):
        b = pp["balance_applied"]
        if not b.get("error"):
            report_parts.append(
                f"  → Tamaño original: {b.get('original_size', '?')} → "
                f"balanceado: {b.get('balanced_size', '?')}"
            )
        else:
            report_parts.append(
                f"  ⚠️  Balanceo no aplicado: instala imbalanced-learn"
            )

    report_parts.append("")

    # ── Resultados de modelos ──────────────────────────────────────
    report_parts.append("🏆  MODELOS ENTRENADOS")
    report_parts.append("-" * 40)

    ranked = train_results.get("ranked_models", [])
    if not ranked:
        report_parts.append("  ❌ No se pudieron entrenar modelos.")
    else:
        for i, r in enumerate(ranked):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
            report_parts.append(
                f"  {medal} {r['name']}: "
                f"CV={r['cv_mean']:.4f} ±{r['cv_std']:.4f} "
                f"({r['time_seconds']:.1f}s)"
            )
            if r.get("best_params"):
                params_str = ", ".join(
                    f"{k}={v}" for k, v in r["best_params"].items()
                    if k not in ("random_state", "n_init", "max_iter")
                )
                report_parts.append(f"     Params: {params_str}")

    report_parts.append("")

    # ── Ensemble ────────────────────────────────────────────────────
    if ensemble_result and ensemble_result.get("predictions") is not None:
        report_parts.append("🧬  ENSEMBLE")
        report_parts.append("-" * 40)
        report_parts.append(
            f"  Método: {ensemble_result.get('method', 'N/A')}"
        )
        report_parts.append(
            f"  Modelos: {ensemble_result.get('models_used', [])}"
        )
        scores = ensemble_result.get("scores", {})
        if scores:
            report_parts.append("  Métricas en test:")
            for metric, value in scores.items():
                report_parts.append(f"    • {metric}: {value}")
        report_parts.append("")

    # ── Comparativa vs individual ──────────────────────────────────
    if ensemble_result.get("scores") and ranked:
        best_individual = ranked[0] if ranked else None
        best_test_score = ensemble_result["scores"].get("accuracy") or \
                          ensemble_result["scores"].get("r2")
        if best_individual and best_test_score:
            improvement = best_test_score - best_individual["cv_mean"]
            pct = (improvement / best_individual["cv_mean"] * 100) if best_individual["cv_mean"] else 0
            report_parts.append(f"📊  MEJORA DEL ENSEMBLE")
            report_parts.append("-" * 40)
            report_parts.append(
                f"  Mejor individual: {best_individual['name']} "
                f"({best_individual['cv_mean']:.4f})"
            )
            report_parts.append(
                f"  Ensemble: {best_test_score:.4f} "
                f"({'+' if improvement >= 0 else ''}{improvement:.4f}, "
                f"{'+' if pct >=0 else ''}{pct:.1f}%)"
            )
            report_parts.append("")

    # ── Insights del NPC ───────────────────────────────────────────
    if use_npc:
        npc_insight = _get_npc_insight(
            profile, plan, train_results, ensemble_result, config
        )
        report_parts.append("🤖  INSIGHTS DEL AGENTE")
        report_parts.append("-" * 40)
        report_parts.append("")
        report_parts.append(npc_insight)
        report_parts.append("")

    # ── Recomendaciones ─────────────────────────────────────────────
    if target_info:
        report_parts.append("💡  RECOMENDACIONES")
        report_parts.append("-" * 40)
        if target_info.get("balance_level") in ("severe", "extreme"):
            report_parts.append(
                "  ⚠️  El dataset tiene desbalanceo severo. "
                "Considera recolectar más datos de la clase minoritaria."
            )
        report_parts.append(
            "  • Revisa las features más importantes del mejor modelo"
        )
        report_parts.append(
            "  • Valida los resultados con datos nuevos (out-of-time)"
        )
        report_parts.append("")

    report_parts.append("=" * 60)
    report_parts.append("  Reporte generado por MELO")
    report_parts.append("=" * 60)

    return "\n".join(report_parts)


def _get_npc_insight(
    profile: dict,
    plan: dict,
    train_results: dict,
    ensemble_result: dict,
    config: Optional[PipelineConfig] = None,
) -> str:
    """
    Genera un insight cualitativo usando LLM local (Ollama)
    o un resumen sintético si no hay LLM disponible.
    """
    cfg = config or PipelineConfig()

    # Construir resumen de resultados para el prompt
    ranked = train_results.get("ranked_models", [])
    top = ranked[0] if ranked else None

    results_summary = (
        f"Dataset: {profile['n_rows']} filas, {profile['n_columns']} columnas\n"
        f"Tarea: {plan.get('task_type', 'N/A')}\n"
        f"Mejor modelo: {top['name'] if top else 'N/A'} "
        f"(CV={top['cv_mean']:.4f})\n"
        f"Modelos entrenados: {len(ranked)}\n"
    )
    if ensemble_result.get("scores"):
        results_summary += (
            f"Ensemble scores: {json.dumps(ensemble_result['scores'])}\n"
        )

    try:
        prompt = f"""Eres un científico de datos senior y consultor de negocio en MELO.
Acabas de completar un análisis automático de datos y debes escribir un párrafo 
de insight ejecutivo en español.

Resumen del análisis:
{results_summary}

Escribe 2-3 párrafos breves en español explicando:
1. Qué revelan los resultados (en lenguaje de negocio, no técnico)
2. Qué variables o patrones parecen más importantes
3. Recomendación accionable para el negocio

Sé conciso, profesional y evita jerga técnica excesiva.
No uses markdown. Responde en español.
"""

        response = get_llm_response(
            prompt,
            model=cfg.npc_model,
            provider=cfg.npc_provider,
            temperature=cfg.npc_temperature,
            max_tokens=500,
        )
        if response and response.strip():
            return response.strip()
        raise ValueError("Respuesta vacía del modelo")

    except Exception as e:
        # Fallback sintético
        fallback_lines = [
            f"Análisis completado con {len(ranked)} modelos.",
        ]
        if top:
            fallback_lines.append(
                f"El mejor modelo fue {top['name']} con un score "
                f"de validación cruzada de {top['cv_mean']:.3f}."
            )
        if ensemble_result.get("scores"):
            fallback_lines.append(
                "El ensemble combinó los mejores modelos "
                f"logrando {' y '.join(f'{k}={v}' for k, v in ensemble_result['scores'].items())}."
            )
        target_info = profile.get("target")
        if target_info and target_info.get("balance_level") not in ("balanced", "mild"):
            fallback_lines.append(
                "Se detectó desbalanceo en la variable objetivo, "
                "por lo que las métricas por clase (precision/recall) "
                "son más relevantes que la accuracy global."
            )
        return "\n".join(fallback_lines)
