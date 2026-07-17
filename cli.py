"""
cli.py — Entry point unificado del pipeline MELO.

Uso:
    python cli.py datos.csv
    python cli.py datos.csv --target churn
    python cli.py datos.csv --target precio --task regression
    python cli.py datos.csv --task clustering
    python cli.py datos.csv --target churn --no-npc
    python cli.py datos.xlsx --target ventas
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Optional

import numpy as np
import pandas as pd


from src.config import PipelineConfig, TASK_DESCRIPTIONS
from src.profiler import profile_dataset, profile_summary_text
from src.strategist import build_plan
from src.pipeline import execute_pipeline
from src.trainer import train_models, ensemble_predictions
from src.reporter import generate_report


def ensure_ollama_running(timeout: int = 15) -> bool:
    """Verifica si Ollama está corriendo y lo inicia si es necesario."""
    import shutil

    # Primero verificar si ya está disponible
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        pass

    # Buscar ollama en PATH (instalación estándar) + rutas comunes
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        for p in [
            os.path.expanduser("~\\AppData\\Local\\Ollama\\ollama.exe"),
            "C:\\Program Files\\Ollama\\ollama.exe",
        ]:
            if os.path.isfile(p):
                ollama_path = p
                break

    if ollama_path:
        try:
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for _ in range(timeout):
                try:
                    urllib.request.urlopen(
                        "http://localhost:11434/api/tags", timeout=2
                    )
                    return True
                except Exception:
                    time.sleep(1)
        except FileNotFoundError:
            pass

    return False


def pre_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-parseo ligero que limpia valores problemáticos ANTES del perfilado.
    Se ejecuta antes de profile_dataset() para que los stats sean precisos.
    """
    df = df.copy()
    for col in df.columns:
        # Infinitos → NaN
        if pd.api.types.is_numeric_dtype(df[col]):
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Moneda: $1,234.56 → numeric
        if df[col].dtype == "object":
            sample = df[col].dropna().iloc[:50]
            if len(sample) > 0:
                has_currency = sample.astype(str).str.contains(r"^\$", na=False).any()
                if has_currency:
                    cleaned = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
                    df[col] = pd.to_numeric(cleaned, errors="coerce")

        # Porcentaje: 45.2% → 0.452
        if df[col].dtype == "object":
            sample = df[col].dropna().iloc[:50]
            if len(sample) > 0:
                has_pct = sample.astype(str).str.contains(r"%$", na=False).any()
                if has_pct:
                    cleaned = df[col].astype(str).str.replace("%", "", regex=False)
                    df[col] = pd.to_numeric(cleaned, errors="coerce") / 100.0

        # Negativos en columnas que parecen positivas: convertir a NaN
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            if min_val is not None and not pd.isna(min_val) and min_val < 0:
                neg_pct = (df[col] < 0).sum() / len(df)
                if neg_pct < 0.05:
                    df.loc[df[col] < 0, col] = np.nan

    return df


def load_data(file_path: str) -> pd.DataFrame:
    """Carga CSV o Excel."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa CSV o Excel.")


def run_pipeline(
    df: pd.DataFrame,
    target: Optional[str] = None,
    force_task: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    use_npc: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo.

    Returns
    -------
    dict con todos los resultados intermedios (para depuración o API).
    """
    cfg = config or PipelineConfig()
    t_start = time.time()

    dataset_name = getattr(df, "name", None)  # preservar antes del copy() en pre_parse
    # ── 0. Pre-parseo ETL (ANTES del perfilado) ───────────────────
    if verbose:
        print("🧹  PRE-PARSEANDO DATOS (monedas, %, infinitos, negativos)...")

    df = pre_parse(df)

    # ── 1. Perfilado ───────────────────────────────────────────────
    if verbose:
        print("\n🔍  PERFILANDO DATASET...")
        print("-" * 50)

    profile = profile_dataset(df, target)

    if verbose:
        print(profile_summary_text(profile))
        print()

    # ── 2. Estrategia ──────────────────────────────────────────────
    if verbose:
        print("🧠  GENERANDO ESTRATEGIA...")
        print("-" * 50)

    plan = build_plan(
        profile,
        target=target,
        force_task=force_task,
        config=cfg,
    )

    if verbose:
        print(f"  Tarea detectada: {plan['task_type']}")
        for a in plan.get("preprocessing", []):
            print(f"  • {a['col']}: {a['action']} ({a.get('method', '')})")
        if plan.get("balance", {}).get("strategy") not in (None, "none", ""):
            print(f"  ⚖️  Balanceo: {plan['balance']['strategy']}")
        print(f"  Modelos: {[m['name'] for m in plan.get('models', [])]}")
        print()

    # ── 3. Pipeline (preprocesamiento) ─────────────────────────────
    if verbose:
        print("⚙️   EJECUTANDO PIPELINE...")
        print("-" * 50)

    pipeline_result = execute_pipeline(df, plan, target, config=cfg)

    X_train = pipeline_result["X_train"]
    y_train = pipeline_result["y_train"]
    X_test = pipeline_result["X_test"]
    y_test = pipeline_result["y_test"]

    if verbose:
        print(
            f"  Train: {len(X_train)} filas x {len(X_train.columns)} cols"
        )
        if X_test is not None:
            print(f"  Test:  {len(X_test)} filas x {len(X_test.columns)} cols")
        if y_train is not None:
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"  Target train: {dict(zip(unique.tolist(), counts.tolist()))}")
        print()

    # ── 4. Entrenamiento ───────────────────────────────────────────
    if y_train is not None and plan.get("task_type") not in (
        "clustering", "dimensionality_reduction", "anomaly_detection"
    ):

        if verbose:
            print("🏋️   ENTRENANDO MODELOS...")
            print("-" * 50)

        train_results = train_models(
            X_train, y_train, plan,
            config=cfg, verbose=verbose,
            X_test=X_test, y_test=y_test,
        )

        print()

        # ── 5. Ensemble ────────────────────────────────────────────────
        if X_test is not None and y_test is not None:
            if verbose:
                print("🧬  EVALUANDO ENSEMBLE...")
                print("-" * 50)

            ensemble_result = ensemble_predictions(
                X_test, y_test, train_results, plan,
                config=cfg, verbose=verbose,
            )
            print()
        else:
            ensemble_result = {
                "predictions": None,
                "scores": {},
                "models_used": [],
                "method": "none",
            }
    else:
        # No supervisado
        train_results = {"ranked_models": [], "top_model": None, "best_score": 0}
        ensemble_result = {
            "predictions": None,
            "scores": {},
            "models_used": [],
            "method": "none",
        }
        if verbose:
            print(f"  📌 Modo no supervisado ({plan['task_type']})")
            print()

    # ── 6. Predicciones ─────────────────────────────────────────────
    scored = None
    if ensemble_result.get("predictions") is not None and pipeline_result.get("y_test") is not None:
        y_test = pipeline_result["y_test"]
        y_pred = ensemble_result["predictions"]
        target_encoder = pipeline_result.get("target_encoder")

        actual = target_encoder.inverse_transform(y_test.astype(int)) if target_encoder else y_test
        preds = target_encoder.inverse_transform(y_pred.astype(int)) if target_encoder else y_pred

        scored = pd.DataFrame({"actual": actual, "prediccion": preds})
        if plan.get("task_type") == "classification":
            scored["correcto"] = scored["actual"] == scored["prediccion"]
        else:
            scored["error_abs"] = abs(scored["actual"] - scored["prediccion"])

        if verbose:
            aciertos = scored["correcto"].sum() if "correcto" in scored else "-"
            print(f"  ✅ Predicciones: {len(scored)} registros, {aciertos} aciertos")
            print()

    # ── 7. Reporte ─────────────────────────────────────────────────
    if verbose:
        print("📋  GENERANDO REPORTE...")
        print("-" * 50)

    report = generate_report(
        profile, plan, pipeline_result,
        train_results, ensemble_result,
        config=cfg, use_npc=use_npc,
    )

    elapsed = time.time() - t_start

    if verbose:
        print(report)
        print(f"\n⏱️  Tiempo total: {elapsed:.1f}s")
        print()

    # Guardar reporte y predicciones
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(
        os.path.basename(dataset_name)
    )[0] if dataset_name else "dataset"

    if cfg.save_report:
        report_path = os.path.join(output_dir, f"{base_name}_reporte.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        if verbose:
            print(f"  📄 Reporte guardado: {report_path}")

    if cfg.save_report and scored is not None:
        pred_path = os.path.join(output_dir, f"{base_name}_predicciones.csv")
        scored.to_csv(pred_path, index_label="fila")
        if verbose:
            print(f"  📄 Predicciones guardadas: {pred_path}")

    return {
        "profile": profile,
        "plan": plan,
        "pipeline_result": pipeline_result,
        "train_results": train_results,
        "ensemble_result": ensemble_result,
        "scored": scored,
        "report": report,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MELO — Pipeline completo de ML con ETL inteligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python cli.py datos.csv                     # Sin target → clustering
  python cli.py datos.csv --target churn       # Clasificación
  python cli.py datos.csv --target precio      # Regresión
  python cli.py datos.csv --task classification # Forzar tipo
  python cli.py datos.csv --no-npc             # Sin insights LLM
  python cli.py datos.csv --target churn --output ./reportes
        """,
    )
    parser.add_argument("file", help="Ruta al archivo CSV o Excel")
    parser.add_argument("--target", "-t", default=None,
                        help="Nombre de la columna objetivo")
    parser.add_argument("--task", default=None,
                        choices=["classification", "regression",
                                 "clustering", "anomaly_detection"],
                        help="Forzar tipo de tarea")
    parser.add_argument("--no-npc", action="store_true",
                        help="Desactivar insights del NPC")
    parser.add_argument("--output", "-o", default="output",
                        help="Directorio de salida (default: output/)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Modo silencioso (solo reporte)")

    args = parser.parse_args()

    # Mostrar tipos de tarea disponibles si solo pide ayuda de task
    if args.task and args.task not in (
        "classification", "regression", "clustering", "anomaly_detection"
    ):
        print("Tipos de tarea disponibles:")
        for k, v in TASK_DESCRIPTIONS.items():
            print(f"  • {k}: {v}")
        return

    # Validar archivo
    if not os.path.exists(args.file):
        print(f"❌ Archivo no encontrado: {args.file}")
        sys.exit(1)

    # Auto-iniciar Ollama si se necesita NPC
    if not args.no_npc and not args.quiet:
        print("🔄  Verificando Ollama...", end=" ", flush=True)
        if ensure_ollama_running():
            print("✅")
        else:
            print("⚠️  No disponible — insights NPC desactivados")
            args.no_npc = True

    # Config
    config = PipelineConfig(output_dir=args.output)

    # Cargar datos
    if not args.quiet:
        print(f"\n📂  Cargando: {args.file}")
        print(f"  Target: {args.target or '(no supervisado)'}")
        print(f"  Tarea:  {args.task or '(auto-detectar)'}")
        if args.no_npc:
            print(f"  NPC:    desactivado")

    df = load_data(args.file)
    df.name = args.file  # para nombre de archivo de reporte

    run_pipeline(
        df,
        target=args.target,
        force_task=args.task,
        config=config,
        use_npc=not args.no_npc,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
