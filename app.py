"""
app.py — Interfaz gráfica Streamlit para MELO.

Uso:
    streamlit run app.py --server.port 8501
"""

import os
import sys
import time
import json
from io import StringIO
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np


from src.config import PipelineConfig, TASK_DESCRIPTIONS
from src.profiler import profile_dataset, profile_summary_text
from src.strategist import build_plan
from src.pipeline import execute_pipeline
from src.trainer import train_models, ensemble_predictions
from src.reporter import generate_report
from cli import pre_parse, ensure_ollama_running

# ── Configuración de página ───────────────────────────────────────

st.set_page_config(
    page_title="MELO",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inicializar session state ─────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "results" not in st.session_state:
    st.session_state.results = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "ollama_checked" not in st.session_state:
    st.session_state.ollama_checked = False

# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### :material/analytics: MELO")
    st.markdown("**AutoML con LLM local**")
    st.divider()

    # ── Upload de archivo ─────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Cargar dataset",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help="Formatos soportados: CSV, Excel (.xlsx, .xls)",
    )

    if uploaded_file is not None:
        # Cargar datos
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Pre-parseo ETL
            df = pre_parse(df)

            # Solo resetear resultados si es un archivo NUEVO (diferente)
            same_file = (st.session_state.df_name == uploaded_file.name
                         and st.session_state.df is not None
                         and len(df) == len(st.session_state.df))
            if not same_file:
                st.session_state.results = None

            st.session_state.df = df
            st.session_state.df_name = uploaded_file.name
            st.success(f"✅ {uploaded_file.name} ({len(df)} filas)")
        except Exception as e:
            st.error(f"Error al cargar: {e}")
            st.session_state.df = None

    st.divider()

    # ── Configuración ─────────────────────────────────────────────
    if st.session_state.df is not None:
        st.markdown("### :material/tune: Configuración")

        df = st.session_state.df

        # Selección de target
        columnas = ["(No supervisado — clustering)"] + list(df.columns)
        target = st.selectbox(
            "Variable objetivo",
            options=columnas,
            index=0,
            help="Columna a predecir. Selecciona '(No supervisado)' para clustering.",
        )
        target_col = target if target != "(No supervisado — clustering)" else None

        # Override de tipo de tarea
        task_options = {
            "auto": "Auto-detectar",
            "classification": "Clasificación",
            "regression": "Regresión",
            "clustering": "Clustering",
            "anomaly_detection": "Detección de anomalías",
        }
        task_choice = st.selectbox(
            "Tipo de tarea",
            options=list(task_options.keys()),
            format_func=lambda x: task_options[x],
            index=0,
            help="Auto: el sistema detecta el tipo según los datos",
        )
        force_task = task_choice if task_choice != "auto" else None

        # ── Advertencia de concentración del target ──────────────
        if target_col and st.session_state.df is not None:
            _series = st.session_state.df[target_col]
            _vc = _series.value_counts()
            _top_val = _vc.index[0]
            _top_pct = round(_vc.iloc[0] / len(_series) * 100, 1)

            if _top_pct > 60:
                if _top_pct > 80:
                    _level = "extremo"
                    _icon = "🚨"
                else:
                    _level = "severo"
                    _icon = "⚠️"

                _is_sentinel = _top_val == 0 or str(_top_val) in ("0", "0.0", "-1", "NaN", "None", "")
                _hint = " Posiblemente representa 'sin dato'." if _is_sentinel else ""

                st.warning(
                    f"{_icon} **Concentración {_level} del target**  \n"
                    f"**{_top_pct}%** de los valores de `{target_col}` son **{_top_val}**."
                    f"{_hint}  \n"
                    f"El modelo puede aprender a predecir siempre {_top_val} "
                    f"y aún así obtener métricas aparentemente buenas."
                )

        # Opciones avanzadas
        with st.expander(":material/settings: Opciones avanzadas"):
            use_npc = st.toggle(
                "Insights con IA (NPC)",
                value=False,
                help="Usa Ollama para generar insights cualitativos. Desactivar es más rápido.",
            )

            # Selector de modelo Ollama (siempre visible)
            if "ollama_models" not in st.session_state:
                st.session_state.ollama_models = None
                st.session_state.ollama_error = None

            col_model, col_refresh = st.columns([4, 1])
            with col_refresh:
                if st.button("↻", help="Refrescar lista de modelos"):
                    st.session_state.ollama_models = None

            with col_model:
                if st.session_state.ollama_models is None:
                    try:
                        import json, urllib.request
                        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
                        all_models = [m["name"] for m in data.get("models", [])]
                        # Filtrar solo modelos de chat (excluir embedding, vision, y retirados)
                        st.session_state.ollama_models = [
                            m for m in all_models
                            if not any(kw in m.lower() for kw in ["embedding", "embed", "nomic"])
                        ]
                        st.session_state.ollama_error = None
                    except Exception as e:
                        st.session_state.ollama_models = []
                        st.session_state.ollama_error = str(e)

                if st.session_state.ollama_models:
                    npc_model = st.selectbox(
                        "Modelo Ollama",
                        options=st.session_state.ollama_models,
                        index=0,
                        disabled=not use_npc,
                        help="Modelo local para insights con IA.",
                    )
                else:
                    npc_model = "qwen2.5-coder:7b"
                    err = st.session_state.ollama_error
                    if use_npc:
                        msg = f"⚠️ {err}" if err else "⚠️ No hay modelos. Ejecuta: ollama pull qwen2.5-coder:7b"
                        st.caption(msg)

            test_size = st.slider(
                "Tamaño del test",
                min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                format="%.0f%%",
            )

            cv_folds = st.selectbox(
                "Folds de validación cruzada",
                options=[3, 5, 10], index=1,
            )

        # ── Botón de ejecución ────────────────────────────────────
        run_btn = st.button(
            ":material/play_arrow: Ejecutar pipeline",
            type="primary",
            width="stretch",
            disabled=st.session_state.pipeline_running,
        )

        if run_btn:
            st.session_state.pipeline_running = True

            # Verificar Ollama si se necesita
            if use_npc and not st.session_state.ollama_checked:
                if not ensure_ollama_running():
                    st.warning("Ollama no disponible. Insights NPC desactivados.")
                    use_npc = False
                st.session_state.ollama_checked = True

            config = PipelineConfig(
                test_size=test_size,
                cv_folds=cv_folds,
                npc_provider="ollama",
                npc_model=npc_model,
            )

            progress_bar = st.progress(0, text="Iniciando...")

            try:
                # 1. Perfilado
                progress_bar.progress(10, text="🔍  Perfilando dataset...")
                profile = profile_dataset(df, target_col)

                # 2. Estrategia
                progress_bar.progress(20, text="🧠  Generando estrategia...")
                plan = build_plan(profile, target=target_col, force_task=force_task, config=config)

                # 3. Pipeline
                progress_bar.progress(35, text="⚙️  Ejecutando pipeline ETL...")
                pipeline_result = execute_pipeline(df, plan, target=target_col, config=config)

                # 4. Entrenamiento
                progress_bar.progress(50, text="🏋️  Entrenando modelos...")
                train_results = train_models(
                    pipeline_result["X_train"],
                    pipeline_result["y_train"],
                    plan,
                    config=config,
                    verbose=False,
                    X_test=pipeline_result["X_test"],
                    y_test=pipeline_result["y_test"],
                )

                # 5. Ensemble
                progress_bar.progress(75, text="🧬  Evaluando ensemble...")
                ensemble_result = ensemble_predictions(
                    pipeline_result["X_test"],
                    pipeline_result["y_test"],
                    train_results,
                    plan,
                    config=config,
                    verbose=False,
                )

                # 6a. Predicciones
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

                # 6. Reporte
                progress_bar.progress(90, text="📋  Generando reporte...")
                report = generate_report(
                    profile, plan, pipeline_result,
                    train_results, ensemble_result,
                    config=config, use_npc=use_npc,
                )

                progress_bar.progress(100, text="✅  Completado!")
                time.sleep(0.5)
                progress_bar.empty()

                # Guardar resultados
                st.session_state.profile = profile
                st.session_state.results = {
                    "profile": profile,
                    "plan": plan,
                    "pipeline_result": pipeline_result,
                    "train_results": train_results,
                    "ensemble_result": ensemble_result,
                    "report": report,
                    "target": target_col,
                    "scored": scored,
                }
                st.session_state.pipeline_running = False
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                st.error(f"Error en el pipeline: {type(e).__name__}: {e}")
                st.session_state.pipeline_running = False

# ── Main area ─────────────────────────────────────────────────────

# Título
st.title(":material/analytics: MELO")
st.caption("Pipeline inteligente de Machine Learning con ETL automático")

# ── Estado inicial (sin datos) ────────────────────────────────────
if st.session_state.df is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### :material/upload_file: Carga")
        st.markdown("Sube un archivo CSV o Excel desde el panel lateral")
    with col2:
        st.markdown("### :material/auto_awesome: Automático")
        st.markdown("ETL, perfilado, y estrategia adaptativa")
    with col3:
        st.markdown("### :material/description: Reporte")
        st.markdown("Resultados claros con insights de IA")

    st.divider()

    # Datos de ejemplo rápidos
    st.markdown("### :material/science: Probar con datos de ejemplo")
    col_a, col_b, col_c = st.columns(3)
    if col_a.button(":material/group: Churn (clasificación)", width="stretch"):
        from sample_data import make_realistic_churn
        df = make_realistic_churn()
        df = pre_parse(df)
        st.session_state.df = df
        st.session_state.df_name = "churn_sintético.csv"
        st.rerun()
    if col_b.button(":material/real_estate_agent: Viviendas (regresión)", width="stretch"):
        from sample_data import make_house_prices_dataset
        df = make_house_prices_dataset()
        df = pre_parse(df)
        st.session_state.df = df
        st.session_state.df_name = "viviendas_sintético.csv"
        st.rerun()
    if col_c.button(":material/dataset: Segmentación (clustering)", width="stretch"):
        from sample_data import make_segmentation_dataset
        df = make_segmentation_dataset()
        df = pre_parse(df)
        st.session_state.df = df
        st.session_state.df_name = "segmentación_sintético.csv"
        st.rerun()

    st.stop()

# ── Datos cargados ────────────────────────────────────────────────
df = st.session_state.df

# Tabs de navegación
tab_preview, tab_profile, tab_results = st.tabs([
    ":material/table: Vista previa",
    ":material/bar_chart: Perfil de datos",
    ":material/analytics: Resultados",
])

# ── TAB 1: Vista previa ──────────────────────────────────────────
with tab_preview:
    st.markdown(f"### :material/table: {st.session_state.df_name}")
    st.caption(f"{len(df)} filas × {len(df.columns)} columnas")

    col_info = st.columns(3)
    col_info[0].metric("Filas", f"{len(df):,}")
    col_info[1].metric("Columnas", len(df.columns))
    col_info[2].metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    st.dataframe(
        df.head(100),
        width="stretch",
        height=400,
    )

    with st.expander(":material/info: Tipos de datos"):
        dtypes_df = pd.DataFrame({
            "Columna": df.dtypes.index,
            "Tipo": df.dtypes.values.astype(str),
            "No nulos": df.count().values,
            "Nulos": df.isna().sum().values,
            "% Nulos": (df.isna().sum() / len(df) * 100).round(1).astype(str) + "%",
            "Valores únicos": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtypes_df, width="stretch", hide_index=True)

# ── TAB 2: Perfil de datos ───────────────────────────────────────
with tab_profile:
    profile = profile_dataset(df, None)  # Perfil sin target específico

    st.markdown("### :material/bar_chart: Estadísticas generales")

    met_cols = st.columns(4)
    met_cols[0].metric("Filas", f"{profile['n_rows']:,}")
    met_cols[1].metric("Columnas", profile["n_columns"])
    met_cols[2].metric("Numéricas", len(profile["numeric_cols"]))
    met_cols[3].metric("Categóricas", len(profile["categorical_cols"]))

    # Perfil de cada columna
    st.markdown("### :material/table_rows: Detalle por columna")
    col_data = []
    for c in profile["columns"]:
        col_data.append({
            "Columna": c["name"],
            "Tipo": c.get("type", "?"),
            "Dtype": c["dtype"],
            "Nulos": f'{c["null_pct"]}%',
            "Únicos": c["unique"],
            "Min": f'{c["min"]:.4g}' if c.get("min") is not None else "-",
            "Max": f'{c["max"]:.4g}' if c.get("max") is not None else "-",
            "Media": f'{c["mean"]:.4g}' if c.get("mean") is not None else "-",
        })
    st.dataframe(pd.DataFrame(col_data), width="stretch", hide_index=True)

    # Correlaciones altas
    if profile["correlations"]["high_pairs"]:
        with st.expander(f":material/link: Correlaciones altas ({len(profile['correlations']['high_pairs'])})"):
            corr_df = pd.DataFrame(profile["correlations"]["high_pairs"])
            st.dataframe(corr_df, width="stretch", hide_index=True)

# ── TAB 3: Resultados ────────────────────────────────────────────
with tab_results:
    if st.session_state.results is None:
        st.info(":material/info: Configura el análisis en el panel lateral y presiona **Ejecutar pipeline**")
        st.stop()

    results = st.session_state.results
    plan = results["plan"]
    train_results = results["train_results"]
    ensemble_result = results["ensemble_result"]
    pipeline_result = results["pipeline_result"]
    report = results["report"]

    base_name = os.path.splitext(st.session_state.df_name)[0] if st.session_state.df_name else "dataset"
    # ── Resumen rápido ────────────────────────────────────────────
    st.markdown("### :material/summarize: Resumen del análisis")

    ranked = train_results.get("ranked_models", [])
    top = ranked[0] if ranked else None

    res_cols = st.columns(4)
    res_cols[0].metric("Tarea", plan.get("task_type", "?").replace("_", " ").title())
    res_cols[1].metric("Modelos", len(ranked))
    res_cols[2].metric(
        "Mejor modelo",
        top["name"] if top else "-",
    )
    res_cols[3].metric(
        "Mejor CV",
        f"{top['cv_mean']:.3f} ±{top['cv_std']:.3f}" if top else "-",
    )

    # ── Preprocesamiento ──────────────────────────────────────────
    with st.expander(":material/build: Preprocesamiento aplicado", expanded=False):
        pp = pipeline_result.get("preprocessor", {})

        pp_info = []
        if pp.get("etl", {}).get("parsed_dates"):
            pp_info.append(f"📅 Fechas parseadas: {pp['etl']['parsed_dates']}")
        if pp.get("etl", {}).get("parsed_currency"):
            pp_info.append(f"💰 Monedas parseadas: {pp['etl']['parsed_currency']}")
        if pp.get("outliers_capped"):
            pp_info.append(f"📊 Outliers capped: {pp['outliers_capped']}")
        if pp.get("dropped_cols"):
            pp_info.append(f"🗑️ Columnas eliminadas: {pp['dropped_cols']}")
        if pp.get("constant_dropped"):
            pp_info.append(f"📌 Columnas constantes: {pp['constant_dropped']}")
        if pp.get("imputed_cols"):
            pp_info.append(f"🩹 Imputaciones: {len(pp['imputed_cols'])} columnas")
        if pp.get("encoded_cols"):
            pp_info.append(f"🔤 Codificaciones: {list(pp['encoded_cols'].keys())}")
        if pp.get("scaled_cols"):
            pp_info.append(f"📐 Escalados: {len(pp['scaled_cols'])} columnas")
        if pp.get("balance_applied"):
            b = pp["balance_applied"]
            if not b.get("error"):
                pp_info.append(f"⚖️ Balanceo: {b['strategy']} ({b.get('original_size','?')} → {b.get('balanced_size','?')})")
            else:
                pp_info.append(f"⚠️ Balanceo no aplicado")
        # Features ingenierizadas
        transformed = pp.get("transformed_cols", {})
        fe_cols = {k: v for k, v in transformed.items() if v.get("method") not in (None, "log1p")}
        if fe_cols:
            fe_list = [f"**{k}** = {v.get('method','?')}" for k, v in fe_cols.items()]
            pp_info.append("🧬 Features ingenierizadas por IA: " + ", ".join(fe_list))

        for item in pp_info:
            st.markdown(f"- {item}")

    # ── Ranking de modelos ────────────────────────────────────────
    st.markdown("### :material/emoji_events: Ranking de modelos")

    if ranked:
        rank_cols = st.columns(min(4, len(ranked)))
        for i, (col, model) in enumerate(zip(rank_cols, ranked)):
            medals = [":material/emoji_events:", ":material/workspace_premium:", ":material/military_tech:"]
            with col:
                st.markdown(
                    f"### {medals[i] if i < 3 else f'{i+1}.'}"
                )
                st.markdown(f"**{model['name']}**")
                st.metric(
                    "CV Score",
                    f"{model['cv_mean']:.4f}",
                    delta=f"±{model['cv_std']:.4f}",
                )
                if model.get("best_params"):
                    params_str = ", ".join(
                        f"{k}={v}" for k, v in model["best_params"].items()
                        if k not in ("random_state", "n_init", "probability", "max_iter")
                    )
                    st.caption(f"Params: {params_str[:80]}...")
                st.caption(f"⏱️ {model.get('time_seconds', 0):.1f}s")

    # ── Importancia de features ─────────────────────────────────
    top = ranked[0] if ranked else None
    feat_names = train_results.get("feature_names", [])
    if top and top.get("importances") and feat_names:
        st.markdown("### :material/bar_chart: Importancia de features")
        st.caption(f"Del mejor modelo: **{top['name']}**")
        imp_df = pd.DataFrame({"feature": feat_names, "importancia": top["importances"]})
        imp_df = imp_df.sort_values("importancia", ascending=False)
        max_imp = imp_df["importancia"].max()
        for _, row in imp_df.iterrows():
            bar = "█" * max(1, int(row["importancia"] / max_imp * 25))
            st.markdown(f"`{row['importancia']:.4f}` {bar} **{row['feature']}**")

    # ── Ensemble ──────────────────────────────────────────────────
    if ensemble_result.get("scores"):
        st.markdown("### :material/neurology: Ensemble")
        st.markdown(f"**Método:** {ensemble_result.get('method', 'N/A')}")

        score_cols = st.columns(len(ensemble_result["scores"]))
        for col, (metric, value) in zip(score_cols, ensemble_result["scores"].items()):
            col.metric(metric, f"{value:.4f}")

    # ── Predicciones ──────────────────────────────────────────────
    scored = results.get("scored")
    if scored is not None and len(scored) > 0:
        st.markdown("### :material/target: Predicciones (test set)")

        display_df = scored.head(20).copy()
        display_df = display_df.rename(columns={
            "actual": "Valor real",
            "prediccion": "Predicción",
        })
        if "correcto" in display_df.columns:
            display_df["Estado"] = display_df["correcto"].map({True: "✅", False: "❌"})
            display_df = display_df.drop(columns=["correcto"])
        if "error_abs" in display_df.columns:
            display_df["Error"] = display_df["error_abs"].round(4)
            display_df = display_df.drop(columns=["error_abs"])

        st.dataframe(display_df, width="stretch", hide_index=True)

        total = len(scored)
        if "correcto" in scored.columns:
            aciertos = scored["correcto"].sum()
            st.caption(
                f"Mostrando 20 de {total} registros del test set "
                f"| ✅ {aciertos}/{total} aciertos ({aciertos/total*100:.1f}%)"
            )
        else:
            st.caption(f"Mostrando 20 de {total} registros del test set")

        csv = scored.to_csv(index_label="fila").encode("utf-8")
        st.download_button(
            label=":material/download: Descargar predicciones (CSV)",
            data=csv,
            file_name=f"{base_name}_predicciones.csv",
            mime="text/csv",
        )

    # ── Reporte completo ──────────────────────────────────────────
    with st.expander(":material/description: Reporte completo", expanded=True):
        st.markdown(f"```\n{report}\n```")

    # ── Descargar reporte ─────────────────────────────────────────
    report_bytes = report.encode("utf-8")
    st.download_button(
        label=":material/download: Descargar reporte (.txt)",
        data=report_bytes,
        file_name=f"{base_name}_reporte.txt",
        mime="text/plain",
        width="stretch",
    )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("MELO  —  Construido con Python + scikit-learn + Ollama")
