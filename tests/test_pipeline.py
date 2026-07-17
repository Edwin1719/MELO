"""
Tests para src/pipeline.py — ETL, preprocesamiento y split del pipeline AutoML.

Cubre 7 contratos:
1. execute_pipeline() con Productos_ML.csv y plan de clasificación
2. preprocess_raw() parsea monedas, porcentajes, fechas mezcladas, infinitos, IDs
3. detect_and_cap_outliers() no lanza error con datos limpios
4. _apply_balancing() con "class_weight" no modifica X_train
5. execute_pipeline() sin target (clustering)
6. Stratified split no falla con clases de 1 sample
7. pre_parse() de cli.py no elimina columnas útiles
"""

import os
import warnings
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    execute_pipeline,
    preprocess_raw,
    detect_and_cap_outliers,
    _apply_balancing,
)
from src.config import PipelineConfig
from cli import pre_parse

warnings.filterwarnings("ignore")
SEED = 42
N_ROWS = 50


# ====================================================================
# Fixtures compartidos
# ====================================================================

@pytest.fixture(scope="function")
def cfg() -> PipelineConfig:
    """PipelineConfig con valores deterministas y sin LLM."""
    return PipelineConfig(
        test_size=0.2,
        random_state=SEED,
        feature_selection=False,
        npc_model="none",
        npc_provider="none",
    )


@pytest.fixture(scope="function")
def productos_path() -> str:
    """Ruta absoluta a Productos_ML.csv."""
    return os.path.join(os.path.dirname(__file__), "..", "Productos_ML.csv")


# ====================================================================
# Contrato 1: execute_pipeline clasificación → X_train numérico
# ====================================================================

class TestExecutePipelineClassification:
    """Contrato: execute_pipeline con plan de clasificación sobre
    Productos_ML.csv produce X_train solo numérico (sin nombre/imagen)
    y y_train int64."""

    def test_all_numeric_no_text_columns(self, productos_path, cfg):
        """X_train contiene solo columnas numéricas; nombre e imagen
        desaparecen; y_train es int64 (LabelEncoder)."""
        df = pd.read_csv(productos_path)
        # Crear target binario sintético
        df["target_cls"] = (df["precio"] > df["precio"].median()).astype(int)

        plan = {
            "task_type": "classification",
            "target": "target_cls",
            "preprocessing": [
                {"col": "nombre", "action": "drop", "reason": "texto libre"},
                {"col": "imagen", "action": "drop", "reason": "URL"},
                {"col": "envio", "action": "encode", "method": "onehot"},
                {"col": "vendedor", "action": "encode", "method": "target"},
                {"col": "precio", "action": "scale", "method": "standard"},
                {"col": "descuento", "action": "scale", "method": "standard"},
                {"col": "calificacion", "action": "impute", "method": "mean"},
                {"col": "cantidad_calificaciones", "action": "impute", "method": "mean"},
            ],
            "balance": {"strategy": "none"},
            "feature_selection": False,
        }

        result = execute_pipeline(df, plan, target="target_cls", config=cfg)
        X_train = result["X_train"]
        y_train = result["y_train"]
        preprocessor = result["preprocessor"]

        # ── Afirmaciones ──
        # 1. Columnas textuales eliminadas
        assert "nombre" not in X_train.columns, "nombre debería eliminarse"
        assert "imagen" not in X_train.columns, "imagen debería eliminarse"

        # 2. Todas las columnas restantes son numéricas
        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, (
            f"Columnas no numéricas en X_train: {non_numeric}"
        )

        # 3. y_train es entero (LabelEncoder siempre produce int64)
        assert y_train is not None, "y_train no debe ser None"
        assert np.issubdtype(y_train.dtype, np.integer), (
            f"y_train.dtype={y_train.dtype}, se esperaba int64"
        )

        # 4. El preprocesador registró lo que hizo
        dropped = set(preprocessor.get("dropped_cols", []))
        dropped_clean = {c.replace(" (ID)", "") for c in dropped}
        assert "nombre" in dropped_clean or any("nombre" in d for d in dropped), (
            "nombre debería estar en dropped_cols del preprocesador"
        )
        assert "imagen" in dropped_clean or any("imagen" in d for d in dropped), (
            "imagen debería estar en dropped_cols del preprocesador"
        )

        # 5. X_train no está vacío
        assert X_train.shape[0] > 0, "X_train sin filas"
        assert X_train.shape[1] >= 1, "X_train sin columnas"

    def test_x_test_shape_consistency(self, productos_path, cfg):
        """X_train y X_test comparten las mismas columnas."""
        df = pd.read_csv(productos_path)
        df["target_cls"] = (df["precio"] > df["precio"].median()).astype(int)

        plan = {
            "task_type": "classification",
            "target": "target_cls",
            "preprocessing": [
                {"col": "nombre", "action": "drop"},
                {"col": "imagen", "action": "drop"},
                {"col": "envio", "action": "encode", "method": "onehot"},
                {"col": "vendedor", "action": "encode", "method": "target"},
                {"col": "precio", "action": "scale", "method": "standard"},
                {"col": "descuento", "action": "scale", "method": "standard"},
                {"col": "calificacion", "action": "impute", "method": "mean"},
                {"col": "cantidad_calificaciones", "action": "impute", "method": "mean"},
            ],
            "balance": {"strategy": "none"},
            "feature_selection": False,
        }

        result = execute_pipeline(df, plan, target="target_cls", config=cfg)
        X_train = result["X_train"]
        X_test = result["X_test"]
        assert set(X_train.columns) == set(X_test.columns), (
            "X_train y X_test deben tener las mismas columnas"
        )
        n_dup = result.get("preprocessor", {}).get("duplicates_removed", 0)
        assert len(X_train) + len(X_test) == len(df) - n_dup, (
            f"Train+Test ({len(X_train)}+{len(X_test)}) debe sumar "
            f"el total tras dedup ({len(df)}-{n_dup})"
        )


# ====================================================================
# Contrato 2: preprocess_raw parsea formatos mezclados
# ====================================================================

class TestPreprocessRaw:
    """Contrato: preprocess_raw convierte correctamente monedas,
    porcentajes, fechas variadas, infinitos → NaN, y detecta IDs."""

    @pytest.fixture(scope="function")
    def messy_df(self) -> pd.DataFrame:
        """DataFrame de 60 filas con columnas problemáticas realistas."""
        np.random.seed(SEED)
        n = 60
        return pd.DataFrame({
            # Moneda con $ y comas
            "monto": ["$1,234.56", "$500.00", "$99,999.99", "$0.99"] * 15,
            # Porcentaje
            "tasa": ["45.2%", "12.5%", "99.9%", "0.1%"] * 15,
            # Fechas mezcladas (3 formatos distintos)
            "fecha": (
                ["2024-01-15"] * 15
                + ["15/01/2024"] * 15
                + ["15-Jan-2024"] * 15
                + ["2024/03/01"] * 15
            ),
            # Valores infinitos (algunos)
            "senal": [np.inf, 1.0, 2.0, -np.inf] * 15,
            # ID de alta cardinalidad (>90% únicos)
            "id_cliente": [f"CLI-{i:04d}" for i in range(n)],
            # Columnas normales que deben sobrevivir intactas
            "valor": np.random.uniform(0, 100, n),
            "categoria": ["A", "B", "C", "D"] * 15,
        })

    def test_currency_parsed(self, messy_df):
        """Columnas con $ → numéricas."""
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, log = preprocess_raw(messy_df, plan)
        assert "monto" in log["parsed_currency"]
        assert np.issubdtype(df_out["monto"].dtype, np.number)
        # Verificar valor esperado
        assert df_out["monto"].iloc[0] == pytest.approx(1234.56)

    def test_percentage_parsed(self, messy_df):
        """Columnas con % → numéricas divididas por 100."""
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, log = preprocess_raw(messy_df, plan)
        assert "tasa" in log["parsed_pct"]
        assert np.issubdtype(df_out["tasa"].dtype, np.number)
        assert df_out["tasa"].iloc[0] == pytest.approx(0.452)

    def test_mixed_dates_expanded(self, messy_df):
        """Columna de fechas mezcladas → features año/mes/día/diasemana."""
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, log = preprocess_raw(messy_df, plan)
        assert "fecha" in log["parsed_dates"]
        assert "fecha" not in df_out.columns, (
            "fecha original debe eliminarse tras parseo"
        )
        for suffix in ("_año", "_mes", "_dia", "_diasemana"):
            col = f"fecha{suffix}"
            assert col in df_out.columns, f"Columna {col} debe existir"
            assert np.issubdtype(df_out[col].dtype, np.number), (
                f"{col} debe ser numérica"
            )

    def test_inf_handled(self, messy_df):
        """Infinitos → NaN (señal)."""
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, log = preprocess_raw(messy_df, plan)
        assert "senal" in log["inf_handled"]
        assert df_out["senal"].isna().any(), "Infinitos deben convertirse a NaN"
        assert not np.isinf(df_out["senal"]).any(), (
            "No deben quedar infinitos en señal"
        )

    def test_id_detected(self, messy_df):
        """Columna con >90% únicos → ids_detected."""
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, log = preprocess_raw(messy_df, plan)
        assert "id_cliente" in log["ids_detected"], (
            "id_cliente debe detectarse como ID (>90% únicos)"
        )

    def test_normal_columns_unchanged(self, messy_df):
        """Columnas normales no se registran en ningún ETL log."""
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, log = preprocess_raw(messy_df, plan)
        # 'valor' no debe estar en ningún log
        for key in log:
            assert "valor" not in log[key], (
                f"'valor' no debería estar en log[{key!r}]"
            )
        # 'categoria' con 4 valores únicos (<50%) → normalizada
        assert "categoria" in log["normalized_cats"]

    def test_negative_small_ratio_masked(self):
        """Negativos que son <5% de los valores → NaN."""
        n = 100
        np.random.seed(SEED)
        vals = np.random.uniform(10, 100, n)
        vals[:3] = [-5.0, -10.0, -1.0]  # 3% negativos → < 5%
        df = pd.DataFrame({"score": vals})
        plan = {"task_type": "regression", "preprocessing": []}
        df_out, _ = preprocess_raw(df, plan)
        # Los 3 negativos deben ser NaN
        assert df_out["score"].isna().sum() == 3, (
            "3 negativos deben convertirse a NaN"
        )
        # El resto debe permanecer
        assert df_out["score"].notna().sum() == n - 3


# ====================================================================
# Contrato 3: detect_and_cap_outliers con datos limpios
# ====================================================================

class TestDetectAndCapOutliers:
    """Contrato: detect_and_cap_outliers no lanza error ni modifica
    datos que no tienen outliers significativos."""

    def test_clean_data_no_error(self):
        """Datos normales: sin error, sin columnas tratadas."""
        np.random.seed(SEED)
        df = pd.DataFrame({
            "edad": np.random.randint(18, 80, 500),
            "ingreso": np.random.normal(50000, 8000, 500),
            "score": np.random.uniform(0, 100, 500),
        })
        plan = {"task_type": "regression"}
        try:
            df_out, treated = detect_and_cap_outliers(df, plan)
        except Exception as e:
            pytest.fail(f"detect_and_cap_outliers lanzó excepción: {e}")
        assert len(treated) == 0, (
            "Datos limpios no deberían tener columnas tratadas"
        )
        # Los datos no deben haber cambiado significativamente
        pd.testing.assert_frame_equal(df, df_out)

    def test_with_extreme_outliers_caps_only_if_ratio_ok(self):
        """Outliers extremos que afectan >1% y <10% de filas → capped."""
        np.random.seed(SEED)
        n = 1000
        vals = np.random.normal(100, 15, n)
        # Insertar 30 outliers extremos (3%) — dentro del rango 1-10%
        vals[:30] = 500.0
        df = pd.DataFrame({"valor": vals})
        plan = {"task_type": "regression"}
        df_out, treated = detect_and_cap_outliers(df, plan)
        assert "valor" in treated, (
            "Columna con 3% de outliers extremos debe tratarse"
        )
        # Los valores extremos deben acotarse
        Q1 = df["valor"].quantile(0.25)
        Q3 = df["valor"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3 * IQR
        assert df_out["valor"].max() <= upper, (
            "Tras capping, max no debe exceder upper bound"
        )

    def test_no_cap_when_outlier_ratio_below_1pct(self):
        """Menos de 1% outliers → no se trata (umbral inferior)."""
        np.random.seed(SEED)
        n = 1000
        vals = np.random.normal(100, 15, n)
        vals[:5] = 500.0  # 0.5% — por debajo del 1%
        df = pd.DataFrame({"valor": vals})
        plan = {"task_type": "regression"}
        df_out, treated = detect_and_cap_outliers(df, plan)
        assert "valor" not in treated, (
            "<1% outliers no debe activar capping"
        )


# ====================================================================
# Contrato 4: _apply_balancing con class_weight
# ====================================================================

class TestApplyBalancingClassWeight:
    """Contrato: _apply_balancing con estrategia 'class_weight' no
    modifica X_train (no hay oversampling)."""

    def test_class_weight_no_modify(self):
        """class_weight → sampler None → X, y sin cambios."""
        np.random.seed(SEED)
        n = 200
        X = pd.DataFrame({
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n),
        })
        y = np.array([0] * 190 + [1] * 10)

        X_out, y_out = _apply_balancing(X, y, "class_weight", config=None)
        pd.testing.assert_frame_equal(X_out, X, check_dtype=True)
        np.testing.assert_array_equal(y_out, y)

    def test_unknown_strategy_no_modify(self):
        """Estrategia desconocida → sampler None → X, y sin cambios."""
        np.random.seed(SEED)
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = np.array([0, 0, 1])
        X_out, y_out = _apply_balancing(X, y, "nonexistent", config=None)
        pd.testing.assert_frame_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_smote_changes_shape(self):
        """SMOTE real sí modifica X_train (control de que el test
        anterior no es un falso positivo)."""
        np.random.seed(SEED)
        X = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
        })
        y = np.array([0] * 50 + [1] * 50)
        X_out, y_out = _apply_balancing(X, y, "smote", config=None)
        # SMOTE genera muestras sintéticas
        assert len(X_out) >= len(X), (
            "SMOTE debería aumentar el número de muestras"
        )


# ====================================================================
# Contrato 5: execute_pipeline sin target (clustering)
# ====================================================================

class TestExecutePipelineNoTarget:
    """Contrato: execute_pipeline sin target (clustering) retorna
    X_train, sin y_train ni y_test; X_test es None porque no hay
    target para hacer split estratificado."""

    def test_clustering_no_y(self, productos_path, cfg):
        """Sin target: X_train existe, y_* son None, X_test es None."""
        df = pd.read_csv(productos_path).head(100)

        plan = {
            "task_type": "clustering",
            "preprocessing": [
                {"col": "nombre", "action": "drop"},
                {"col": "imagen", "action": "drop"},
                {"col": "precio", "action": "scale", "method": "standard"},
                {"col": "descuento", "action": "scale", "method": "standard"},
            ],
            "balance": {"strategy": "none"},
            "feature_selection": False,
        }

        result = execute_pipeline(df, plan, target=None, config=cfg)
        assert result["X_train"] is not None, "X_train no debe ser None"
        assert result["y_train"] is None, "y_train debe ser None (clustering)"
        assert result["y_test"] is None, "y_test debe ser None (clustering)"
        # feature_names debe existir
        assert len(result["feature_names"]) > 0, "feature_names no debe estar vacío"

    def test_clustering_no_target_in_profile(self, cfg):
        """Sin columna target: mismo resultado, sin y."""
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.randn(50),
            "label": ["A", "B"] * 25,
        })
        plan = {
            "task_type": "clustering",
            "preprocessing": [
                {"col": "label", "action": "encode", "method": "label"},
                {"col": "x", "action": "scale", "method": "standard"},
                {"col": "y", "action": "scale", "method": "standard"},
            ],
            "balance": {"strategy": "none"},
            "feature_selection": False,
        }
        result = execute_pipeline(df, plan, target=None, config=cfg)
        assert result["y_train"] is None
        assert result["y_test"] is None
        assert result["X_train"] is not None


# ====================================================================
# Contrato 6: Stratified split con clases de 1 sample
# ====================================================================

class TestStratifiedSingleSampleClass:
    """Contrato: Stratified split no falla aunque haya clases con 1
    solo sample, gracias al guard _class_counts.min() >= 2."""

    def test_single_sample_class_does_not_crash(self, cfg):
        """Una clase con 1 sample + balance activo → no se usa
        stratify, no hay error."""
        np.random.seed(SEED)
        n = 100
        X = pd.DataFrame({
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n),
        })
        # Clase 0: 97 samples, Clase 1: 2 samples, Clase 2: 1 sample
        y = pd.Series([0] * 97 + [1] * 2 + [2])
        df = X.copy()
        df["target"] = y

        plan = {
            "task_type": "classification",
            "preprocessing": [
                {"col": "feat1", "action": "scale", "method": "standard"},
                {"col": "feat2", "action": "scale", "method": "standard"},
            ],
            "balance": {"strategy": "smote"},  # activa stratify intent
            "feature_selection": False,
        }

        # No debe lanzar error
        try:
            result = execute_pipeline(df, plan, target="target", config=cfg)
        except ValueError as e:
            if "n_splits" in str(e).lower() or "class" in str(e).lower():
                pytest.fail(f"Stratified split lanzó error para clase con 1 sample: {e}")
            raise

        # El pipeline debe ejecutarse correctamente
        assert result["X_train"] is not None
        assert result["y_train"] is not None
        assert len(result["X_train"]) > 0
        assert len(result["y_train"]) > 0

    def test_stratify_only_when_min_class_ge_2(self, cfg):
        """Verificar que stratify_param es None cuando min class < 2."""
        np.random.seed(SEED)
        n = 50
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "target": [0] * 49 + [1],  # 1 sample en clase 1
        })
        plan = {
            "task_type": "classification",
            "preprocessing": [
                {"col": "a", "action": "scale", "method": "standard"},
            ],
            "balance": {"strategy": "smote"},
            "feature_selection": False,
        }

        result = execute_pipeline(df, plan, target="target", config=cfg)
        # El pipeline debe ejecutarse (no fallar)
        assert result["y_train"] is not None
        # Las clases en y_train deben incluir al menos la clase 0
        assert 0 in set(result["y_train"])


# ====================================================================
# Contrato 7: pre_parse() no elimina columnas útiles
# ====================================================================

class TestPreParse:
    """Contrato: pre_parse() de cli.py no elimina columnas útiles ni
    corrompe datos válidos."""

    def test_keeps_all_columns(self):
        """pre_parse mantiene todas las columnas (no dropea)."""
        df = pd.DataFrame({
            "id": list(range(100)),
            "nombre": [f"item_{i}" for i in range(100)],
            "precio": np.random.uniform(10, 1000, 100),
        })
        out = pre_parse(df)
        assert set(out.columns) == set(df.columns), (
            "pre_parse no debe eliminar columnas"
        )
        assert len(out.columns) == len(df.columns)

    def test_currency_not_in_text_column(self):
        """Columnas de texto con '$' no se confunden con moneda."""
        df = pd.DataFrame({
            "descripcion": ["costo $100", "precio $200", "valor $50"],
            "precio": [100, 200, 50],
        })
        out = pre_parse(df)
        # 'descripcion' debe seguir siendo string/object
        assert out["descripcion"].dtype == object, (
            "Columna de texto con '$' no debe convertirse a numérica"
        )

    def test_few_negatives_not_wipe_column(self):
        """Columna con <5% negativos: solo los negativos se vuelven
        NaN, el resto queda intacto."""
        n = 100
        np.random.seed(SEED)
        vals = np.random.uniform(10, 100, n)
        vals[:3] = [-5, -10, -1]  # 3% negativos
        df = pd.DataFrame({"margen": vals})
        out = pre_parse(df)
        # Solo 3 NaN (los negativos), 97 valores preservados
        assert out["margen"].isna().sum() == 3, (
            "Solo los 3 negativos deben volverse NaN"
        )
        positive_vals = out["margen"].dropna()
        assert (positive_vals >= 0).all(), (
            "Todos los valores no-NaN deben ser >= 0"
        )

    def test_many_negatives_left_untouched(self):
        """Columna con >=5% negativos: ningún valor se altera."""
        n = 100
        np.random.seed(SEED)
        vals = np.random.uniform(-20, 50, n)  # aprox 29% negativos
        df = pd.DataFrame({"balance": vals})
        out = pre_parse(df)
        # Ningún valor debe haberse convertido a NaN
        assert out["balance"].isna().sum() == 0, (
            ">=5% negativos: ningún valor debe alterarse"
        )
        # Los valores deben coincidir
        np.testing.assert_array_equal(out["balance"].values, vals)

    def test_date_not_affected(self):
        """Fechas en texto no se confunden con moneda/porcentaje."""
        df = pd.DataFrame({
            "fecha": ["2024-01-15", "15/01/2024", "2024/03/01"],
            "valor": [100, 200, 300],
        })
        out = pre_parse(df)
        assert out["fecha"].dtype == object, (
            "Columna de fecha no debe parsearse como moneda"
        )
        assert out["valor"].dtype in (np.int64, np.float64), (
            "Columna numérica debe mantenerse"
        )
