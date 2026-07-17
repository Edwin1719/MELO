"""
predict.py — Scoring sobre datos nuevos con un modelo entrenado por MELO.

Aplica el mismo plan de preprocesamiento usado durante el entrenamiento
para que los datos nuevos pasen por las mismas transformaciones.

Uso:
    python predict.py output/modelo.pkl datos_nuevos.csv -o predicciones.csv
"""

import argparse
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from cli import pre_parse, load_data


def apply_preprocessing(df: pd.DataFrame, plan: dict, task_type: str) -> pd.DataFrame:
    """Aplica el plan de preprocesamiento sobre datos nuevos (sin target)."""
    df = df.copy()

    for action in plan.get("preprocessing", []):
        col = action["col"]
        if col not in df.columns:
            continue
        action_type = action["action"]

        if action_type == "drop":
            if col in df.columns:
                df = df.drop(columns=[col])

        elif action_type == "impute":
            method = action.get("method", "mean")
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                df[f"{col}_is_missing"] = 1
            if method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "zero":
                df[col] = df[col].fillna(0)
            elif method == "mode":
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
            elif method == "drop":
                df = df.dropna(subset=[col])

        elif action_type == "encode":
            method = action.get("method", "onehot")
            if method == "label":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            elif method == "onehot":
                ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
                encoded = ohe.fit_transform(df[[col]])
                categories = ohe.categories_[0][1:] if len(ohe.categories_[0]) > 1 else []
                col_names = [f"{col}_{cat}" for cat in categories]
                if col_names:
                    encoded_df = pd.DataFrame(encoded, columns=col_names, index=df.index)
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

        elif action_type == "scale":
            method = action.get("method", "standard")
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                continue
            df[col] = scaler.fit_transform(df[[col]]).ravel()

    return df


def main():
    parser = argparse.ArgumentParser(
        description="MELO — Predecir con modelo entrenado",
    )
    parser.add_argument("modelo", help="Ruta al .pkl del modelo entrenado")
    parser.add_argument("datos", help="Ruta al CSV con datos nuevos (sin target)")
    parser.add_argument("--output", "-o", default=None,
                        help="Ruta de salida (default: stdout)")
    args = parser.parse_args()

    # Cargar artifact
    artifact = joblib.load(args.modelo)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    task_type = artifact["task_type"]
    plan = artifact.get("plan", {})
    target_encoder = artifact.get("target_encoder")
    meta = artifact.get("metadata", {})

    print(f"  Modelo: {meta.get('name', '?')} "
          f"(CV: {meta.get('cv_mean', '?')} ±{meta.get('cv_std', '?')})", file=sys.stderr)
    print(f"  Features requeridas: {len(feature_names)} columnas", file=sys.stderr)
    print(file=sys.stderr)

    # Cargar datos y aplicar pre-parseo ligero (monedas, %, infinitos, negativos)
    df = load_data(args.datos)
    df = pre_parse(df)

    # Aplicar el plan de preprocesamiento
    df = apply_preprocessing(df, plan, task_type)

    # Verificar columnas requeridas
    missing = set(feature_names) - set(df.columns)
    if missing:
        print(f"❌ Faltan columnas: {missing}", file=sys.stderr)
        print(f"   Disponibles: {list(df.columns)}", file=sys.stderr)
        return

    # Seleccionar solo las features del entrenamiento (mismo orden)
    X = df[feature_names]

    # Predecir
    y_pred = model.predict(X)

    # Decodificar si es clasificación
    if target_encoder and task_type == "classification":
        y_pred = target_encoder.inverse_transform(y_pred.astype(int))

    # Armar resultado
    result = df.copy()
    result["prediccion"] = y_pred

    # Guardar o stdout
    if args.output:
        result.to_csv(args.output, index_label="fila")
        print(f"✅ Predicciones guardadas: {args.output}", file=sys.stderr)
    else:
        result.to_csv(sys.stdout, index_label="fila")


if __name__ == "__main__":
    main()
