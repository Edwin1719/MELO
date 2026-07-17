"""
sample_data.py — Genera datasets sintéticos con problemas REALES
que encontrarias en datos de clientes: nulos, outliers, formatos
inconsistentes, fechas, valores sucios, etc.
"""

import numpy as np
import pandas as pd


def make_realistic_churn(n_samples=2000, seed=42) -> pd.DataFrame:
    """
    Dataset de churn realista con problemas comunes de ETL:

    ✅ Columnas numéricas con outliers
    ✅ Nulos en múltiples formatos y porcentajes
    ✅ Categóricas inconsistentes (Male, male, M)
    ✅ Fechas reales
    ✅ Valores con formato ($1,234.56, 45%, etc.)
    ✅ IDs de cliente (alta cardinalidad)
    ✅ Valores infinitos
    ✅ Ceros y negativos donde no debería haber
    """
    rng = np.random.default_rng(seed)

    n = n_samples

    df = pd.DataFrame()
    df["id_cliente"] = [f"CLI-{i:05d}" for i in range(10001, 10001 + n)]

    # ── Ingreso mensual (con outliers reales) ─────────────────────
    base_ingreso = rng.lognormal(mean=5.5, sigma=0.6, size=n)
    # Outliers: algunos ingresos extremadamente altos
    outlier_mask = rng.random(n) < 0.03
    base_ingreso[outlier_mask] = rng.uniform(50000, 200000, size=outlier_mask.sum())
    df["ingreso_mensual"] = np.round(base_ingreso, 0)

    # ── Ticket promedio (con formato $X,XXX.XX) ───────────────────
    ticket = rng.lognormal(mean=4.0, sigma=0.8, size=n)
    ticket = np.clip(ticket, 1000, 500000)
    # Convertir a string con formato moneda (sucio)
    ticket_str = []
    for i, val in enumerate(ticket):
        if rng.random() < 0.6:
            ticket_str.append(f"${val:,.0f}")
        else:
            ticket_str.append(f"{val:,.0f}")
    df["ticket_promedio"] = ticket_str

    # ── Antigüedad en meses (con valores negativos - error) ──────
    antiguedad = rng.exponential(scale=18, size=n).astype(int)
    neg_mask = rng.random(n) < 0.02
    antiguedad[neg_mask] = -antiguedad[neg_mask]
    df["antigüedad_meses"] = np.clip(antiguedad, -12, 120)

    # ── Llamadas al soporte (con valores anómalos altos) ─────────
    llamadas = rng.poisson(lam=2, size=n).astype(float)
    outlier_mask2 = rng.random(n) < 0.02
    llamadas[outlier_mask2] = rng.integers(50, 200, size=outlier_mask2.sum())
    df["llamadas_soporte"] = llamadas

    # ── Satisfacción (escala 1-5, con valores fuera de rango) ────
    satisfaccion = rng.integers(1, 6, size=n).astype(float)
    bad_mask = rng.random(n) < 0.03
    satisfaccion[bad_mask] = rng.choice([-1, 0, 6, 7, 99], size=bad_mask.sum())
    df["satisfaccion"] = satisfaccion

    # ── Porcentaje de descuento usado (con formato "45%") ────────
    desc_pct = rng.uniform(0, 50, size=n)
    desc_str = []
    for val in desc_pct:
        if rng.random() < 0.7:
            desc_str.append(f"{val:.1f}%")
        else:
            desc_str.append(f"{val:.1f}")
    # Algunos sin porcentaje
    no_pct = rng.random(n) < 0.05
    desc_str = ["" if no_pct[i] else desc_str[i] for i in range(n)]
    df["descuento_usado"] = desc_str

    # ── Género (inconsistente: Male, male, M, Masculino) ─────────
    generos_base = rng.choice(["Masculino", "Femenino", "Otro"], size=n, p=[0.45, 0.45, 0.1])
    generos_final = []
    for g in generos_base:
        r = rng.random()
        if g == "Masculino":
            generos_final.append(rng.choice(["M", "Male", "male", "Masculino", "H"]))
        elif g == "Femenino":
            generos_final.append(rng.choice(["F", "Female", "female", "Femenino", "Mujer"]))
        else:
            generos_final.append(rng.choice(["Otro", "NB", "No binario", "Prefiero no decirlo"]))
    df["genero"] = generos_final

    # ── Región con mezcla de mayúsculas/minúsculas ───────────────
    regiones = rng.choice(
        ["Bogotá", "bogota", "MEDELLIN", "Medellin", "Cali", "cali",
         "Barranquilla", "BARRANQUILLA", "Cartagena", "carthagena",
         "Cúcuta", "Cucuta", "cucuta"],
        size=n,
    )
    df["region"] = regiones

    # ── Fecha de última compra (con nulos y formatos mezclados) ──
    base_date = pd.Timestamp("2025-06-01")
    days_ago = rng.integers(0, 365, size=n).astype(float)
    # 8% nulos en fecha
    null_mask = rng.random(n) < 0.08
    days_ago[null_mask] = np.nan

    fechas = []
    for d in days_ago:
        if pd.isna(d):
            fechas.append("")
        else:
            dt = base_date - pd.Timedelta(days=int(d))
            fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y"])
            fechas.append(dt.strftime(fmt))
    df["fecha_ultima_compra"] = fechas

    # ── Nº de productos (con algunos valores negativos) ───────────
    num_prod = rng.poisson(lam=5, size=n).astype(float)
    neg_prod = rng.random(n) < 0.01
    num_prod[neg_prod] = -1
    df["num_productos"] = num_prod

    # ── Gasto total anual (con infinitos!) ───────────────────────
    gasto = rng.lognormal(mean=6.0, sigma=0.7, size=n)
    inf_mask = rng.random(n) < 0.01
    gasto[inf_mask] = np.inf
    df["gasto_total_anual"] = gasto

    # ── Edad (con valores 0 o >120) ──────────────────────────────
    edad = rng.normal(loc=42, scale=15, size=n).astype(int)
    edad = np.clip(edad, 0, 130)
    zero_mask = rng.random(n) < 0.02
    edad[zero_mask] = 0
    df["edad"] = edad

    # ── Nulos en múltiples columnas ──────────────────────────────
    # Ingreso: 3% nulos
    null_mask_i = rng.random(n) < 0.03
    df.loc[null_mask_i, "ingreso_mensual"] = np.nan

    # Antigüedad: 5% nulos
    null_mask_a = rng.random(n) < 0.05
    df.loc[null_mask_a, "antigüedad_meses"] = np.nan

    # Satisfacción: 10% nulos (alta)
    null_mask_s = rng.random(n) < 0.10
    df.loc[null_mask_s, "satisfaccion"] = np.nan

    # ── Variable objetivo (churn) con desbalanceo moderado ───────
    # Regla subyacente: clientes con baja satisfacción, muchas llamadas
    # de soporte, bajo ticket y poca antigüedad tienen más churn
    churn_prob = 0.1
    churn_prob += (satisfaccion < 3).astype(int).clip(0, 1) * 0.3
    churn_prob += (llamadas > 5).astype(int).clip(0, 1) * 0.2
    churn_prob += (ticket < 30000).astype(int).clip(0, 1) * 0.15
    churn_prob = np.clip(churn_prob * rng.uniform(0.8, 1.2, size=n), 0, 1)
    df["churn"] = (rng.random(n) < churn_prob).astype(int)

    return df


def make_realistic_sales(n_samples=2000, seed=42) -> pd.DataFrame:
    """
    Dataset de ventas realista para regresión.

    Problemas: nulos, outliers, fechas, formatos, IDs, valores negativos.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    df = pd.DataFrame()
    df["id_transaccion"] = [f"TXN-{i:08d}" for i in range(n)]

    # Fecha de venta
    start = pd.Timestamp("2024-01-01")
    dates = [start + pd.Timedelta(days=int(d)) for d in rng.integers(0, 545, n)]
    df["fecha_venta"] = [d.strftime("%Y-%m-%d") for d in dates]

    # Producto (categórica con alta cardinalidad)
    products = [f"PROD-{i:04d}" for i in range(1, 101)]
    df["producto"] = rng.choice(products, n)

    # Categoría de producto
    cats = ["Electrónicos", "Hogar", "Ropa", "Alimentos", "Salud",
            "Deportes", "Juguetes", "Libros"]
    df["categoria"] = rng.choice(cats, n)

    # Precio unitario (con formato $)
    precio = rng.lognormal(mean=4.5, sigma=0.9, size=n)
    # Convertir a formato sucio
    precio_str = [f"${p:,.2f}" for p in precio]
    # Algunos sin $
    no_dollar = rng.random(n) < 0.1
    precio_str = [precio_str[i] if not no_dollar[i] else str(precio[i])[:8] for i in range(n)]
    # Algunos con comas mal puestas
    bad_comma = rng.random(n) < 0.05
    precio_str = [ps.replace(",", ".") if bad_comma[i] else ps for i, ps in enumerate(precio_str)]
    df["precio_unitario"] = precio_str

    # Cantidad (con valores cero y negativos)
    cantidad = rng.poisson(lam=3, size=n).astype(float)
    cantidad[cantidad == 0] = 0  # algunos cero
    neg_cant = rng.random(n) < 0.01
    cantidad[neg_cant] = -cantidad[neg_cant]
    df["cantidad"] = cantidad

    # Descuento (% o valor fijo)
    desc_tipo = rng.choice(["porcentaje", "fijo", "ninguno"], n, p=[0.3, 0.2, 0.5])
    desc_valor = np.where(
        desc_tipo == "porcentaje",
        rng.uniform(5, 50, n),
        np.where(desc_tipo == "fijo", rng.uniform(1000, 50000, n), 0),
    )
    df["descuento"] = desc_valor
    df["tipo_descuento"] = desc_tipo

    # Región del vendedor
    df["region"] = rng.choice(
        ["Norte", "norte", "SUR", "Sur", "Este", "ESTE", "Oeste", "oeste",
         "Centro", "centro", "Nororiente"],
        n,
    )

    # Score del vendedor (1-10, con algunos >10)
    score = rng.integers(1, 11, size=n).astype(float)
    bad_score = rng.random(n) < 0.02
    score[bad_score] = rng.choice([-1, 0, 11, 15, 99], size=bad_score.sum())
    df["score_vendedor"] = score

    # Nulos
    null_precio = rng.random(n) < 0.04
    df.loc[null_precio, "precio_unitario"] = ""
    null_cant = rng.random(n) < 0.02
    df.loc[null_cant, "cantidad"] = np.nan

    # Outliers en precio
    outlier_precio = rng.random(n) < 0.02
    df.loc[outlier_precio, "precio_unitario"] = "$9,999,999.99"

    # Target: ingreso total de la venta (regresión)
    total_ingreso = np.where(
        cantidad > 0,
        precio * cantidad * (1 - desc_valor / 100 * (desc_tipo == "porcentaje")) - desc_valor * (desc_tipo == "fijo"),
        rng.uniform(0, 10000, n),  # ventas con cantidad 0 o negativa
    )
    df["ingreso_total"] = np.abs(total_ingreso) * rng.uniform(0.8, 1.2, n) * (
        1 + (score / 100)
    )

    return df


def make_realistic_segmentation(n_samples=1000, seed=42) -> pd.DataFrame:
    """
    Dataset de segmentación de clientes (no supervisado).

    Problemas: valores extremos, nulos, categóricas inconsistentes, IDs.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    df = pd.DataFrame()
    df["id"] = [f"USR-{i:06d}" for i in range(n)]

    # Ingreso
    ingreso = rng.lognormal(mean=5.2, sigma=0.7, size=n)
    outlier_ing = rng.random(n) < 0.02
    ingreso[outlier_ing] = rng.uniform(1000000, 5000000, size=outlier_ing.sum())
    df["ingreso_estimado"] = ingreso

    # Frecuencia de compra (con valores 0 y extremos)
    freq = rng.poisson(lam=8, size=n).astype(float)
    zero_freq = rng.random(n) < 0.05
    freq[zero_freq] = 0
    extreme_freq = rng.random(n) < 0.01
    freq[extreme_freq] = rng.integers(100, 500, size=extreme_freq.sum())
    df["frecuencia_anual"] = freq

    # Gasto promedio
    gasto = rng.lognormal(mean=4.2, sigma=0.9, size=n)
    df["gasto_promedio"] = gasto * (1 + rng.uniform(-0.05, 0.05, n))

    # Antigüedad con valores negativos y extremos
    ant = rng.exponential(scale=24, size=n)
    neg_ant = rng.random(n) < 0.03
    ant[neg_ant] = -ant[neg_ant]
    old_ant = rng.random(n) < 0.01
    ant[old_ant] = rng.uniform(200, 500, size=old_ant.sum())
    df["antigüedad_meses"] = ant

    # Nivel educativo (ordinal con errores)
    niveles = ["Primaria", "Secundaria", "Técnico", "Universidad", "Postgrado"]
    # Con errores de tipeo
    niveles_err = niveles + ["primaria", "universidad", "Universitario", "Tecnico", "POSTGRADO"]
    df["educacion"] = rng.choice(niveles_err, n)

    # Ciudad (con nombres mal escritos)
    ciudades = ["Bogotá", "Bogota", "Medellín", "Medellin", "Cali",
                "Barranquilla", "Cartagena", "Cartajena", "Cúcuta",
                "Cucuta", "Bucaramanga", "Pereira"]
    df["ciudad"] = rng.choice(ciudades, n)

    # Nulos en varias columnas
    for col in ["ingreso_estimado", "gasto_promedio", "antigüedad_meses"]:
        null_msk = rng.random(n) < 0.06
        df.loc[null_msk, col] = np.nan

    return df


if __name__ == "__main__":
    import os
    output_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(output_dir, exist_ok=True)

    print("Generando datasets realistas con problemas de ETL...\n")

    churn = make_realistic_churn()
    churn.to_csv(os.path.join(output_dir, "churn_real.csv"), index=False)
    print(f"✓ churn_real.csv ({len(churn)} filas, {len(churn.columns)} cols)")
    print(f"  Problemas incluidos:")
    print(f"  - Nulos en ingreso (3%), antigüedad (5%), satisfacción (10%)")
    print(f"  - Outliers en ingreso ($200K+), llamadas (200+), edad (0/130)")
    print(f"  - Moneda: ticket_promedio ($X,XXX), % en descuento_usado")
    print(f"  - Categóricas inconsistentes: Male/male/M, Bogotá/bogota")
    print(f"  - Fechas en múltiples formatos (3 formatos distintos)")
    print(f"  - Valores negativos en antigüedad y num_productos")
    print(f"  - Valores infinitos (∞) en gasto_total_anual")
    print(f"  - IDs de alta cardinalidad (id_cliente)")
    print(f"  - Target: churn con regla subyacente no trivial (basada en múltiples variables)")
    print()

    sales = make_realistic_sales()
    sales.to_csv(os.path.join(output_dir, "ventas_reales.csv"), index=False)
    print(f"✓ ventas_reales.csv ({len(sales)} filas, {len(sales.columns)} cols)")
    print()

    seg = make_realistic_segmentation()
    seg.to_csv(os.path.join(output_dir, "segmentacion_real.csv"), index=False)
    print(f"✓ segmentacion_real.csv ({len(seg)} filas, {len(seg.columns)} cols)")
    print()

    print("Listo. Para probar:")
    print("  python cli.py datasets/churn_real.csv --target churn")
    print("  python cli.py datasets/ventas_reales.csv --target ingreso_total")
    print("  python cli.py datasets/segmentacion_real.csv")
