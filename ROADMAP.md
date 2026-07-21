# Roadmap — MELO

> Próximas mejoras priorizadas por impacto vs. esfuerzo, orientadas a convertir esta herramienta en un asistente completo para analistas y científicos de datos.

---

## Leyenda

| Símbolo | Significado |
|---|---|
| 🔥 | Impacto alto, código bajo (1-2 días) |
| 🚀 | Impacto alto, código medio (3-5 días) |
| 💎 | Valor añadido, esfuerzo medio |
| 🧪 | Experimental / investigación |

---

## Fase 1 — Fortalecer el análisis (prioridad máxima)

### ✅ 1.1 Importancia de features visual

**Estado:** ✅ Completado — `src/trainer.py` extrae `feature_importances_`/`coef_` del mejor modelo, `app.py` lo muestra como barras en Resultados.

**Problema:** El reporte dice "*Revisa las features más importantes del mejor modelo*" pero no muestra cuáles son. Es la pregunta #1 de todo data scientist después de entrenar.

**Solución:** Extraer `feature_importances_` (RandomForest, GradientBoosting) y `coef_` (LogisticRegression, SVC lineal) del mejor modelo y mostrarlos como gráfico de barras horizontal en la UI.

**Archivos a modificar:**
- `src/trainer.py` — Capturar importances en `ranked_models`
- `app.py` — Agregar sección "Importancia de features" en el tab Resultados, después del Ensemble

**Ejemplo visual:**
```
🏆 Importancia de features — RandomForestClassifier
┌─────────────────────────────────────────┐
│ precio                  ████████████ 42%│
│ descuento               ████████     28%│
│ vendedor                ████         14%│
│ precio_relativo_vendedor ███         12%│
│ envio                   ██           4% │
│ cantidad_calificaciones  █           2% │
│ precio_con_descuento     0%             │
└─────────────────────────────────────────┘
```

**Esfuerzo:** ~40 líneas | **Riesgo:** Bajo (solo lectura de atributos de sklearn)

**Criterio de éxito:** Al hacer clic en cualquier modelo del ranking, se muestra su importancia de features ordenada.

---

### ✅ 1.2 Exportar modelo entrenado (descargar .pkl)

**Estado:** ✅ Completado — `--save-model` en CLI, botón de descarga en UI, y `predict.py` para scoring batch.

**Problema:** `save_model=True` existe en `config.py` pero no hay botón en la UI para descargar el modelo. El usuario no puede llevarse el modelo a otro entorno.

**Solución:** Agregar botón "Descargar modelo (.pkl)" en la UI que serialice el mejor modelo con joblib y lo ofrezca como descarga. Además, crear script `predict.py` para cargar el modelo y predecir sobre datos nuevos.

**Archivos a modificar:**
- `app.py` — Botón de descarga en Resultados después del Ensemble + tab "Predecir"
- `predict.py` (nuevo) — Script CLI para batch scoring: `python predict.py modelo.pkl datos_nuevos.csv`

**Esfuerzo:** ~50 líneas | **Riesgo:** Bajo (joblib ya está como dependencia)

**Criterio de éxito:** Usuario puede descargar el modelo desde la UI, y ejecutar `python predict.py modelo.pkl datos_nuevos.csv` para obtener predicciones.

---

## Fase 2 — Cerrar el ciclo producción

### ✅ 2.1 Batch scoring (parcial)

**Estado:** ✅ Parcialmente completado — `predict.py` funcional y tab "Predecir" en UI. Pendiente: serializar el pipeline completo (preprocessor + modelo) para que predict.py aplique el preprocesamiento exacto del entrenamiento (hoy lo aproxima con los datos nuevos).

**Problema:** El modelo entrenado solo predice sobre el test set. Para usarlo en producción, el usuario necesita cargar datos nuevos (sin target), aplicar el mismo preprocesamiento, y obtener predicciones.

**Solución:** El botón "Descargar modelo" incluye el pipeline completo (preprocessor + modelo) en un solo `.pkl`. El script `predict.py` carga el pipeline y predice. Además, agregar un tab "Predecir" en la UI donde el usuario sube un CSV sin target y obtiene predicciones.

**Archivos a modificar/crear:**
- `predict.py` (nuevo) — Script de batch scoring
- `app.py` — Nuevo tab "Predecir" para scoring desde la UI
- `src/pipeline.py` — Guardar el preprocessor junto con los scalers/encoders

**Esfuerzo:** ~100 líneas | **Riesgo:** Medio (hay que asegurar que el pipeline se serializa correctamente)

**Criterio de éxito:** Usuario entrena un modelo, descarga el `.pkl`, y corre `python predict.py modelo.pkl datos_nuevos.csv > predicciones.csv`.

---

### 🚀 2.2 Dashboard de calidad de datos

---

## Fase 3 — EDA inteligente y datasets masivos (DuckDB)

> DuckDB entra como **motor analítico embebido**, no como reemplazo de nada. Dos roles: (1) ejecutar preguntas de negocio en lenguaje natural antes del pipeline, y (2) perfilar datasets grandes sin saturar RAM. En ambos casos, DuckDB opera sobre el CSV o DataFrame directamente, zero-copy.

### 🚀 3.1 EDA en lenguaje natural (DuckDB + LLM)

**Estado:** ⏳ Propuesto

**Problema:** MELO hoy es un pipeline ciego: cargas un CSV y arranca a procesar sin que nadie —ni el usuario ni el sistema— entienda realmente qué hay en los datos. El analista termina abriendo Jupyter o Excel para explorar antes de volver a MELO. El estratega decide features basado solo en estadísticas frías (nulos, cardinalidad, skew) sin contexto de negocio.

**Solución:** DuckDB + LLM habilitan **dos capacidades independientes** que comparten el mismo motor NL→SQL, pero sirven a audiencias distintas:

```
┌── Capa 1: EDA conversacional para el analista ──────────────────────┐
│                                                                      │
│  "No necesito Jupyter. Le pregunto a mis datos."                    │
│                                                                      │
│  Usuario: "¿cuál es el ticket promedio por vendedor en el Q4?"       │
│       ↓                                                              │
│  LLM → SQL → DuckDB → tabla interactiva en la UI                    │
│                                                                      │
│  ⚡ Valor: elimina el ciclo "abrir Jupyter → cargar CSV → escribir   │
│     pandas → interpretar resultado → volver a MELO". El analista     │
│     explora sus datos en español, sin código, sin salir de la app.   │
│     Esta capa se justifica sola — incluso si nunca ejecutas el       │
│     pipeline, ya ganaste.                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌── Capa 2: Insights automáticos para el estratega ───────────────────┐
│                                                                      │
│  "No necesito que el usuario me diga qué preguntar."                 │
│                                                                      │
│  El pipeline lanza 5-10 preguntas automáticas basadas en el perfil   │
│  del dataset (columnas, tipos, target) y ejecuta:                    │
│                                                                      │
│  • "¿qué segmentos tienen la tasa de churn más alta?"                │
│  • "¿hay columnas con comportamiento atípico por categoría?"         │
│  • "¿cómo varía el target entre los valores extremos de cada num?"   │
│  • "¿existen grupos de filas con patrones contradictorios?"          │
│       ↓                                                              │
│  Resultados → profile["eda_insights"] → prompt del estratega         │
│                                                                      │
│  ⚡ Valor: el estratega ya no decide a ciegas. Ve patrones de        │
│     negocio reales, no solo estadísticas descriptivas. Sus           │
│     sugerencias de features y su justificación están basadas en      │
│     datos, no en heurísticas genéricas.                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**¿Qué gana cada actor con esto?**

| Actor | Sin DuckDB | Con DuckDB |
|---|---|---|
| **Analista** | Abre Jupyter, escribe `df.groupby('ciudad')['churn'].mean()`, interpreta, vuelve a MELO | Escribe en español en la UI: *"¿churn promedio por ciudad?"* → tabla en <2s. Sin salir de la app. |
| **Estratega** | Ve: `antigüedad_meses`, skew 1.2, 5% nulos | Ve: *"Clientes con <6 meses de antigüedad tienen 4× más churn"* → sugiere `bin(antigüedad, [0,6,12,24])` |
| **Estratega** | Ve: `ciudad`, 15 categorías, cardinalidad media | Ve: *"Bogotá concentra 40% del churn total"* → sugiere `keyword(ciudad, 'Bogotá')` como flag |
| **Estratega** | Ve: correlación `ingreso ~ churn = -0.15` | Ve: *"En el segmento premium, ingreso y churn correlacionan -0.45"* → sugiere `interact(ingreso, es_premium)` |
| **Estratega** | Justificación genérica: *"desbalanceo moderado, se usará SMOTE"* | Justificación rica: *"Los clientes premium en Bogotá con <6 meses de antigüedad concentran el 60% del churn. Se crearán features de interacción para capturar este patrón."* |
| **Usuario final** | Recibe un reporte con métricas | Recibe un reporte que **incluye hallazgos de negocio** descubiertos durante el EDA automático, no solo scores del modelo |

**Arquitectura técnica:**

```
src/explorer.py (nuevo)
├── explore_dataset(df, questions, config) → dict
│   ├── _nl_to_sql(prompt, schema) → str       # LLM traduce pregunta a SQL
│   ├── _execute_duckdb(df, sql) → DataFrame   # DuckDB ejecuta (zero-copy)
│   └── _auto_questions(profile) → list[str]   # Preguntas predefinidas según tipo de datos
│
└── DuckDB opera sobre:
    - DataFrames de pandas (zero-copy vía Arrow)
    - CSVs directos (sin cargar en pandas) para datasets grandes
```

**Prompt engineering para NL→SQL:**
```
Eres un analista SQL. Traduce esta pregunta a una consulta DuckDB válida.

Esquema de la tabla 'df':
{columnas con tipos}

Reglas:
- Usa sintaxis DuckDB (compatible con PostgreSQL)
- Nombres de columna entre comillas dobles si tienen tildes o espacios
- Para porcentajes, multiplica por 100 y redondea a 1 decimal
- Limita resultados a 20 filas máximo
- Solo SELECT, nunca INSERT/UPDATE/DELETE

Pregunta: "¿qué productos tienen mayor churn por ciudad?"

Respuesta (solo SQL):
```

**Archivos a modificar/crear:**
- `src/explorer.py` (nuevo) — Motor de EDA: NL→SQL→DuckDB, ~150 líneas
- `src/strategist.py` — `build_plan()` recibe `eda_insights` y los incluye en el prompt del NPC
- `src/profiler.py` — `profile_dataset()` acepta un parámetro opcional `eda_insights: dict`
- `app.py` — Nuevo tab "Explorar" (antes de "Resultados") con chat input + tabla de resultados
- `cli.py` — Flag `--explore` para ejecutar preguntas automáticas antes del pipeline

**Esfuerzo:** ~250 líneas | **Riesgo:** Medio (dependencia nueva: DuckDB; la generación NL→SQL requiere prompt engineering cuidadoso)

**Criterio de éxito:**
1. Usuario escribe "¿cuál es el ticket promedio por vendedor?" y ve una tabla con la respuesta en <2 segundos
2. Sin intervención del usuario, el pipeline genera 3-5 preguntas automáticas y los insights aparecen en el reporte final
3. El estratega menciona hallazgos concretos del EDA en la justificación del plan

---

### 🚀 3.2 Perfilado acelerado para datasets grandes (>100 MB)

**Estado:** ⏳ Propuesto

**Problema:** `profile_dataset()` carga el CSV completo en pandas para calcular estadísticas. Con 5M+ de filas, esto consume gigabytes de RAM y tarda minutos. El usuario con un archivo de 2 GB simplemente no puede usar MELO.

**Solución:** Detectar tamaño del archivo al cargar. Si supera el umbral (configurable, default 100 MB), delegar a DuckDB las operaciones costosas del perfilado. DuckDB ejecuta en streaming — no carga el dataset completo en memoria.

**Operaciones que DuckDB acelera:**

| Operación | pandas (hoy) | DuckDB (propuesto) | Ganancia |
|---|---|---|---|
| `COUNT(*)` / `COUNT(DISTINCT col)` | `df[col].nunique()` — carga toda la columna | `SELECT COUNT(DISTINCT col) FROM df` — streaming | RAM: de O(n) a O(1) |
| Nulos por columna | `df[col].isna().sum()` — escanea toda la columna | `SELECT COUNT(*) - COUNT(col) FROM df` — una pasada para todas las columnas | Velocidad: 5-10× en datasets anchos |
| Percentiles (Q1, Q3, etc.) | `df[col].quantile([.25,.75])` — ordena la columna | `PERCENTILE_CONT(0.25) WITHIN GROUP` — aproximación eficiente | RAM + velocidad |
| Matriz de correlación | `df.corr()` — O(n × c²) en memoria | `SELECT CORR(a, b) FROM df` — vectorizado, paralelo | Velocidad: 3-5× |
| `value_counts()` para categóricas | `df[col].value_counts()` — hash table en RAM | `SELECT col, COUNT(*) FROM df GROUP BY col` — paralelo | Velocidad: 2-3× |

**Arquitectura:**

```python
# src/profiler.py — nuevo flujo
def profile_dataset(df_or_path, target=None, use_duckdb="auto"):
    if use_duckdb == "auto":
        use_duckdb = _should_use_duckdb(df_or_path)  # >100 MB o >500K filas
    
    if use_duckdb:
        return _profile_with_duckdb(df_or_path, target)
    else:
        return _profile_with_pandas(df_or_path, target)  # implementación actual

def _profile_with_duckdb(path_or_df, target):
    import duckdb
    con = duckdb.connect()  # in-memory, zero-conf
    
    # Si es path, registrar el CSV directamente (sin cargar en pandas)
    if isinstance(path_or_df, (str, Path)):
        con.execute(f"CREATE TABLE df AS SELECT * FROM read_csv_auto('{path_or_df}')")
    else:
        con.register("df", path_or_df)  # zero-copy desde pandas DataFrame
    
    # Una sola consulta para todas las estadísticas por columna
    stats = con.execute("""
        SELECT column_name, data_type,
               COUNT(*) as n_total,
               COUNT(column_name) as n_non_null,
               COUNT(DISTINCT column_name) as n_unique
        FROM (UNPIVOT df ON COLUMNS(* exclude(...)) INTO NAME column_name VALUE val)
        GROUP BY column_name, data_type
    """).fetchdf()
    
    # ... etc para correlaciones, percentiles, distribuciones
```

**Archivos a modificar:**
- `src/profiler.py` — Agregar `_should_use_duckdb()`, `_profile_with_duckdb()`, y el router en `profile_dataset()`
- `src/config.py` — Agregar `duckdb_threshold_mb: int = 100` y `duckdb_threshold_rows: int = 500_000`
- `requirements.txt` — Agregar `duckdb>=1.0`

**Esfuerzo:** ~180 líneas | **Riesgo:** Medio (DuckDB tiene su propio sistema de tipos; hay que manejar edge cases de compatibilidad con el perfil actual)

**Criterio de éxito:** Un CSV de 2 GB / 5M filas se perfila en <10s y con <500 MB de RAM (vs. los ~4 GB y >60s actuales).

---

### 💎 3.3 Reporte HTML exportable

**Estado:** ⏳ Propuesto (movido desde antigua sección huérfana)

**Problema:** El reporte actual es texto plano. Los interesados de negocio esperan un documento formateado con gráficos, tablas y diseño profesional.

**Solución:** Generar un reporte HTML con:
- Métricas clave con indicadores visuales
- Gráfico de importancia de features
- Tabla de predicciones con paginación
- Estilo profesional (CSS inline para portabilidad)
- Sección de insights EDA (si DuckDB estuvo activo)

**Archivos a modificar:**
- `src/reporter.py` — Nueva función `generate_html_report()`

**Esfuerzo:** ~120 líneas | **Riesgo:** Bajo

**Criterio de éxito:** Botón "Descargar reporte (.html)" en la UI que abre un documento con diseño profesional en el navegador.

---

### 💎 3.4 Explicabilidad con SHAP

**Estado:** ⏳ Propuesto

**Problema:** El ranking de modelos muestra importancia de features, pero no explica *por qué* una predicción individual fue X en lugar de Y. Para casos de alto riesgo (churn de cliente VIP, fraude), el usuario necesita explicaciones por instancia.

**Solución:** Integrar SHAP sobre el mejor modelo del ranking. Para modelos basados en árboles (RandomForest, GradientBoosting) usar `TreeExplainer` (rápido). Para modelos lineales (LogisticRegression, Ridge) usar `LinearExplainer`. Mostrar:
- Waterfall plot por instancia (UI interactiva)
- Summary plot global (qué features empujan las predicciones en cada dirección)

**Archivos a modificar:**
- `src/trainer.py` — Calcular SHAP values para el mejor modelo
- `app.py` — Sección "Explicabilidad" en tab Resultados
- `requirements.txt` — Agregar `shap>=0.46`

**Esfuerzo:** ~150 líneas | **Riesgo:** Medio (`shap` es pesado, puede relentizar la UI; calcular sobre una muestra)

**Criterio de éxito:** Usuario selecciona una fila del test set y ve un waterfall plot explicando por qué esa predicción tomó ese valor.

---

## Fase 4 — Experimentación y automatización

### 🧪 4.1 Búsqueda automática de modelos (AutoML extendido)

**Problema:** Hoy el pipeline prueba 4 modelos fijos. Un AutoML profesional debería explorar más opciones y seleccionar las mejores.

**Solución:** Agregar estrategias de búsqueda:
- `--strategy quick`: 4 modelos actuales (default)
- `--strategy deep`: Más folds, más combinaciones de hiperparámetros
- `--strategy exhaustive`: Prueba todos los modelos del catálogo con Optuna/Hyperopt

**Archivos a modificar:**
- `src/config.py` — Agregar perfiles de búsqueda
- `src/trainer.py` — Agregar soporte para Optuna

**Esfuerzo:** ~200 líneas | **Riesgo:** Medio (nueva dependencia)

---

### 🧪 4.2 Plugins de transformación personalizados

**Problema:** El feature engineering por IA sugiere transformaciones de una whitelist fija. Usuarios avanzados necesitan agregar sus propias transformaciones.

**Solución:** Sistema de plugins: el usuario coloca un archivo `.py` en `features/` con funciones decoradas que el pipeline descubre automáticamente:

```python
# features/mis_features.py
@feature_transform
def ingresos_por_persona(df, ingresos, personas):
    return df[ingresos] / df[personas].clip(lower=1)
```

**Archivos a modificar/crear:**
- `src/pipeline.py` — Cargador de plugins
- `features/` (nuevo directorio)

**Esfuerzo:** ~80 líneas | **Riesgo:** Bajo (no hay eval/exec)

---

## Resumen de prioridades

| # | Mejora | Fase | Impacto | Esfuerzo | Dependencias | Estado |
|---|---|---|---|---|---|---|
| 1 | Importancia de features visual | 1 | 🔥 Alto | ~40 líneas | Ninguna | ✅ |
| 2 | Exportar modelo + predict.py | 1 | 🔥 Alto | ~50 líneas | joblib | ✅ |
| 3 | Comparación de ejecuciones | 1 | 🔥 Alto | ~60 líneas | Ninguna | ⏳ |
| 4 | Batch scoring desde UI | 2 | 🚀 Alto | ~100 líneas | #2 completado | ⏳ parcial |
| 5 | Dashboard de calidad de datos | 2 | 🚀 Alto | ~120 líneas | matplotlib/plotly | ⏳ |
| 6 | EDA en lenguaje natural (DuckDB + LLM) | 3 | 🚀 Alto | ~250 líneas | duckdb, Ollama | ⏳ |
| 7 | Perfilado acelerado datasets grandes (DuckDB) | 3 | 🚀 Alto | ~180 líneas | duckdb | ⏳ |
| 8 | Reporte HTML exportable | 3 | 💎 Medio | ~120 líneas | Jinja2 | ⏳ |
| 9 | Explicabilidad SHAP | 3 | 💎 Medio | ~150 líneas | shap | ⏳ |
| 10 | AutoML extendido (Optuna) | 4 | 🧪 Medio | ~200 líneas | Optuna | ⏳ |
| 11 | Plugins de transformación | 4 | 🧪 Bajo | ~80 líneas | Ninguna | ⏳ |

---

## Notas técnicas

- **Prerrequisito transversal:** Todas las mejoras deben ser 100% locales, sin dependencia de cloud.
- **Compatibilidad:** Cada mejora debe funcionar con y sin Ollama. Sin LLM, las funcionalidades de IA se omiten gracefulmente.
- **Tests:** Cada mejora debe incluir tests unitarios o de integración según corresponda.
- **Versiones:** Recordar actualizar `requirements.txt` si se agregan nuevas dependencias.

---

## Refactorings pendientes (hallazgos de auditoría)

Priorizados por impacto × facilidad. Los ✅ ya están aplicados.

| # | Hallazgo | Archivo | Esfuerzo | Prioridad |
|---|---|---|---|---|
| ✅ | `import re` no usado | `pipeline.py` | 1 línea | 🔴 |
| ✅ | `SimpleImputer` no usado | `pipeline.py` | 1 línea | 🔴 |
| ✅ | `import sys` no usado | `app.py` | 1 línea | 🔴 |
| ✅ | Modelo guardado 2 veces | `trainer.py` | 3 líneas | 🔴 |
| ✅ | `numba()` → `nunique()` (bug) | `pipeline.py` | 1 línea | 🔴 |
| 1 | `execute_pipeline()`: 287 líneas | `pipeline.py` | Extraer helpers | 🟡 |
| 2 | `_default_plan()`: 218 líneas | `strategist.py` | Tablas de decisión | 🟡 |
| 3 | `train_models()`: 184 líneas | `trainer.py` | Extraer grid search | 🟡 |
| 4 | Parseo duplicado pre_parse vs preprocess_raw | `cli.py` + `pipeline.py` | Unificar ETL | 🟡 |
| 5 | Lógica de balanceo en config vs profiler | `config.py` + `profiler.py` | Centralizar | 🟢 |
| 6 | `app.py`: 701 líneas monolítico | `app.py` | Separar tabs | 🟢 |
| 7 | `reporter.py`: prompt LLM inline | `reporter.py` | Extraer a archivo | 🟢 |

---

*Última actualización: 2026-07-21*
