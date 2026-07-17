# MELO

> **De datos sucios a predicciones listas, en un solo comando. 100% local.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)](https://scikit-learn.org/)
[![Ollama](https://img.shields.io/badge/Ollama-ready-8A2BE2)](https://ollama.com/)

---

## El problema

El 80% del tiempo de un científico de datos no se va en modelos — se va en **limpiar datos**, **entender las columnas**, **probar transformaciones**, y **decidir qué modelo usar**. Los AutoML existentes ignoran este problema: asumen datos limpios, no entienden el significado de las columnas, y te dejan sin predicciones exportables.

**Tú solo quieres una respuesta:** dado mi archivo CSV, ¿qué modelo funciona mejor y qué predicciones genera?

---

## La solución

MELO es un pipeline que toma tu archivo **sucio** (CSV o Excel) y en segundos te entrega:

```
📄 Reporte ejecutivo    →  métricas, ranking, ensemble, insights cualitativos
📊 Predicciones         →  tabla real vs predicción, descargable como CSV
🧬 Nuevas features      →  IA local sugiere transformaciones según tus columnas
🔧 Sin preprocessing    →  ETL automático: monedas, %, fechas, nulos, outliers
```

Todo **sin enviar datos a la nube**, sin instalar bases de datos, sin configurar pipelines de ETL.

---

## Lo que te ahorras

| Tú no tienes que... | MELO lo hace por ti |
|---|---|
| Parsear `$1,234.56` o `45.2%` | ✅ ETL automático |
| Decidir imputación (media vs mediana) | ✅ Según skew de cada columna |
| Detectar IDs, columnas constantes, outliers | ✅ Automático |
| Elegir encoding (onehot vs target) | ✅ Según cardinalidad |
| Balancear clases (SMOTE vs ADASYN vs weight) | ✅ Según tamaño de clase minoritaria |
| Crear features manualmente | ✅ IA local sugiere combinaciones |
| Probar 4 modelos con grid search | ✅ Automático + ensemble |
| Exportar predicciones | ✅ CSV descargable |

---

## Demo: 30 segundos

```bash
pip install -e .                          # 1. Instalar como paquete
python cli.py datos.csv --target churn     # 2. Ejecutar
```

Resultado:
```
📋 output/datos_reporte.txt            → Reporte completo
📊 output/datos_predicciones.csv       → Predicciones listas
```

O desde la interfaz gráfica:
```bash
streamlit run app.py
```
---

## Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ CARGA DATOS  │───▶│ ETL INTELIGENTE  │───▶│ FEATURE ENG.     │───▶│ AUTO-ML       │───▶│ ENSEMBLE      │───▶│ PREDICCIONES │
│ CSV / Excel  │    │ Duplicados       │    │ POR IA (Opcional)│    │ Grid search   │    │ Votación      │    │ + Reporte    │
│              │    │ Nulos textuales  │    │ ratio/interact   │    │ Cross-val     │    │ Ponderada     │    │ Ejecutivo    │
│              │    │ Dtype coercion   │    │ str_len/keyword  │    │ Ranking       │    │               │    │              │
│              │    │ Skew transform   │    │ date_diff/bin    │    │ Balanceo      │    │               │    │              │
│              │    │ Missing flags    │    │                  │    │ adaptativo    │    │               │    │              │
│              │    │ Perfilado        │    │                  │    │               │    │               │    │              │
│              │    │ Estrategia       │    │                  │    │               │    │               │    │              │
└─────────────┘    └──────────────────┘    └──────────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

### ETL automatizado — datos reales, no competencias de Kaggle

| Problema real | Solución |
|---|---|
| `$1,234.56` | Parseo automático → 1234.56 |
| `45.2%` | → 0.452 |
| Fechas en 5 formatos distintos | Multi-parse + features (año, mes, día, día_semana) |
| `Infinity`, `-Infinity` | → NaN → imputado |
| `Male`, `male`, `M`, `Masculino` | Normalización → `masculino` |
| IDs únicos (`CLI-05931`) | Detección y eliminación |
| Outliers extremos | Capping por IQR (3×) |
| Columnas constantes | Eliminación automática |
| Datos faltantes | Imputación adaptativa (media/mediana según skew) |
| **Filas duplicadas** | **Eliminación automática antes de ETL** |
| **Nulos textuales** (`"N/A"`, `"?"`, `"-"`) | **Reemplazo por NaN** |
| **Booleanos** | **Conversión a 0/1** |
| **Dtype incorrecto** (columna numérica leída como string) | **Coerción automática a numérico** |
| **Skew alto** en features numéricas | **Nueva feature `_log` además de la original** |
| **Missing informativo** | **Indicador `_is_missing` al imputar** |

### Feature Engineering por IA — el pipeline entiende tus columnas

Cuando Ollama está disponible, el pipeline **analiza los nombres y tipos de columna** de tu dataset y sugiere nuevas features con sentido de negocio:

| Operación | Descripción | Ejemplo real |
|---|---|---|
| `ratio(col, grupo)` | Valor relativo al promedio del grupo | `precio / precio_promedio_categoria` |
| `interact(a, b)` | Multiplicación de dos columnas | `precio × descuento` |
| `str_len(col)` | Longitud de texto como proxy de detalle | `largo_descripción` |
| `keyword(col, patrón)` | Flag semántico desde texto | `es_premium`, `tiene_descuento` |
| `date_diff(a, b)` | Días entre fechas | `antigüedad_cliente` |
| `bin(col, n)` | Discretización en rangos | `rango_edad`, `tramo_ingreso` |

Sin LLM, el pipeline funciona exactamente igual — solo omite este paso.

### Corrección inteligente del target

| Problema | Detección | Acción automática |
|---|---|---|
| **Target concentrado** (>60% en un valor) | `concentration_level: extreme/severe` | 🔴 Advertencia en UI antes de ejecutar |
| **Regresión sobre target discreto** (≤15 valores únicos) | `unique ≤ 15 + concentration extreme` | 🔄 Auto-switch a clasificación + balanceo |
| **Clase minoritaria muy pequeña** (<2 samples) | `min_class_count < 2` | `class_weight="balanced"` (sin SMOTE) |
| **Clase minoritaria pequeña** (2-5 samples) | `min_class_count < 6` | `SMOTE` (oversampling controlado) |
| **Desbalanceo severo** (>80% en una clase) | `balance_level: extreme` | `ADASYN` (oversampling adaptativo) |
| **Regresión con target concentrado** | `concentration_level + regression` | `sample_weight` por frecuencia inversa |

### Tipos de tarea soportados

| Tarea | Detección | Modelos |
|---|---|---|
| **Clasificación** | Auto (target categórico) | RandomForest, GradientBoosting, LogisticRegression, SVC |
| **Regresión** | Auto (target numérico continuo) | RandomForest Regressor, GradientBoosting Regressor, Ridge, SVR |
| **Clustering** | Sin target o `--task clustering` | KMeans, DBSCAN, Agglomerative |
| **Detección anomalías** | `--task anomaly_detection` | IsolationForest, LocalOutlierFactor |

---

## Requisitos

- **Python 3.10+**
- **Ollama** (opcional) — [Descargar](https://ollama.com/download)
- Modelos compatibles: `qwen2.5-coder`, `llama3.2`, `gemma3`, `gpt-oss`

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/tu-usuario/databiq-automl.git
cd databiq-automl

# 2. Entorno (opcional pero recomendado)
conda create -n automl python=3.11
conda activate automl

# 3. Instalar como paquete (recomendado)
pip install -e .

#    O solo las dependencias mínimas:
##  pip install -r requirements.txt

# 4. Ollama (opcional)
ollama serve
ollama pull qwen2.5-coder:7b

# 5. Verificar que funciona
python cli.py datasets/churn_clientes.csv --target churn --no-npc
```

### Extras opcionales

```bash
pip install -e .[full]   # + XGBoost, joblib
pip install -e .[npc]    # + Ollama, OpenAI
pip install -e .[test]   # + pytest para desarrollo
```

---

## Uso

### Clasificación
```bash
python cli.py datos.csv --target churn
```

### Regresión
```bash
python cli.py ventas.csv --target precio
```

### Clustering (sin target)
```bash
python cli.py clientes.csv
```

### Forzar tipo de tarea
```bash
python cli.py datos.csv --target score --task regression
```

### Sin LLM
```bash
python cli.py datos.csv --target churn --no-npc
```

### Interfaz gráfica
```bash
streamlit run app.py --server.port 8501
```

### Datasets de prueba
```bash
# Generar datasets sintéticos con problemas ETL reales
python -c "from src.sample_data import make_realistic_churn; make_realistic_churn().to_csv('datasets/mi_churn.csv', index=False)"

# Probar
python cli.py datasets/churn_real.csv --target churn
python cli.py datasets/ventas_reales.csv --target ingreso_total
python cli.py datasets/segmentacion_real.csv
```

### Opciones disponibles

```
python cli.py --help

Argumentos:
  file                  Ruta al archivo CSV o Excel
  --target, -t          Nombre de la columna objetivo
  --task                Forzar tipo de tarea
  --no-npc              Desactivar insights del NPC
  --output, -o          Directorio de salida (default: output/)
  --quiet, -q           Modo silencioso (solo reporte)
```

---

## Resultados

Después de ejecutar el pipeline, obtienes 3 entregables:

### 📋 Reporte ejecutivo
```
📁  DATOS — 1,077 filas, 8 columnas, target: calificacion
🔧  PREPROCESAMIENTO — 2 eliminadas, 2 codificaciones, 3 escalados
    🧬 Features por IA: precio_relativo_vendedor = ratio, precio_con_descuento = interact
🏆  MODELOS — 🥇 RandomForest 0.971 ±0.013
🧬  ENSEMBLE — accuracy 0.891, f1_weighted 0.921
🤖  INSIGHTS — Análisis cualitativo generado por IA
💡  RECOMENDACIONES — Accionables para el negocio
```

### 📊 Predicciones exportables
Archivo `output/dataset_predicciones.csv` con `valor_real`, `prediccion`, y `acierto` para cada fila del test set. Descargable desde la UI como botón CSV.

### 🧬 Nuevas features
Cuando Ollama está activo, el reporte incluye las columnas generadas por IA, visibles en "Preprocesamiento aplicado" de la UI.

---

## Arquitectura

### Flujo de decisión

```
DATOS CRUDOS → PRE-PARSEO → PERFILADO → ESTRATEGIA → FEATURE ENG. → PIPELINE → AUTO-ML → PREDICCIONES + REPORTE
       │            │            │             │             │           │          │            │
       │            │            └── Plan JSON │             │           │          │            │
       │  (monedas,    (tipos, nulos,  (imputación,        (LLM        (split,     (GridSearch, │
       │   %, inf,    balanceo,      encoding,              local)     encoding,   ensemble)    │
       │   duplicados, correlaciones,  escalado,                       scaling,                │
       │   coercion)   concentración)  modelos)                        balanceo)               │
       │                                                                                       │
       └───────────────────────────────────────────────────────────────────────────────────────┘
                                   REPORTE + PREDICCIONES
```

### Estrategia adaptativa

| Condición | Decisión automática | Por qué |
|---|---|---|
| Target binario con 70/30 | `SMOTE + class_weight` | Desbalanceo moderado |
| Target numérico continuo | `StandardScaler + Regression` | Tipo de tarea |
| **Target con >80% en un valor** | **Advertencia + class_weight o auto-switch** | Concentración extrema |
| **Regresión sobre target discreto** | **Auto-switch a clasificación + balanceo** | El modelo no aprende de valores constantes |
| **Clase minoritaria con 1-2 samples** | **class_weight (sin oversampling)** | SMOTE/ADASYN necesitan ≥6 vecinos |
| Columna numérica con skew > 2 | **RobustScaler + imputación mediana + feature `_log`** | Outliers + relación no-lineal |
| Columna numérica con nulos | **Imputación + indicador `_is_missing`** | El dato faltante puede ser informativo |
| Categórica con >10 valores | **Target encoding** | OneHot sería muy disperso |
| Sin target especificado | `Clustering + PCA + Silhouette` | No supervisado |

---

## Estructura del proyecto

```
Data Science/
├── src/                      # Código fuente del pipeline
│   ├── __init__.py
│   ├── config.py             # Catálogo de modelos, métricas, configuraciones
│   ├── profiler.py           # Perfilado automático del dataset
│   ├── strategist.py         # Estrategia adaptativa + feature engineering por IA
│   ├── pipeline.py           # ETL + pipeline sklearn + transforms seguros
│   ├── trainer.py            # Auto-ML con grid search + cross-validation + ensemble
│   ├── reporter.py           # Reporte ejecutivo con insights del agente
│   ├── llm_client.py         # Adaptador Ollama/OpenAI minimalista
│   └── sample_data.py        # Generador de datasets sintéticos
├── app.py                    # Interfaz gráfica Streamlit
├── cli.py                    # Entry point de línea de comandos
├── tests/                    # Tests unitarios (78+ tests)
├── datasets/                 # Datasets de prueba
├── output/                   # Reportes y predicciones generadas
├── setup.py                  # Instalación como paquete pip
├── Makefile                  # Comandos comunes
├── requirements.txt          # Dependencias congeladas
└── README.md                 # Este archivo
```

---

## Dependencias

`requirements.txt` usa rangos abiertos (`>=`) para compatibilidad máxima. Las versiones específicas se resuelven al instalar:

| Paquete | Rango |
|---|---|
| scikit-learn | >=1.9.0 |
| pandas | >=3.0.0, <4 |
| numpy | >=2.4.0, <3 |
| imbalanced-learn | >=0.12.0 |
| streamlit | >=1.35.0 |
| openpyxl | >=3.1.0 |

Para congelar versiones exactas en producción:
```bash
pip freeze > requirements.lock.txt
```
---

## Tecnologías

- **[scikit-learn](https://scikit-learn.org/)** — Modelos de ML y preprocessing
- **[pandas](https://pandas.pydata.org/)** — Manipulación de datos
- **[imbalanced-learn](https://imbalanced-learn.org/)** — Balanceo de clases (SMOTE, ADASYN)
- **[Streamlit](https://streamlit.io/)** — Interfaz gráfica interactiva
- **[Ollama](https://ollama.com/)** — IA local para insights y feature engineering

---

## Licencia

MIT
