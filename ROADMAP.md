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

### 💎 3.2 Reporte HTML exportable

**Problema:** El reporte actual es texto plano. Los interesados de negocio esperan un documento formateado con gráficos, tablas y diseño profesional.

**Solución:** Generar un reporte HTML con:
- Métricas clave con indicadores visuales
- Gráfico de importancia de features
- Tabla de predicciones con paginación
- Estilo profesional (CSS inline para portabilidad)

**Archivos a modificar:**
- `src/reporter.py` — Nueva función `generate_html_report()`

**Esfuerzo:** ~100 líneas | **Riesgo:** Bajo

**Criterio de éxito:** Botón "Descargar reporte (.html)" en la UI que abre un documento con diseño profesional en el navegador.

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
| 6 | Explicabilidad SHAP | 3 | 💎 Medio | ~150 líneas | shap | ⏳ |
| 7 | Reporte HTML | 3 | 💎 Medio | ~100 líneas | Jinja2 | ⏳ |
| 8 | AutoML extendido | 4 | 🧪 Medio | ~200 líneas | Optuna | ⏳ |
| 9 | Plugins de transformación | 4 | 🧪 Bajo | ~80 líneas | Ninguna | ⏳ |

---

## Notas técnicas

- **Prerrequisito transversal:** Todas las mejoras deben ser 100% locales, sin dependencia de cloud.
- **Compatibilidad:** Cada mejora debe funcionar con y sin Ollama. Sin LLM, las funcionalidades de IA se omiten gracefulmente.
- **Tests:** Cada mejora debe incluir tests unitarios o de integración según corresponda.
- **Versiones:** Recordar actualizar `requirements.txt` si se agregan nuevas dependencias.

---

*Última actualización: 2026-07-17*
