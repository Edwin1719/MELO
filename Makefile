# ─────────────────────────────────────────────────────────────────
# MELO — Comandos comunes
# Compatible con Windows (cmd) y Linux/macOS
# ─────────────────────────────────────────────────────────────────

.PHONY: help install install-full test test-churn test-regression \
        test-clustering clean data output

help:
	@echo "MELO — Comandos disponibles"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Instalacion:"
	@echo "  make install        Instalar dependencias minimas"
	@echo "  make install-full   Instalar todas las dependencias"
	@echo ""
	@echo "Pruebas rapidas:"
	@echo "  make test           Probar clasificacion (churn simple)"
	@echo "  make test-churn     Probar con dataset realista"
	@echo "  make test-regression Probar regresion"
	@echo "  make test-clustering Probar clustering"
	@echo ""
	@echo "Interfaz grafica:"
	@echo "  make run           Iniciar Streamlit app"
	@echo ""
	@echo "Utilidades:"
	@echo "  make data           Generar datasets de prueba"
	@echo "  make output         Ver reportes generados"
	@echo "  make clean          Limpiar archivos generados"
	@echo ""

install:
	pip install -r requirements.txt

install-full:
	pip install -r requirements.txt xgboost joblib

data:
	python sample_data.py

test:
	python cli.py datasets/churn_clientes.csv --target churn --no-npc

test-churn:
	python cli.py datasets/churn_real.csv --target churn --no-npc

test-regression:
	python cli.py datasets/precios_viviendas.csv --target precio --no-npc

test-clustering:
	python cli.py datasets/segmentacion_real.csv --no-npc

run:
	streamlit run app.py --server.port 8501

clean:
	if exist output rmdir /s /q output
	if exist datasets rmdir /s /q datasets
	if exist __pycache__ rmdir /s /q __pycache__
	@echo "Limpieza completada"

output:
	@dir /b output 2>nul || echo No hay reportes aun. Ejecuta: make test
