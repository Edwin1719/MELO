"""MELO — Pipeline inteligente de Machine Learning."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="melo",
    version="1.0.0",
    description="Pipeline AutoML con ETL inteligente, LLM local y reportes ejecutivos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MELO",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "scikit-learn>=1.9.0",
        "pandas>=3.0.0,<4",
        "numpy>=2.4.0,<3",
        "scipy>=1.17.0,<2",
        "imbalanced-learn>=0.12.0",
        "streamlit>=1.35.0",
        "openpyxl>=3.1.0",
    ],
    extras_require={
        "full": ["xgboost~=3.2.0", "joblib>=1.3"],
        "npc": ["ollama~=0.6.2", "openai"],
        "test": ["pytest>=7"],
    },
    entry_points={
        "console_scripts": [
            "databiq=cli:main",
        ],
    },
)
