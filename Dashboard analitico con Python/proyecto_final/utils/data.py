"""
utils/data.py
Carga, limpieza y preparación del dataset Student Performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Student_Performance_Dataset.csv"

# Orden canónico para Performance_Level (de peor a mejor)
PERFORMANCE_ORDER = ["Poor", "Average", "Good", "Excellent"]

# Paleta de colores consistente para niveles de desempeño
PERFORMANCE_COLORS = {
    "Poor":      "#EF4444",   # rojo
    "Average":   "#F59E0B",   # ámbar
    "Good":      "#3B82F6",   # azul
    "Excellent": "#10B981",   # verde
}

PARENTAL_ORDER = ["High School", "Graduate", "Postgraduate"]


def cargar_datos() -> pd.DataFrame:
    """
    Lee el CSV, ajusta tipos y devuelve un DataFrame limpio.
    Convierte columnas categóricas ordenadas a pd.Categorical.
    """
    df = pd.read_csv(DATA_PATH, sep=";")

    # Tipos numéricos (ya vienen bien, pero lo forzamos por seguridad)
    num_cols = [
        "Age", "Study_Hours_Per_Day", "Attendance_Percentage",
        "Math_Score", "Science_Score", "English_Score",
        "Previous_Year_Score", "Final_Percentage",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categóricas ordenadas
    df["Performance_Level"] = pd.Categorical(
        df["Performance_Level"], categories=PERFORMANCE_ORDER, ordered=True
    )
    df["Parental_Education"] = pd.Categorical(
        df["Parental_Education"], categories=PARENTAL_ORDER, ordered=True
    )

    # Columna auxiliar: promedio de las tres materias
    df["Avg_Score"] = df[["Math_Score", "Science_Score", "English_Score"]].mean(axis=1).round(2)

    return df


def aplicar_filtros(
    df: pd.DataFrame,
    generos: list,
    grados: list,
    educacion_padres: list,
    niveles: list,
    rango_horas: list,
    rango_asistencia: list,
) -> pd.DataFrame:
    """
    Filtra el DataFrame según los parámetros enviados por los controles del dashboard.

    Parameters
    ----------
    generos            : lista de valores de Gender
    grados             : lista de valores de Class (int)
    educacion_padres   : lista de valores de Parental_Education
    niveles            : lista de valores de Performance_Level
    rango_horas        : [min, max] de Study_Hours_Per_Day
    rango_asistencia   : [min, max] de Attendance_Percentage
    """
    mask = (
        df["Gender"].isin(generos)
        & df["Class"].isin([int(g) for g in grados])
        & df["Parental_Education"].isin(educacion_padres)
        & df["Performance_Level"].isin(niveles)
        & df["Study_Hours_Per_Day"].between(rango_horas[0], rango_horas[1])
        & df["Attendance_Percentage"].between(rango_asistencia[0], rango_asistencia[1])
    )
    return df[mask].copy()


def kpis(df: pd.DataFrame) -> dict:
    """
    Calcula las métricas clave a partir del DataFrame (ya filtrado).
    Devuelve un dict con los valores formateados para las tarjetas KPI.
    """
    if df.empty:
        return {
            "total":        "0",
            "promedio":     "—",
            "asistencia":   "—",
            "excelentes":   "0 %",
            "top_materia":  "—",
            "mejora":       "—",
        }

    excelentes_pct = (df["Performance_Level"] == "Excellent").mean() * 100

    scores = {
        "Matemáticas": df["Math_Score"].mean(),
        "Ciencias":    df["Science_Score"].mean(),
        "Inglés":      df["English_Score"].mean(),
    }
    top_materia = max(scores, key=scores.get)
    top_valor   = scores[top_materia]

    mejora = (df["Final_Percentage"] - df["Previous_Year_Score"]).mean()

    return {
        "total":       f"{len(df):,}",
        "promedio":    f"{df['Final_Percentage'].mean():.1f} %",
        "asistencia":  f"{df['Attendance_Percentage'].mean():.1f} %",
        "excelentes":  f"{excelentes_pct:.1f} %",
        "top_materia": f"{top_materia} ({top_valor:.1f})",
        "mejora":      f"{mejora:+.1f} pts",
    }
