"""
modelo/entrenar_modelo.py
──────────────────────────
Entrena un RandomForestClassifier para predecir Performance_Level
y guarda el pipeline completo en modelo/modelo.pkl.

Ejecución:
    python modelo/entrenar_modelo.py
"""

import sys
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from utils.data import cargar_datos, PERFORMANCE_ORDER

MODELO_PATH = ROOT / "modelo" / "modelo.pkl"


def preparar_features(df: pd.DataFrame):
    """Devuelve X (features) e y (target) listos para sklearn."""

    FEATURES_NUM = [
        "Age",
        "Study_Hours_Per_Day",
        "Attendance_Percentage",
        "Math_Score",
        "Science_Score",
        "English_Score",
        "Previous_Year_Score",
    ]

    FEATURES_CAT = [
        "Gender",
        "Class",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities",
    ]

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    # Convertir Class a str para el encoder categórico
    X["Class"] = X["Class"].astype(str)

    # Target: codificación ordinal (Poor=0, Average=1, Good=2, Excellent=3)
    le = LabelEncoder()
    le.classes_ = np.array(PERFORMANCE_ORDER)
    y = le.transform(df["Performance_Level"].astype(str))

    return X, y, FEATURES_NUM, FEATURES_CAT, le


def construir_pipeline(features_num, features_cat):
    """Construye el pipeline de preprocesamiento + modelo."""

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), features_num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), features_cat),
    ])

    pipeline = Pipeline(steps=[
        ("preprocesamiento", preprocessor),
        ("modelo", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return pipeline


def entrenar():
    print("=" * 55)
    print("  ENTRENAMIENTO DEL MODELO DE RENDIMIENTO ESTUDIANTIL")
    print("=" * 55)

    # ── Carga y preparación ──────────────────────────────────────
    df = cargar_datos()
    print(f"\n✔  Dataset cargado: {len(df):,} filas, {df.shape[1]} columnas")

    X, y, feat_num, feat_cat, label_encoder = preparar_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"✔  Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras")

    # ── Entrenamiento ────────────────────────────────────────────
    print("\n⏳ Entrenando RandomForestClassifier …")
    pipeline = construir_pipeline(feat_num, feat_cat)
    pipeline.fit(X_train, y_train)

    # ── Evaluación ───────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n✔  Accuracy en test: {acc:.4f} ({acc*100:.2f} %)")
    print("\nReporte de clasificación:")
    print(classification_report(
        y_test, y_pred,
        target_names=PERFORMANCE_ORDER,
    ))

    # Validación cruzada (5-fold)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"✔  CV 5-fold accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Importancia de features
    rf_model    = pipeline.named_steps["modelo"]
    all_features = feat_num + feat_cat
    importancias = pd.Series(rf_model.feature_importances_, index=all_features)
    print("\nTop 7 variables más importantes:")
    print(importancias.sort_values(ascending=False).head(7).to_string())

    # ── Serialización ────────────────────────────────────────────
    artefacto = {
        "pipeline":       pipeline,
        "label_encoder":  label_encoder,
        "features_num":   feat_num,
        "features_cat":   feat_cat,
        "accuracy":       acc,
        "cv_mean":        cv_scores.mean(),
        "cv_std":         cv_scores.std(),
        "performance_order": PERFORMANCE_ORDER,
    }

    joblib.dump(artefacto, MODELO_PATH)
    print(f"\n✔  Modelo guardado en: {MODELO_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    entrenar()
