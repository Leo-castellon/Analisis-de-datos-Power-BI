# 📊 Student Performance Dashboard

Dashboard analítico interactivo para explorar y predecir el rendimiento académico estudiantil.  
Desarrollado con **Python · Pandas · Plotly · Dash · scikit-learn**.

---

## 🗂️ Estructura del proyecto

```
proyecto_final/
│
├── data/
│   └── Student_Performance_Dataset.csv   # Dataset fuente (5 000 registros)
│
├── modelo/
│   ├── entrenar_modelo.py                # Script de entrenamiento ML
│   └── modelo.pkl                        # Pipeline serializado (se genera al entrenar)
│
├── utils/
│   ├── __init__.py
│   ├── data.py       # Carga, limpieza, filtros y KPIs
│   ├── graficos.py   # Todas las figuras Plotly (8 gráficos)
│   └── helpers.py    # Componentes reutilizables (tarjetas, encabezados)
│
├── assets/
│   └── styles.css    # Tema visual del dashboard
│
├── dashboard.py      # App principal Dash
├── requirements.txt  # Dependencias
└── README.md
```

---

## ⚙️ Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Ejecución

### Paso 1 — Entrenar el modelo (solo la primera vez)
```bash
python modelo/entrenar_modelo.py
```
Esto genera `modelo/modelo.pkl` con el RandomForestClassifier entrenado.

### Paso 2 — Lanzar el dashboard
```bash
python dashboard.py
```
Abre el navegador en **http://127.0.0.1:8050**

---

## 🎛️ Filtros interactivos (6 en total)

| # | Filtro | Tipo de control | Variable |
|---|--------|-----------------|----------|
| 1 | Género | Checklist | `Gender` |
| 2 | Grado escolar | Checklist | `Class` (9–12) |
| 3 | Educación de los padres | Dropdown multi | `Parental_Education` |
| 4 | Nivel de desempeño | Checklist | `Performance_Level` |
| 5 | Horas de estudio/día | RangeSlider | `Study_Hours_Per_Day` |
| 6 | Porcentaje de asistencia | RangeSlider | `Attendance_Percentage` |

---

## 📈 Gráficos y preguntas de negocio

| # | Tipo | Pregunta respondida |
|---|------|---------------------|
| 1 | **Donut** | ¿Cómo se distribuye la población por nivel de desempeño? |
| 2 | **Scatter + tendencia** | ¿Las horas de estudio se relacionan con el rendimiento? |
| 3 | **Barras agrupadas** | ¿La educación de los padres influye en las notas por materia? |
| 4 | **Box plot** | ¿La asistencia diferencia los niveles de desempeño? |
| 5 | **Radar** | ¿Cuál es el perfil académico completo de cada nivel? |
| 6 | **Heatmap correlación** | ¿Qué variables numéricas se correlacionan más con el rendimiento? |
| 7 | **Barras horizontales** | ¿Internet, extracurriculares o género hacen diferencia? |

---

## 🃏 KPIs (6 tarjetas)

| KPI | Métrica |
|-----|---------|
| Total Estudiantes | Conteo de registros filtrados |
| Promedio Final | Media de `Final_Percentage` |
| Asistencia Media | Media de `Attendance_Percentage` |
| % Excelentes | Porcentaje con nivel Excellent |
| Mejor Materia | Materia con mayor promedio + valor |
| Δ vs Año Anterior | Diferencia media Final − Previous_Year_Score |

---

## 🤖 Modelo de Machine Learning

- **Algoritmo:** RandomForestClassifier  
- **Target:** `Performance_Level` (Poor / Average / Good / Excellent)  
- **Features:** 12 variables (numéricas + categóricas)  
- **Pipeline:** StandardScaler → OrdinalEncoder → RandomForest  
- **Validación:** Train/Test 80/20 + Cross-validation 5-fold  
- **Predictor:** Panel interactivo que estima el nivel y muestra probabilidades por clase

---

## 📦 Tecnologías

| Librería | Uso |
|----------|-----|
| `pandas` | Carga, limpieza y transformación de datos |
| `numpy` | Operaciones numéricas auxiliares |
| `plotly` | Generación de gráficos interactivos |
| `dash` | Framework del dashboard web |
| `dash-bootstrap-components` | Layout responsivo (grid, tarjetas, iconos) |
| `scikit-learn` | Pipeline de ML: preprocesamiento y clasificación |
| `joblib` | Serialización del modelo |
