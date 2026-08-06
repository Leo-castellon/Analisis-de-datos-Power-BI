"""
dashboard.py
────────────
Dashboard principal de análisis del rendimiento estudiantil.
Tecnologías: Python · Pandas · Plotly · Dash · dash-bootstrap-components · scikit-learn

Ejecución:
    python dashboard.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from utils.data   import cargar_datos, aplicar_filtros, kpis, PERFORMANCE_ORDER, PARENTAL_ORDER
from utils.helpers import kpi_card, grafico_card, seccion_titulo
from utils.graficos import (
    fig_distribucion_desempeno,
    fig_horas_vs_rendimiento,
    fig_materias_por_educacion,
    fig_asistencia_por_nivel,
    fig_radar_perfil,
    fig_heatmap_correlacion,
    fig_factores_binarios,
    fig_prediccion_ml,
)

# ── Inicialización ────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    title="Student Performance Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # para despliegue con gunicorn

# ── Carga inicial de datos ────────────────────────────────────────────────────
DF_GLOBAL = cargar_datos()

MODELO_PATH = ROOT / "modelo" / "modelo.pkl"
MODELO_DISPONIBLE = MODELO_PATH.exists()
artefacto_ml = joblib.load(MODELO_PATH) if MODELO_DISPONIBLE else None


# ═══════════════════════════════════════════════════════════════════════════════
#   LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
def construir_sidebar() -> html.Div:
    """Panel lateral con todos los filtros interactivos."""
    df = DF_GLOBAL

    return html.Div(id="sidebar", children=[
        # ── Logo / título ──────────────────────────────────────────
        html.Div([
            html.I(className="bi bi-mortarboard-fill me-2",
                   style={"fontSize": "1.4rem", "color": "#60A5FA"}),
            html.Div([
                html.H4("Student Performance", className="mb-0"),
                html.P("Dashboard Analítico", className="mb-0"),
            ])
        ], className="sidebar-header d-flex align-items-center"),

        # ── 1. Género (RadioItems) ─────────────────────────────────
        html.Div([
            html.Div("Género", className="sidebar-section-label"),
            dcc.Checklist(
                id="filtro-genero",
                options=[{"label": f" {g}", "value": g} for g in ["Male", "Female"]],
                value=["Male", "Female"],
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
            ),
        ], className="sidebar-section"),

        # ── 2. Grado escolar (Checklist) ───────────────────────────
        html.Div([
            html.Div("Grado escolar", className="sidebar-section-label"),
            dcc.Checklist(
                id="filtro-grado",
                options=[{"label": f" Grado {g}", "value": g}
                         for g in sorted(df["Class"].unique())],
                value=sorted(df["Class"].unique()),
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
            ),
        ], className="sidebar-section"),

        # ── 3. Educación de los padres (Dropdown) ─────────────────
        html.Div([
            html.Div("Educación de los padres", className="sidebar-section-label"),
            dcc.Dropdown(
                id="filtro-educacion",
                options=[{"label": e, "value": e} for e in PARENTAL_ORDER],
                value=PARENTAL_ORDER,
                multi=True,
                placeholder="Seleccionar…",
                style={"fontSize": "0.83rem"},
            ),
        ], className="sidebar-section"),

        # ── 4. Nivel de desempeño (Checklist) ─────────────────────
        html.Div([
            html.Div("Nivel de desempeño", className="sidebar-section-label"),
            dcc.Checklist(
                id="filtro-nivel",
                options=[{"label": f" {n}", "value": n} for n in PERFORMANCE_ORDER],
                value=PERFORMANCE_ORDER,
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
            ),
        ], className="sidebar-section"),

        # ── 5. Horas de estudio (RangeSlider) ─────────────────────
        html.Div([
            html.Div("Horas de estudio / día", className="sidebar-section-label"),
            dcc.RangeSlider(
                id="filtro-horas",
                min=0.5, max=6.0, step=0.5,
                value=[0.5, 6.0],
                marks={i: {"label": str(i), "style": {"color": "#94A3B8", "fontSize": "0.7rem"}}
                       for i in [0.5, 2, 4, 6]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], className="sidebar-section"),

        # ── 6. Asistencia (RangeSlider) ───────────────────────────
        html.Div([
            html.Div("Asistencia (%)", className="sidebar-section-label"),
            dcc.RangeSlider(
                id="filtro-asistencia",
                min=50, max=100, step=5,
                value=[50, 100],
                marks={i: {"label": f"{i}%", "style": {"color": "#94A3B8", "fontSize": "0.7rem"}}
                       for i in [50, 65, 80, 100]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], className="sidebar-section"),

        # ── Footer ────────────────────────────────────────────────
        html.Div([
            html.P("Ciencia de Datos · 2024",
                   style={"color": "#475569", "fontSize": "0.72rem", "textAlign": "center",
                          "marginTop": "12px"}),
        ], className="sidebar-section"),
    ])


def construir_layout() -> dbc.Container:
    """Layout principal: sidebar izquierdo + área de contenido."""
    return dbc.Container(fluid=True, children=[
        dbc.Row([
            # ── SIDEBAR ───────────────────────────────────────────
            dbc.Col(construir_sidebar(), width=12, lg=2, className="px-0"),

            # ── CONTENIDO PRINCIPAL ───────────────────────────────
            dbc.Col(id="main-content", width=12, lg=10, children=[

                # Header
                html.Div([
                    dbc.Row(align="center", children=[
                        dbc.Col([
                            html.H2("📊 Análisis de Rendimiento Estudiantil"),
                            html.P("Visualización interactiva de factores académicos y predicción de desempeño"),
                        ], width=8),
                        dbc.Col([
                            html.Div(id="timestamp",
                                     style={"textAlign": "right", "opacity": "0.8",
                                            "fontSize": "0.85rem"}),
                        ], width=4),
                    ])
                ], className="dashboard-header"),

                # ── KPIs ──────────────────────────────────────────
                dbc.Row(className="kpi-row g-3", children=[
                    dbc.Col(kpi_card("Total Estudiantes",  "kpi-total",      "bi bi-people-fill",       "primary"), width=6, md=4, lg=2),
                    dbc.Col(kpi_card("Promedio Final",     "kpi-promedio",   "bi bi-bar-chart-fill",    "success"), width=6, md=4, lg=2),
                    dbc.Col(kpi_card("Asistencia Media",   "kpi-asistencia", "bi bi-calendar-check-fill","warning"), width=6, md=4, lg=2),
                    dbc.Col(kpi_card("% Excelentes",       "kpi-excelentes", "bi bi-trophy-fill",       "purple"),  width=6, md=4, lg=2),
                    dbc.Col(kpi_card("Mejor Materia",      "kpi-materia",    "bi bi-book-fill",         "teal"),    width=6, md=4, lg=2),
                    dbc.Col(kpi_card("Δ vs Año Anterior",  "kpi-mejora",     "bi bi-graph-up-arrow",    "danger"),  width=6, md=4, lg=2),
                ]),

                # ── FILA 1: Distribución + Scatter ────────────────
                html.Hr(className="section-divider"),
                html.P("Distribución y relación horas–rendimiento", className="section-label"),
                dbc.Row(className="g-3 mb-3", children=[
                    dbc.Col(grafico_card(
                        "Distribución por Nivel de Desempeño",
                        "¿Cómo se distribuye la población estudiantil?",
                        "graf-distribucion",
                    ), width=12, lg=4),
                    dbc.Col(grafico_card(
                        "Horas de Estudio vs. Rendimiento Final",
                        "¿Estudiar más horas garantiza mejor nota?",
                        "graf-horas-rendimiento",
                    ), width=12, lg=8),
                ]),

                # ── FILA 2: Educación padres + Boxplot asistencia ─
                html.P("Factores externos e internos", className="section-label"),
                dbc.Row(className="g-3 mb-3", children=[
                    dbc.Col(grafico_card(
                        "Notas por Materia según Educación de los Padres",
                        "¿La formación familiar impacta el rendimiento académico?",
                        "graf-materias-educacion",
                    ), width=12, lg=7),
                    dbc.Col(grafico_card(
                        "Asistencia por Nivel de Desempeño",
                        "¿La asistencia diferencia a los estudiantes?",
                        "graf-asistencia",
                    ), width=12, lg=5),
                ]),

                # ── FILA 3: Radar + Heatmap ────────────────────────
                html.P("Perfiles y correlaciones", className="section-label"),
                dbc.Row(className="g-3 mb-3", children=[
                    dbc.Col(grafico_card(
                        "Perfil Académico Multidimensional",
                        "¿Cómo difieren los perfiles entre niveles?",
                        "graf-radar",
                    ), width=12, lg=5),
                    dbc.Col(grafico_card(
                        "Mapa de Correlación de Variables",
                        "¿Qué variables se relacionan con el rendimiento final?",
                        "graf-heatmap",
                    ), width=12, lg=7),
                ]),

                # ── FILA 4: Factores binarios ──────────────────────
                html.P("Impacto de factores socioeconómicos", className="section-label"),
                dbc.Row(className="g-3 mb-3", children=[
                    dbc.Col(grafico_card(
                        "Impacto de Factores Binarios en el Promedio Final",
                        "¿Internet, extracurriculares o género hacen diferencia?",
                        "graf-factores",
                    ), width=12),
                ]),

                # ── PREDICTOR ML ──────────────────────────────────
                html.Hr(className="section-divider"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5([
                                html.I(className="bi bi-cpu-fill me-2", style={"color": "#4F46E5"}),
                                "Predictor de Desempeño (Machine Learning)",
                            ]),
                            html.P("Ingresa las características de un estudiante para predecir su nivel de rendimiento.",
                                   style={"color": "#4B5563", "fontSize": "0.87rem"}),
                        ], width=12),
                    ]),
                    dbc.Row(className="g-3 mt-1", children=[
                        dbc.Col([
                            html.Label("Edad", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-age", min=14, max=19, step=1, value=16,
                                       marks={i: str(i) for i in range(14, 20)},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Horas de estudio / día", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-horas", min=0.5, max=6, step=0.5, value=3,
                                       marks={i: str(i) for i in [0.5, 2, 4, 6]},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Asistencia (%)", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-asistencia", min=50, max=100, step=5, value=75,
                                       marks={i: f"{i}%" for i in [50, 65, 80, 100]},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Nota Matemáticas", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-math", min=0, max=100, step=5, value=65,
                                       marks={i: str(i) for i in [0, 25, 50, 75, 100]},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Nota Ciencias", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-science", min=0, max=100, step=5, value=65,
                                       marks={i: str(i) for i in [0, 25, 50, 75, 100]},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Nota Inglés", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-english", min=0, max=100, step=5, value=65,
                                       marks={i: str(i) for i in [0, 25, 50, 75, 100]},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Nota Año Anterior", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Slider(id="pred-prev", min=40, max=95, step=5, value=67,
                                       marks={i: str(i) for i in [40, 55, 70, 85, 95]},
                                       tooltip={"placement": "bottom"}),
                        ], width=12, md=6, lg=3),
                        dbc.Col([
                            html.Label("Género", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Dropdown(id="pred-genero",
                                         options=[{"label": "Male", "value": "Male"},
                                                  {"label": "Female", "value": "Female"}],
                                         value="Male", clearable=False,
                                         style={"fontSize": "0.83rem"}),
                        ], width=6, md=3, lg=2),
                        dbc.Col([
                            html.Label("Grado", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Dropdown(id="pred-grado",
                                         options=[{"label": f"Grado {g}", "value": str(g)}
                                                  for g in [9, 10, 11, 12]],
                                         value="10", clearable=False,
                                         style={"fontSize": "0.83rem"}),
                        ], width=6, md=3, lg=2),
                        dbc.Col([
                            html.Label("Educ. padres", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Dropdown(id="pred-educacion",
                                         options=[{"label": e, "value": e} for e in PARENTAL_ORDER],
                                         value="Graduate", clearable=False,
                                         style={"fontSize": "0.83rem"}),
                        ], width=6, md=3, lg=2),
                        dbc.Col([
                            html.Label("Internet", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Dropdown(id="pred-internet",
                                         options=[{"label": "Sí", "value": "Yes"},
                                                  {"label": "No", "value": "No"}],
                                         value="Yes", clearable=False,
                                         style={"fontSize": "0.83rem"}),
                        ], width=6, md=3, lg=2),
                        dbc.Col([
                            html.Label("Extracurricular", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                            dcc.Dropdown(id="pred-extra",
                                         options=[{"label": "Sí", "value": "Yes"},
                                                  {"label": "No", "value": "No"}],
                                         value="No", clearable=False,
                                         style={"fontSize": "0.83rem"}),
                        ], width=6, md=3, lg=2),
                    ]),
                    dbc.Row(className="mt-3", children=[
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-lightning-charge-fill me-2"), "Predecir Desempeño"],
                                id="btn-predecir",
                                className="btn-predict",
                                disabled=not MODELO_DISPONIBLE,
                            ),
                            html.Div(
                                "⚠️ Modelo no disponible. Ejecuta: python modelo/entrenar_modelo.py",
                                style={"color": "#EF4444", "fontSize": "0.8rem", "marginTop": "8px"},
                            ) if not MODELO_DISPONIBLE else html.Div(),
                        ], width=12, md=4, lg=3),
                        dbc.Col([
                            html.Div(id="resultado-prediccion"),
                        ], width=12, md=8, lg=9),
                    ]),
                    # Gráfico de probabilidades
                    dbc.Row(className="mt-3", children=[
                        dbc.Col(
                            dcc.Graph(id="graf-prediccion-proba",
                                      config={"displayModeBar": False},
                                      style={"height": "280px"}),
                        ),
                    ]) if MODELO_DISPONIBLE else html.Div(),
                ], className="predictor-card mb-4"),

                html.Div(style={"height": "30px"}),  # espaciado final
            ]),
        ]),
    ])


app.layout = construir_layout()


# ═══════════════════════════════════════════════════════════════════════════════
#   CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper interno ────────────────────────────────────────────────────────────
def _filtrar(generos, grados, educacion, niveles, horas, asistencia):
    """Aplica filtros con validación básica."""
    if not generos:   generos   = ["Male", "Female"]
    if not grados:    grados    = DF_GLOBAL["Class"].unique().tolist()
    if not educacion: educacion = PARENTAL_ORDER
    if not niveles:   niveles   = PERFORMANCE_ORDER
    return aplicar_filtros(DF_GLOBAL, generos, grados, educacion, niveles, horas, asistencia)


# ── Callback 1: KPIs ──────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-total",      "children"),
    Output("kpi-promedio",   "children"),
    Output("kpi-asistencia", "children"),
    Output("kpi-excelentes", "children"),
    Output("kpi-materia",    "children"),
    Output("kpi-mejora",     "children"),
    Input("filtro-genero",     "value"),
    Input("filtro-grado",      "value"),
    Input("filtro-educacion",  "value"),
    Input("filtro-nivel",      "value"),
    Input("filtro-horas",      "value"),
    Input("filtro-asistencia", "value"),
)
def actualizar_kpis(generos, grados, educacion, niveles, horas, asistencia):
    df  = _filtrar(generos, grados, educacion, niveles, horas, asistencia)
    k   = kpis(df)
    return k["total"], k["promedio"], k["asistencia"], k["excelentes"], k["top_materia"], k["mejora"]


# ── Callback 2: Todos los gráficos ────────────────────────────────────────────
@app.callback(
    Output("graf-distribucion",    "figure"),
    Output("graf-horas-rendimiento","figure"),
    Output("graf-materias-educacion","figure"),
    Output("graf-asistencia",      "figure"),
    Output("graf-radar",           "figure"),
    Output("graf-heatmap",         "figure"),
    Output("graf-factores",        "figure"),
    Input("filtro-genero",     "value"),
    Input("filtro-grado",      "value"),
    Input("filtro-educacion",  "value"),
    Input("filtro-nivel",      "value"),
    Input("filtro-horas",      "value"),
    Input("filtro-asistencia", "value"),
)
def actualizar_graficos(generos, grados, educacion, niveles, horas, asistencia):
    df = _filtrar(generos, grados, educacion, niveles, horas, asistencia)
    return (
        fig_distribucion_desempeno(df),
        fig_horas_vs_rendimiento(df),
        fig_materias_por_educacion(df),
        fig_asistencia_por_nivel(df),
        fig_radar_perfil(df),
        fig_heatmap_correlacion(df),
        fig_factores_binarios(df),
    )


# ── Callback 3: Predicción ML ─────────────────────────────────────────────────
if MODELO_DISPONIBLE:
    @app.callback(
        Output("resultado-prediccion",    "children"),
        Output("graf-prediccion-proba",   "figure"),
        Input("btn-predecir", "n_clicks"),
        State("pred-age",        "value"),
        State("pred-horas",      "value"),
        State("pred-asistencia", "value"),
        State("pred-math",       "value"),
        State("pred-science",    "value"),
        State("pred-english",    "value"),
        State("pred-prev",       "value"),
        State("pred-genero",     "value"),
        State("pred-grado",      "value"),
        State("pred-educacion",  "value"),
        State("pred-internet",   "value"),
        State("pred-extra",      "value"),
        prevent_initial_call=True,
    )
    def predecir(n_clicks, age, horas, asistencia, math, science, english,
                 prev, genero, grado, educacion, internet, extra):
        if not n_clicks:
            return html.Div(), fig_prediccion_ml({})

        pipeline      = artefacto_ml["pipeline"]
        label_encoder = artefacto_ml["label_encoder"]
        feat_num      = artefacto_ml["features_num"]
        feat_cat      = artefacto_ml["features_cat"]

        fila = pd.DataFrame([{
            "Age":                         age,
            "Study_Hours_Per_Day":         horas,
            "Attendance_Percentage":       asistencia,
            "Math_Score":                  math,
            "Science_Score":               science,
            "English_Score":               english,
            "Previous_Year_Score":         prev,
            "Gender":                      genero,
            "Class":                       str(grado),
            "Parental_Education":          educacion,
            "Internet_Access":             internet,
            "Extracurricular_Activities":  extra,
        }])

        idx_pred = pipeline.predict(fila)[0]
        nivel    = label_encoder.classes_[idx_pred]
        probas   = pipeline.predict_proba(fila)[0]
        proba_dict = {label_encoder.classes_[i]: probas[i] for i in range(len(probas))}

        colores = {
            "Poor": "#EF4444", "Average": "#F59E0B",
            "Good": "#3B82F6", "Excellent": "#10B981",
        }
        conf = proba_dict[nivel] * 100

        resultado_ui = html.Div([
            html.Div([
                html.Span("Nivel predicho: ", style={"fontWeight": "600", "color": "#374151"}),
                html.Span(nivel, className="result-badge ms-2",
                           style={"backgroundColor": colores.get(nivel, "#6B7280"),
                                  "color": "white"}),
                html.Span(f" · Confianza: {conf:.1f}%",
                           style={"color": "#6B7280", "fontSize": "0.9rem", "marginLeft": "10px"}),
            ]),
        ], className="predictor-result")

        return resultado_ui, fig_prediccion_ml(proba_dict)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚀 Iniciando Student Performance Dashboard…")
    if not MODELO_DISPONIBLE:
        print("⚠️  Modelo no encontrado. Ejecuta primero:")
        print("     python modelo/entrenar_modelo.py\n")
    else:
        print(f"✔  Modelo cargado (accuracy: {artefacto_ml['accuracy']:.2%})")
    print("📊 Dashboard disponible en: http://127.0.0.1:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
