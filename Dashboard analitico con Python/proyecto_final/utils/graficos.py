"""
utils/graficos.py
Funciones que reciben un DataFrame (ya filtrado) y devuelven figuras Plotly.
Cada función responde a una pregunta de negocio específica.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data import PERFORMANCE_ORDER, PERFORMANCE_COLORS, PARENTAL_ORDER
from utils.helpers import alerta_sin_datos

# ── Tema base compartido ────────────────────────────────────────────────────────
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

BASE_LAYOUT = dict(
    font_family=FONT_FAMILY,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=30, b=40, l=50, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    colorway=list(PERFORMANCE_COLORS.values()),
)


# ── 1. Distribución de niveles de desempeño (Donut) ────────────────────────────
def fig_distribucion_desempeno(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿Cómo se distribuye la población estudiantil por nivel de desempeño?
    Tipo: Donut chart
    """
    if df.empty:
        return alerta_sin_datos()

    counts = (
        df["Performance_Level"]
        .value_counts()
        .reindex(PERFORMANCE_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Nivel", "Cantidad"]

    fig = go.Figure(go.Pie(
        labels=counts["Nivel"],
        values=counts["Cantidad"],
        hole=0.55,
        marker_colors=[PERFORMANCE_COLORS[n] for n in counts["Nivel"]],
        textinfo="label+percent",
        textfont_size=12,
        pull=[0.03] * len(counts),
        hovertemplate="<b>%{label}</b><br>Estudiantes: %{value:,}<br>%{percent}<extra></extra>",
    ))

    layout = {**BASE_LAYOUT, "legend": dict(orientation="v", x=0.85, y=0.5)}
    fig.update_layout(
        **layout,
        showlegend=True,
        annotations=[dict(
            text=f"<b>{len(df):,}</b><br><span style='font-size:11px'>estudiantes</span>",
            x=0.5, y=0.5, font_size=14, showarrow=False,
        )],
    )
    return fig


# ── 2. Horas de estudio vs. Porcentaje final (Scatter) ─────────────────────────
def fig_horas_vs_rendimiento(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿Las horas de estudio diario se relacionan con el rendimiento final?
    Tipo: Scatter plot con línea de tendencia por nivel
    """
    if df.empty:
        return alerta_sin_datos()

    fig = px.scatter(
        df,
        x="Study_Hours_Per_Day",
        y="Final_Percentage",
        color="Performance_Level",
        color_discrete_map=PERFORMANCE_COLORS,
        category_orders={"Performance_Level": PERFORMANCE_ORDER},
        opacity=0.55,
        trendline="ols",
        trendline_scope="overall",
        labels={
            "Study_Hours_Per_Day": "Horas de estudio / día",
            "Final_Percentage":    "Porcentaje final (%)",
            "Performance_Level":   "Nivel",
        },
        hover_data={"Math_Score": True, "Science_Score": True, "English_Score": True},
    )

    fig.update_traces(marker_size=5)
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ── 3. Promedio por materia según educación de los padres (Barras agrupadas) ───
def fig_materias_por_educacion(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿La educación de los padres influye en las notas por materia?
    Tipo: Barras agrupadas
    """
    if df.empty:
        return alerta_sin_datos()

    grp = (
        df.groupby("Parental_Education", observed=True)[
            ["Math_Score", "Science_Score", "English_Score"]
        ]
        .mean()
        .round(1)
        .reindex(PARENTAL_ORDER)
        .reset_index()
    )

    materias = {
        "Math_Score":    ("Matemáticas", "#6366F1"),
        "Science_Score": ("Ciencias",    "#F59E0B"),
        "English_Score": ("Inglés",      "#10B981"),
    }

    fig = go.Figure()
    for col, (nombre, color) in materias.items():
        fig.add_trace(go.Bar(
            name=nombre,
            x=grp["Parental_Education"],
            y=grp[col],
            marker_color=color,
            text=grp[col],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{nombre}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="group",
        xaxis_title="Educación de los padres",
        yaxis_title="Promedio de nota",
        yaxis_range=[0, 105],
    )
    return fig


# ── 4. Boxplot: Asistencia por nivel de desempeño ──────────────────────────────
def fig_asistencia_por_nivel(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿La asistencia a clases diferencia los niveles de desempeño?
    Tipo: Box plot
    """
    if df.empty:
        return alerta_sin_datos()

    fig = go.Figure()
    for nivel in PERFORMANCE_ORDER:
        sub = df[df["Performance_Level"] == nivel]["Attendance_Percentage"]
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            y=sub,
            name=nivel,
            marker_color=PERFORMANCE_COLORS[nivel],
            boxmean="sd",
            hovertemplate=f"<b>{nivel}</b><br>Asistencia: %{{y}} %<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        showlegend=False,
        xaxis_title="Nivel de desempeño",
        yaxis_title="Asistencia (%)",
        yaxis_range=[45, 105],
    )
    return fig


# ── 5. Radar: Perfil académico promedio por nivel ──────────────────────────────
def fig_radar_perfil(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿Cuál es el perfil académico completo de cada nivel de desempeño?
    Tipo: Radar / Spider chart
    """
    if df.empty:
        return alerta_sin_datos()

    dimensiones = [
        "Math_Score", "Science_Score", "English_Score",
        "Attendance_Percentage", "Study_Hours_Per_Day",
    ]
    etiquetas = ["Matemáticas", "Ciencias", "Inglés", "Asistencia", "Horas estudio"]

    # Normalizar 0–100 para que sean comparables en la misma escala
    df_norm = df.copy()
    df_norm["Study_Hours_Per_Day"] = df_norm["Study_Hours_Per_Day"] / 6 * 100
    df_norm["Attendance_Percentage"] = df_norm["Attendance_Percentage"]

    fig = go.Figure()
    for nivel in PERFORMANCE_ORDER:
        sub = df_norm[df_norm["Performance_Level"] == nivel]
        if sub.empty:
            continue
        vals = [sub[d].mean() for d in dimensiones]
        vals += [vals[0]]  # cerrar el polígono
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=etiquetas + [etiquetas[0]],
            fill="toself",
            name=nivel,
            line_color=PERFORMANCE_COLORS[nivel],
            opacity=0.65,
            hovertemplate="<b>%{theta}</b>: %{r:.1f}<extra>" + nivel + "</extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont_size=9),
        ),
        showlegend=True,
    )
    return fig


# ── 6. Heatmap de correlación ───────────────────────────────────────────────────
def fig_heatmap_correlacion(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿Qué variables numéricas están más correlacionadas con el rendimiento final?
    Tipo: Heatmap de correlación
    """
    if df.empty or len(df) < 5:
        return alerta_sin_datos()

    cols = [
        "Math_Score", "Science_Score", "English_Score",
        "Attendance_Percentage", "Study_Hours_Per_Day",
        "Previous_Year_Score", "Final_Percentage",
    ]
    etiquetas = [
        "Matemáticas", "Ciencias", "Inglés",
        "Asistencia", "Horas estudio",
        "Nota año anterior", "Porcentaje final",
    ]

    corr = df[cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=etiquetas,
        y=etiquetas,
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont_size=10,
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlación: %{z:.2f}<extra></extra>",
    ))

    heatmap_layout = {**BASE_LAYOUT, "margin": dict(t=10, b=80, l=100, r=20)}
    fig.update_layout(
        **heatmap_layout,
        xaxis_tickangle=-35,
    )
    return fig


# ── 7. Barras horizontales: Impacto de factores binarios en el promedio final ──
def fig_factores_binarios(df: pd.DataFrame) -> go.Figure:
    """
    Pregunta: ¿El acceso a internet y las actividades extracurriculares
              mejoran el rendimiento académico?
    Tipo: Barras horizontales comparativas (lollipop chart)
    """
    if df.empty:
        return alerta_sin_datos()

    factores = {
        "Internet_Access":              "Acceso a internet",
        "Extracurricular_Activities":   "Actividades extracurriculares",
        "Gender":                       "Género (Female vs Male)",
    }

    fig = go.Figure()
    colores_si = "#3B82F6"
    colores_no = "#E5E7EB"

    y_labels, vals_si, vals_no = [], [], []

    for col, label in factores.items():
        if col == "Gender":
            v_si = df[df[col] == "Female"]["Final_Percentage"].mean()
            v_no = df[df[col] == "Male"]["Final_Percentage"].mean()
            label_si, label_no = "Female", "Male"
        else:
            v_si = df[df[col] == "Yes"]["Final_Percentage"].mean()
            v_no = df[df[col] == "No"]["Final_Percentage"].mean()
            label_si, label_no = "Sí", "No"

        y_labels.append(label)
        vals_si.append(round(v_si, 1) if not np.isnan(v_si) else 0)
        vals_no.append(round(v_no, 1) if not np.isnan(v_no) else 0)

    fig.add_trace(go.Bar(
        name="Sí / Female",
        y=y_labels,
        x=vals_si,
        orientation="h",
        marker_color=colores_si,
        text=[f"{v:.1f}%" for v in vals_si],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="No / Male",
        y=y_labels,
        x=vals_no,
        orientation="h",
        marker_color="#F59E0B",
        text=[f"{v:.1f}%" for v in vals_no],
        textposition="outside",
    ))

    factores_layout = {**BASE_LAYOUT, "legend": dict(orientation="h", y=1.1)}
    fig.update_layout(
        **factores_layout,
        barmode="group",
        xaxis_title="Promedio Final (%)",
        yaxis_title="",
        xaxis_range=[0, 110],
    )
    return fig


# ── 8. Predicción ML: distribución de probabilidades del modelo ────────────────
def fig_prediccion_ml(proba_dict: dict) -> go.Figure:
    """
    Recibe un dict {nivel: probabilidad} y muestra barras de probabilidad.
    Tipo: Barras verticales con gradiente de color.
    """
    if not proba_dict:
        return alerta_sin_datos()

    niveles = PERFORMANCE_ORDER
    probs   = [proba_dict.get(n, 0) for n in niveles]
    colores = [PERFORMANCE_COLORS[n] for n in niveles]

    fig = go.Figure(go.Bar(
        x=niveles,
        y=probs,
        marker_color=colores,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Probabilidad: %{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        showlegend=False,
        xaxis_title="Nivel de desempeño",
        yaxis_title="Probabilidad",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.15],
    )
    return fig
