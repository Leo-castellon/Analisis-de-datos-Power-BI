"""
utils/helpers.py
Funciones de utilidad para layout y componentes reutilizables del dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html


def kpi_card(titulo: str, valor_id: str, icono: str, color: str = "primary") -> dbc.Card:
    """
    Genera una tarjeta KPI con ícono, título y valor dinámico.

    Parameters
    ----------
    titulo    : texto descriptivo de la métrica
    valor_id  : id del componente html.Span que se actualiza via callback
    icono     : clase Bootstrap Icon (ej: "bi bi-people-fill")
    color     : variante de color Bootstrap
    """
    color_map = {
        "primary":   ("#3B82F6", "#EFF6FF"),
        "success":   ("#10B981", "#ECFDF5"),
        "warning":   ("#F59E0B", "#FFFBEB"),
        "danger":    ("#EF4444", "#FEF2F2"),
        "purple":    ("#8B5CF6", "#F5F3FF"),
        "teal":      ("#14B8A6", "#F0FDFA"),
    }
    icon_color, bg_color = color_map.get(color, color_map["primary"])

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(
                    html.I(className=icono, style={"fontSize": "1.5rem", "color": icon_color}),
                    className="kpi-icon-wrapper",
                    style={
                        "backgroundColor": bg_color,
                        "borderRadius": "12px",
                        "padding": "10px",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "width": "48px",
                        "height": "48px",
                    },
                ),
                html.Div([
                    html.P(titulo, className="kpi-label mb-0",
                           style={"fontSize": "0.75rem", "color": "#6B7280", "fontWeight": "500",
                                  "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.Span("—", id=valor_id,
                              style={"fontSize": "1.6rem", "fontWeight": "700", "color": "#111827"}),
                ], className="ms-3"),
            ], style={"display": "flex", "alignItems": "center"}),
        ]),
        className="shadow-sm h-100",
        style={"borderRadius": "14px", "border": "1px solid #F3F4F6"},
    )


def seccion_titulo(texto: str, subtexto: str = "") -> html.Div:
    """Encabezado de sección con línea decorativa."""
    children = [
        html.H5(texto, className="mb-0", style={"fontWeight": "700", "color": "#111827"}),
    ]
    if subtexto:
        children.append(
            html.P(subtexto, className="mb-0 mt-1", style={"color": "#6B7280", "fontSize": "0.85rem"})
        )
    return html.Div(children, className="mb-3")


def grafico_card(titulo: str, subtitulo: str, grafico_id: str) -> dbc.Card:
    """
    Tarjeta contenedora para un gráfico Plotly con encabezado descriptivo.
    """
    from dash import dcc
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Strong(titulo, style={"color": "#111827", "fontSize": "0.95rem"}),
                html.P(subtitulo, className="mb-0 mt-1",
                       style={"color": "#6B7280", "fontSize": "0.78rem"}),
            ])
        ], style={"backgroundColor": "#FAFAFA", "borderBottom": "1px solid #F3F4F6"}),
        dbc.CardBody(
            dcc.Graph(
                id=grafico_id,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "360px"},
            ),
            className="p-2",
        ),
    ], className="shadow-sm", style={"borderRadius": "14px", "border": "1px solid #F3F4F6"})


def alerta_sin_datos() -> dict:
    """Figura vacía con mensaje cuando no hay datos para el filtro aplicado."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        annotations=[{
            "text": "⚠️ Sin datos para los filtros seleccionados",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 15, "color": "#9CA3AF"},
        }],
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FAFAFA",
    )
    return fig
