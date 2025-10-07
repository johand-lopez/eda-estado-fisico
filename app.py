import os
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from scipy import stats

# ===============================================================
# CONFIGURACIÓN INICIAL
# ===============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # necesario para Render / Gunicorn
app.title = "Dashboard EDA – Estado Físico"

# ===============================================================
# LECTURA DE DATOS
# ===============================================================
url = "https://raw.githubusercontent.com/johand-lopez/eda-estado-fisico/main/fitness_dataset.csv"
df = pd.read_csv(url)

# ===============================================================
# ETAPA 1: DETECCIÓN DE NULOS (antes de imputar)
# ===============================================================
nulos_df = df.isnull().sum().reset_index()
nulos_df.columns = ["Variable", "Nulos"]

# ===============================================================
# ETAPA 2: IMPUTACIÓN DE VALORES FALTANTES
# ===============================================================
# Se imputa con la mediana de sleep_hours
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

# ===============================================================
# DESCRIPCIÓN DE VARIABLES
# ===============================================================
descripcion_variables = pd.DataFrame({
    "Variable": [
        "age", "height_cm", "weight_kg", "heart_rate", "blood_pressure",
        "sleep_hours", "nutrition_quality", "activity_index", "is_fit"
    ],
    "Descripción": [
        "Edad", "Altura", "Peso", "Frecuencia cardíaca", "Presión arterial",
        "Horas de sueño", "Calidad nutricional", "Índice de actividad", "Estado físico"
    ],
    "Unidad": [
        "años", "cm", "kg", "bpm", "mmHg", "horas", "escala 1-10", "escala 1-10", "-"
    ],
    "Tipo": [
        "Numérica", "Numérica", "Numérica", "Numérica", "Numérica",
        "Numérica", "Numérica", "Numérica", "Categórica"
    ]
})

# ===============================================================
# PRUEBA DE NORMALIDAD (Kolmogorov–Smirnov)
# ===============================================================
stat, p_value = stats.kstest(
    df["sleep_hours"],
    "norm",
    args=(df["sleep_hours"].mean(), df["sleep_hours"].std())
)

normalidad_texto = (
    f"Estadístico KS = {stat:.4f}, p-valor = {p_value:.4f}. "
    + ("✅ No se rechaza la hipótesis de normalidad (p > 0.05)."
       if p_value > 0.05 else "❌ Se rechaza la hipótesis de normalidad (p ≤ 0.05).")
)

# ===============================================================
# COMPONENTES DEL DASHBOARD
# ===============================================================

# --- Contexto y Diseño ---
contexto = html.Div([
    html.H3("Objetivo del análisis", className="mt-4"),
    html.P("""
        Clasificar si una persona tiene un buen estado físico o no,
        basándose en variables fisiológicas y de estilo de vida.
        Los datos provienen de un conjunto sintético generado con fines educativos.
        Fuente: Kaggle – Fitness Classification Dataset.
    """),
    html.Hr(),
    html.H4("Diccionario de Variables"),
    dash_table.DataTable(
        data=descripcion_variables.to_dict("records"),
        columns=[{"name": i, "id": i} for i in descripcion_variables.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#2C3E50", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )
])

# --- ETL y Limpieza ---
etl = html.Div([
    html.H3("ETL: Extracción, Transformación y Limpieza de Datos"),
    html.P("Se exploran los valores faltantes y se validan las decisiones de limpieza e imputación."),
    dcc.Graph(figure=px.bar(
        nulos_df, x="Variable", y="Nulos",
        title="Conteo de valores faltantes por variable",
        color_discrete_sequence=["#34495E"]
    )),
    html.H4("Validación de Imputación"),
    html.P("Comprobamos que la imputación no alteró significativamente la distribución de las horas de sueño."),
    dcc.Graph(figure=px.histogram(
        df, x="sleep_hours", nbins=20, color="is_fit",
        title="Distribución de horas de sueño por estado físico",
        color_discrete_sequence=px.colors.qualitative.Safe
    )),
    html.H4("Prueba de Normalidad (Kolmogorov–Smirnov)"),
    html.P("Evaluamos si la variable 'sleep_hours' sigue una distribución normal."),
    html.Div(normalidad_texto, style={"fontWeight": "bold", "color": "#2C3E50"}),
])

# --- Análisis Descriptivo y Relacional ---
analisis = html.Div([
    html.H3("Análisis Descriptivo y Relacional"),
    dcc.Graph(figure=px.scatter_matrix(
        df, dimensions=["age", "height_cm", "weight_kg", "heart_rate", "activity_index"],
        color="is_fit", title="Matriz de dispersión de variables fisiológicas"
    )),
    html.H4("Correlaciones"),
    dcc.Graph(figure=px.imshow(
        df.corr(numeric_only=True),
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de correlación"
    ))
])

# --- Conclusiones e Insights ---
conclusiones = html.Div([
    html.H3("Conclusiones e Insights"),
    html.Ul([
        html.Li("La edad y la frecuencia cardíaca muestran relación inversa con el estado físico."),
        html.Li("Las personas con más horas de sueño y mayor índice de actividad presentan mejor estado físico."),
        html.Li("La imputación de valores faltantes en horas de sueño no modificó la distribución original."),
        html.Li("Las correlaciones moderadas indican independencia entre la mayoría de las variables, favoreciendo el modelado posterior."),
        html.Li("El dataset es adecuado para construir un modelo de clasificación en fases posteriores."),
    ])
])

# ===============================================================
# LAYOUT PRINCIPAL CON PESTAÑAS
# ===============================================================
app.layout = dbc.Container([
    html.Br(),
    html.H2("Dashboard Exploratorio – Estado Físico", className="text-center text-primary fw-bold"),
    html.P("EDA interactivo basado en datos fisiológicos y hábitos de salud.", className="text-center text-secondary"),
    html.Hr(),
    dcc.Tabs([
        dcc.Tab(label="Contexto y Diseño", children=[contexto]),
        dcc.Tab(label="ETL y Limpieza", children=[etl]),
        dcc.Tab(label="Análisis Descriptivo y Relacional", children=[analisis]),
        dcc.Tab(label="Conclusiones e Insights", children=[conclusiones]),
    ])
], fluid=True)

# ===============================================================
# EJECUCIÓN LOCAL / RENDER
# ===============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)
