# ===============================================================
# DASHBOARD EXPLORATORIO – ESTADO FÍSICO
# Autores: Johan Díaz · David Márquez
# ===============================================================

import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from scipy import stats
import io
import os
from dash.dcc import send_file

# ===============================================================
# 1️⃣ LECTURA DE DATOS
# ===============================================================
df = pd.read_csv("fitness_dataset.csv")

# ===============================================================
# 2️⃣ TRATAMIENTO DE VALORES FALTANTES
# ===============================================================
df_original = df.copy()

if df["sleep_hours"].isna().sum() > 0:
    mediana_sueño = df["sleep_hours"].median()
    df["sleep_hours"] = df["sleep_hours"].fillna(mediana_sueño)

ks_stat, ks_p = stats.ks_2samp(
    df_original["sleep_hours"].dropna(), df["sleep_hours"].dropna()
)

# ===============================================================
# 3️⃣ CONFIGURACIÓN GENERAL DEL DASH
# ===============================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "EDA – Estado Físico"

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

background_color = "#F8F9FA"
header_color = "#1C2E4A"
accent_color = "#2E75B6"
text_color = "#333333"

custom_style = {
    "backgroundColor": background_color,
    "color": text_color,
    "fontFamily": "Segoe UI, sans-serif",
}

tab_style = {
    "padding": "10px",
    "fontWeight": "500",
    "color": text_color,
    "backgroundColor": "#E9ECEF",
}

tab_selected_style = {
    "backgroundColor": accent_color,
    "color": "white",
}

# ===============================================================
# 4️⃣ ENCABEZADO SUPERIOR
# ===============================================================
navbar = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H3("Dashboard Exploratorio – Estado Físico",
                    style={"color": "white", "marginBottom": "0", "fontWeight": "600"}),
            html.Small("Autores: Johan Díaz · David Márquez", style={"color": "#DDE6F1"})
        ])
    ]),
    color=header_color,
    dark=True,
    sticky="top",
)

# ===============================================================
# 5️⃣ PESTAÑAS
# ===============================================================

# --- CONTEXTO ---
tab_contexto = dbc.Card(
    dbc.CardBody([
        html.H4("Contexto del Análisis", style={"color": header_color}),
        html.P("""
        Este análisis exploratorio de datos (EDA) busca comprender los factores asociados al estado físico de las personas.
        Se analizan variables fisiológicas (edad, peso, altura, frecuencia cardíaca, presión arterial) y de estilo de vida
        (sueño, nutrición, actividad física), con el propósito de identificar patrones que expliquen el bienestar general.
        """, style={"textAlign": "justify"}),
        html.P("Fuente del conjunto de datos:"),
        html.A(
            "Fitness Classification Dataset – Kaggle",
            href="https://www.kaggle.com/datasets/muhammedderric/fitness-classification-dataset-synthetic?select=fitness_dataset.csv",
            target="_blank",
            style={"color": accent_color, "fontWeight": "500"}
        ),
        html.Hr(),
        html.H5("Diccionario de Variables", style={"color": accent_color}),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in ["Variable", "Descripción", "Unidad", "Tipo"]],
            data=[
                {"Variable": "age", "Descripción": "Edad", "Unidad": "años", "Tipo": "Numérica"},
                {"Variable": "height_cm", "Descripción": "Altura", "Unidad": "cm", "Tipo": "Numérica"},
                {"Variable": "weight_kg", "Descripción": "Peso", "Unidad": "kg", "Tipo": "Numérica"},
                {"Variable": "heart_rate", "Descripción": "Frecuencia cardíaca", "Unidad": "bpm", "Tipo": "Numérica"},
                {"Variable": "blood_pressure", "Descripción": "Presión arterial", "Unidad": "mmHg", "Tipo": "Numérica"},
                {"Variable": "sleep_hours", "Descripción": "Horas de sueño", "Unidad": "horas", "Tipo": "Numérica"},
                {"Variable": "nutrition_quality", "Descripción": "Calidad nutricional", "Unidad": "escala 1-10", "Tipo": "Numérica"},
                {"Variable": "activity_index", "Descripción": "Índice de actividad física", "Unidad": "escala 1-10", "Tipo": "Numérica"},
                {"Variable": "is_fit", "Descripción": "Estado físico (yes/no)", "Unidad": "-", "Tipo": "Categórica"},
            ],
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_cell={'textAlign': 'left'},
        )
    ])
)

# --- ETL, LIMPIEZA Y NORMALIDAD ---
tab_etl = dbc.Card(
    dbc.CardBody([
        html.H4("ETL, Limpieza y Validación Estadística", style={"color": header_color}),
        html.P("""
        En esta fase se analizó la calidad del conjunto de datos, identificando valores ausentes, duplicados
        y posibles inconsistencias. Se aplicó una imputación por mediana en la variable 'sleep_hours' para
        reemplazar los valores faltantes, verificando posteriormente que esta operación no alterara su distribución.
        Adicionalmente, se evaluó la normalidad de las variables numéricas mediante la prueba de Shapiro-Wilk.
        """, style={"textAlign": "justify"}),
        html.Br(),
        html.H5("Resumen de valores faltantes", style={"color": accent_color}),
        dcc.Graph(
            figure=px.bar(df_original.isna().sum().reset_index(),
                          x="index", y=0,
                          labels={"index": "Variable", "0": "Valores Nulos"},
                          color_discrete_sequence=[accent_color])
        ),
        html.Br(),
        html.H5("Distribución antes y después de la imputación", style={"color": accent_color}),
        dcc.Graph(
            figure=px.histogram(df_original, x="sleep_hours", nbins=20,
                                opacity=0.5, color_discrete_sequence=["#A7C4E0"],
                                title="Horas de sueño (antes de imputación)")
        ),
        dcc.Graph(
            figure=px.histogram(df, x="sleep_hours", nbins=20,
                                opacity=0.5, color_discrete_sequence=["#2E75B6"],
                                title="Horas de sueño (después de imputación)")
        ),
        html.P(f"Prueba KS entre distribuciones pre y post imputación: KS={ks_stat:.3f}, p-value={ks_p:.3f}",
               style={"fontStyle": "italic"}),
        html.Br(),
        html.P(f"Filas duplicadas: {df.duplicated().sum()}"),
        html.Hr(),
        html.H5("Prueba de Normalidad (Shapiro-Wilk)", style={"color": accent_color}),
        html.P("""
        Esta prueba permite determinar si la distribución de una variable numérica se ajusta o no
        a una distribución normal. Un p-value mayor a 0.05 indica comportamiento normal.
        """, style={"textAlign": "justify"}),
        html.Label("Selecciona una variable para aplicar Shapiro-Wilk:"),
        dcc.Dropdown(id="var-shapiro", options=[{"label": c, "value": c} for c in num_cols],
                     value=num_cols[0], style={'width': '50%'}),
        html.Div(id="resultado-shapiro", style={"marginTop": "10px"})
    ])
)

# --- ANÁLISIS DESCRIPTIVO ---
tab_analisis = dbc.Card(
    dbc.CardBody([
        html.H4("Análisis Descriptivo y Relacional", style={"color": header_color}),
        html.P("""
        Se examinan las distribuciones de las variables numéricas y su relación con el estado físico.
        Los gráficos permiten visualizar la dispersión, la simetría y los posibles valores atípicos,
        así como contrastar patrones entre los grupos 'yes' y 'no'.
        """, style={"textAlign": "justify"}),
        html.Label("Selecciona una variable numérica:"),
        dcc.Dropdown(id="num-var", options=[{"label": c, "value": c} for c in num_cols],
                     value=num_cols[0], style={'width': '50%'}),
        dcc.Graph(id="histograma"),
        dcc.Graph(id="boxplot"),
        html.Hr(),
        html.H5("Matriz de Correlación", style={"color": accent_color}),
        dcc.Graph(figure=px.imshow(df[num_cols].corr(),
                                   color_continuous_scale="Blues",
                                   text_auto=True,
                                   title="Correlaciones entre variables numéricas"))
    ])
)

# --- CONCLUSIONES ---
tab_conclusiones = dbc.Card(
    dbc.CardBody([
        html.H4("Conclusiones e Insights", style={"color": header_color}),
        html.P("""
        El análisis exploratorio muestra que el estado físico de las personas está estrechamente relacionado
        con hábitos de sueño, calidad de la nutrición y nivel de actividad física. Los individuos con mayor número
        de horas de sueño y mejor nutrición tienden a presentar indicadores fisiológicos más equilibrados y un
        mejor estado físico.
        """, style={"textAlign": "justify"}),
        html.P("""
        También se evidencia que las variables fisiológicas como frecuencia cardíaca y presión arterial tienden
        a presentar mayores valores promedio en personas con menor condición física, lo cual concuerda con lo
        esperado desde un punto de vista clínico.
        """, style={"textAlign": "justify"}),
        html.P("""
        La imputación de valores faltantes en 'sleep_hours' no alteró significativamente la distribución original
        (p-value elevado en la prueba KS), garantizando consistencia estadística. Además, no se hallaron duplicados
        ni sesgos notables en el conjunto de datos, reflejando buena calidad en la fuente y un equilibrio entre las
        clases de la variable objetivo.
        """, style={"textAlign": "justify"}),
    ])
)

# ===============================================================
# 6️⃣ LAYOUT
# ===============================================================
app.layout = html.Div([
    navbar,
    dbc.Container([
        html.Br(),
        dbc.Tabs([
            dbc.Tab(tab_contexto, label="Contexto", tab_style=tab_style, active_tab_style=tab_selected_style),
            dbc.Tab(tab_etl, label="ETL y Validación", tab_style=tab_style, active_tab_style=tab_selected_style),
            dbc.Tab(tab_analisis, label="Análisis Descriptivo", tab_style=tab_style, active_tab_style=tab_selected_style),
            dbc.Tab(tab_conclusiones, label="Conclusiones", tab_style=tab_style, active_tab_style=tab_selected_style),
        ])
    ], fluid=True, style=custom_style)
])

# ===============================================================
# 7️⃣ CALLBACKS
# ===============================================================
@app.callback(
    [Output("histograma", "figure"),
     Output("boxplot", "figure")],
    [Input("num-var", "value")]
)
def actualizar_graficos(variable):
    fig1 = px.histogram(df, x=variable, nbins=30, color="is_fit",
                        color_discrete_sequence=["#2E75B6", "#A7C4E0"],
                        title=f"Distribución de {variable} por estado físico")
    fig2 = px.box(df, x="is_fit", y=variable, color="is_fit",
                  color_discrete_sequence=["#2E75B6", "#A7C4E0"],
                  title=f"{variable} según estado físico")
    return fig1, fig2


@app.callback(
    Output("resultado-shapiro", "children"),
    [Input("var-shapiro", "value")]
)
def prueba_shapiro(variable):
    stat, p = stats.shapiro(df[variable].dropna())
    resultado = "Distribución normal" if p > 0.05 else "Distribución no normal"
    return html.P(f"Estadístico = {stat:.3f}, p-value = {p:.3f} → {resultado}")

@app.server.route("/download_csv")
def download_csv():
    return send_file(io.BytesIO(df.to_csv(index=False).encode()),
                     mimetype="text/csv",
                     download_name="fitness_dataset_limpio.csv",
                     as_attachment=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
