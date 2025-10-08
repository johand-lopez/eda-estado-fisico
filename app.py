import os
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from scipy import stats

# ===============================================================
# CONFIGURACIÓN INICIAL
# ===============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Dashboard Exploratorio – Estado Físico"

# ===============================================================
# LECTURA DE DATOS
# ===============================================================
url = "https://raw.githubusercontent.com/johand-lopez/eda-estado-fisico/main/fitness_dataset.csv"
df = pd.read_csv(url)

# Copia para conservar datos originales antes de imputar
df_original = df.copy()

# ===============================================================
# ETAPA 1: DETECCIÓN DE NULOS
# ===============================================================
nulos = df.isnull().sum().reset_index()
nulos.columns = ["Variable", "Valores Nulos"]

# ===============================================================
# ETAPA 2: IMPUTACIÓN
# ===============================================================
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

# ===============================================================
# DISTRIBUCIONES PRE Y POST IMPUTACIÓN
# ===============================================================
fig_pre = px.histogram(df_original, x="sleep_hours", nbins=20,
                       title="Horas de sueño (antes de imputación)",
                       color_discrete_sequence=["#5DADE2"])
fig_post = px.histogram(df, x="sleep_hours", nbins=20,
                        title="Horas de sueño (después de imputación)",
                        color_discrete_sequence=["#2874A6"])

# ===============================================================
# PRUEBA KS ENTRE DISTRIBUCIONES PRE Y POST
# ===============================================================
df_pre = df_original["sleep_hours"].dropna()
df_post = df["sleep_hours"]
ks_stat, ks_pvalue = stats.ks_2samp(df_pre, df_post)

# ===============================================================
# PRUEBA DE NORMALIDAD (SHAPIRO-WILK)
# ===============================================================
numeric_vars = df.select_dtypes(include=np.number).columns.tolist()

def prueba_shapiro(variable):
    stat, p = stats.shapiro(df[variable])
    return stat, p

# ===============================================================
# COMPONENTES DEL DASHBOARD
# ===============================================================

# =============================
# 1. CONTEXTO
# =============================
contexto = html.Div([
    html.H3("INTRODUCCIÓN: EXPLORANDO LOS FACTORES DETERMINANTES DEL ESTADO FÍSICO", className="mt-4"),
    html.P("""
        En un mundo donde la salud y el estado físico se han convertido en pilares fundamentales del bienestar personal,
        una pregunta persiste: ¿qué factores realmente determinan que una persona mantenga un estilo de vida activo y saludable?
        Mientras millones de personas establecen metas de fitness cada año, la brecha entre la intención y la acción sigue siendo significativa.
        Este proyecto surge de la curiosidad por descifrar los patrones detrás de esta ecuación compleja.
    """),
    html.H4("UNA MIRADA BASADA EN DATOS"),
    html.P("""
        Utilizando el conjunto de datos "Fitness Classification Dataset", nos embarcamos en un viaje exploratorio para identificar los 
        elementos clave que diferencian a las personas que mantienen rutinas consistentes de aquellas que luchan por establecer hábitos duraderos.
        Este análisis no se centra en prescripciones universales, sino en comprender la diversidad de caminos que llevan al mismo destino: 
        un estilo de vida activo y saludable.
    """),
    html.H4("EL CAMINO DE LA INVESTIGACIÓN"),
    html.Ul([
        html.Li("Patrones de Comportamiento: examinaremos cómo la frecuencia, consistencia y tipos de actividad física se relacionan con el mantenimiento del estado físico."),
        html.Li("Preferencias Personales: investigaremos cómo las elecciones individuales respecto a disciplinas y horarios influyen en la adherencia a largo plazo."),
        html.Li("Factores Contextuales: analizaremos el papel que juegan características demográficas y circunstancias personales en este proceso.")
    ]),
    html.H4("OBJETIVOS"),
    html.P("""
        Al sintetizar estos diferentes aspectos, buscamos crear una imagen multidimensional de lo que significa estar “en forma” en la práctica.
        Los hallazgos de esta investigación pretenden contribuir a una comprensión más matizada y personalizada del bienestar físico,
        reconociendo que cada individuo requiere una combinación única de elementos para alcanzar sus metas de salud.
    """),
    html.P("""
        Este estudio representa un paso hacia la democratización del conocimiento sobre fitness, transformando observaciones de datos
        en información significativa que pueda inspirar a las personas en su viaje personal hacia una vida más activa y saludable.
    """),
    html.Hr(),
    html.H3("CONTEXTO DEL ANÁLISIS"),
    html.P("""
        Este análisis exploratorio de datos (EDA) busca comprender los factores asociados al estado físico de las personas. 
        Se analizan variables fisiológicas (edad, peso, altura, frecuencia cardíaca, presión arterial) y de estilo de vida 
        (sueño, nutrición, actividad física), con el propósito de identificar patrones que expliquen el bienestar general.
    """),
    html.P([
        "Fuente del conjunto de datos: ",
        html.A("Fitness Classification Dataset – Kaggle",
               href="https://www.kaggle.com/datasets",
               target="_blank", style={"color": "#2874A6"})
    ]),
    html.Hr(),
    html.H4("DICCIONARIO DE VARIABLES"),
    dash_table.DataTable(
        data=pd.DataFrame({
            "Variable": [
                "age", "height_cm", "weight_kg", "heart_rate", "blood_pressure",
                "sleep_hours", "nutrition_quality", "activity_index", "is_fit"
            ],
            "Descripción": [
                "Edad", "Altura", "Peso", "Frecuencia cardíaca", "Presión arterial",
                "Horas de sueño", "Calidad nutricional", "Índice de actividad física", "Estado físico (yes/no)"
            ],
            "Unidad": [
                "años", "cm", "kg", "bpm", "mmHg", "horas",
                "escala 1-10", "escala 1-10", "-"
            ],
            "Tipo": [
                "Numérica", "Numérica", "Numérica", "Numérica",
                "Numérica", "Numérica", "Numérica", "Numérica", "Categórica"
            ]
        }).to_dict("records"),
        columns=[{"name": i, "id": i} for i in ["Variable", "Descripción", "Unidad", "Tipo"]],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#2C3E50", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )
])

# =============================
# 2. ETL Y VALIDACIÓN
# =============================
etl = html.Div([
    html.H3("ETL, LIMPIEZA Y VALIDACIÓN ESTADÍSTICA", className="mt-4"),
    html.P("""
        En esta fase se analizó la calidad del conjunto de datos, identificando valores ausentes, duplicados y posibles 
        inconsistencias. Se aplicó una imputación por mediana en la variable 'sleep_hours' para reemplazar los valores faltantes, 
        verificando posteriormente que esta operación no alterara su distribución. 
        Adicionalmente, se evaluó la normalidad de las variables numéricas mediante la prueba de Shapiro-Wilk.
    """),
    html.H4("RESUMEN DE VALORES FALTANTES"),
    dcc.Graph(figure=px.bar(nulos, x="Variable", y="Valores Nulos",
                            color_discrete_sequence=["#2874A6"],
                            title="Resumen de valores faltantes")
              .update_layout(yaxis_title="Valores Nulos")),
    html.Hr(),
    html.H4("DISTRIBUCIÓN ANTES Y DESPUÉS DE LA IMPUTACIÓN"),
    dcc.Graph(figure=fig_pre),
    dcc.Graph(figure=fig_post),
    html.I(f"Prueba KS entre distribuciones pre y post imputación: KS={ks_stat:.3f}, p-value={ks_pvalue:.3f}",
            style={"color": "#34495E"}),
    html.Br(),
    html.P(f"Filas duplicadas: {df.duplicated().sum()}"),
    html.Br(),
    html.H4("PRUEBA DE NORMALIDAD (SHAPIRO–WILK)"),
    html.P("""
        Esta prueba permite determinar si la distribución de una variable numérica se ajusta o no a una distribución normal. 
        Un p-value mayor a 0.05 indica comportamiento normal.
    """),
    dcc.Dropdown(
        id="var_shapiro",
        options=[{"label": v, "value": v} for v in numeric_vars],
        value=numeric_vars[0],
        clearable=False,
        style={"width": "40%"}
    ),
    html.Div(id="resultado_shapiro", className="mt-3", style={"fontWeight": "bold"})
])

@app.callback(
    Output("resultado_shapiro", "children"),
    Input("var_shapiro", "value")
)
def actualizar_shapiro(variable):
    stat, p = prueba_shapiro(variable)
    interpretacion = "Distribución normal" if p > 0.05 else "Distribución no normal"
    return f"Estadístico = {stat:.3f}, p-value = {p:.3f} → {interpretacion}"

# =============================
# 3. ANÁLISIS DESCRIPTIVO
# =============================
analisis = html.Div([
    html.H3("ANÁLISIS DESCRIPTIVO Y RELACIONAL", className="mt-4"),
    html.P("""
        Se examinan las distribuciones de las variables numéricas y su relación con el estado físico. 
        Los gráficos permiten visualizar la dispersión, la simetría y los posibles valores atípicos, 
        así como contrastar patrones entre los grupos 'yes' y 'no'.
    """),
    html.P("Selecciona una variable numérica:"),
    dcc.Dropdown(
        id="var_numerica",
        options=[{"label": v, "value": v} for v in numeric_vars],
        value="age",
        clearable=False,
        style={"width": "40%"}
    ),
    html.Br(),
    html.Div(id="graficos_descriptivos"),
    html.H4("MATRIZ DE CORRELACIÓN", className="mt-4"),
    html.P("Correlaciones entre variables numéricas"),
    dcc.Graph(
        figure=px.imshow(df.corr(numeric_only=True),
                         color_continuous_scale="Blues",
                         height=700, width=900)
    ),
    html.H4("VARIABLES MÁS CORRELACIONADAS CON EL ESTADO FÍSICO", className="mt-4"),
])

# --- Gráfico de correlaciones más altas con 'is_fit' ---
corr_df = df.corr(numeric_only=True)["is_fit"].drop("is_fit").abs().sort_values(ascending=False).reset_index()
corr_df.columns = ["Variable", "Correlación"]
corr_bar = px.bar(corr_df, x="Variable", y="Correlación",
                  color="Correlación", color_continuous_scale="Blues",
                  title="Correlaciones absolutas con el estado físico")
analisis.children.append(dcc.Graph(figure=corr_bar))

@app.callback(
    Output("graficos_descriptivos", "children"),
    Input("var_numerica", "value")
)
def actualizar_graficos(var):
    hist = px.histogram(df, x=var, color="is_fit",
                        barmode="overlay",
                        title=f"Distribución de {var} por estado físico",
                        color_discrete_sequence=["#21618C", "#5DADE2"])
    box = px.box(df, y="is_fit", x=var, color="is_fit",
                 title=f"{var} según estado físico",
                 orientation="h",
                 color_discrete_sequence=["#21618C", "#5DADE2"])
    return html.Div([dcc.Graph(figure=hist), dcc.Graph(figure=box)])

# =============================
# 4. CONCLUSIONES
# =============================
conclusiones = html.Div([
    html.H3("CONCLUSIONES E INSIGHTS", className="mt-4"),
    html.P("""
        El análisis exploratorio muestra que el estado físico de las personas está estrechamente relacionado 
        con hábitos de sueño, calidad de la nutrición y nivel de actividad física. Los individuos con mayor número 
        de horas de sueño y mejor nutrición tienden a presentar indicadores fisiológicos más equilibrados y un mejor estado físico.
    """),
    html.P("""
        También se evidencia que las variables fisiológicas como frecuencia cardíaca y presión arterial tienden a presentar 
        mayores valores promedio en personas con menor condición física, lo cual concuerda con lo esperado desde un punto de vista clínico.
    """),
    html.P("""
        La imputación de valores faltantes en 'sleep_hours' no alteró significativamente la distribución original 
        (p-value elevado en la prueba KS), garantizando consistencia estadística. Además, no se hallaron duplicados 
        ni sesgos notables en el conjunto de datos, reflejando buena calidad en la fuente y un equilibrio entre las 
        clases de la variable objetivo.
    """)
])

# ===============================================================
# LAYOUT PRINCIPAL
# ===============================================================
app.layout = dbc.Container([
    html.Div([
        html.H2("DASHBOARD EXPLORATORIO – ESTADO FÍSICO",
                className="text-center text-white p-3",
                style={"backgroundColor": "#1A5276"}),
        html.P("AUTORES: JOHAN DÍAZ · DAVID MÁRQUEZ",
               className="text-center text-white",
               style={"backgroundColor": "#1A5276", "marginTop": "-15px"})
    ]),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label="Contexto", children=[contexto]),
        dcc.Tab(label="ETL y Validación", children=[etl]),
        dcc.Tab(label="Análisis Descriptivo", children=[analisis]),
        dcc.Tab(label="Conclusiones", children=[conclusiones]),
    ])
], fluid=True)

# ===============================================================
# EJECUCIÓN LOCAL / RENDER
# ===============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)
