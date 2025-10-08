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

df_original = df.copy()

# ===============================================================
# DETECCIÓN DE NULOS E IMPUTACIÓN
# ===============================================================
nulos = df.isnull().sum().reset_index()
nulos.columns = ["Variable", "Valores Nulos"]
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
# PRUEBAS ESTADÍSTICAS
# ===============================================================
df_pre = df_original["sleep_hours"].dropna()
df_post = df["sleep_hours"]
ks_stat, ks_pvalue = stats.ks_2samp(df_pre, df_post)

numeric_vars = df.select_dtypes(include=np.number).columns.tolist()

def prueba_shapiro(variable):
    stat, p = stats.shapiro(df[variable])
    return stat, p

# ===============================================================
# CONTEXTO
# ===============================================================
contexto = html.Div([
    html.H3("INTRODUCCIÓN: EXPLORANDO LOS FACTORES DETERMINANTES DEL ESTADO FÍSICO", className="mt-4"),
    html.P("""
        En un mundo donde la salud y el estado físico se han convertido en pilares fundamentales del bienestar personal,
        una pregunta persiste: ¿qué factores realmente determinan que una persona mantenga un estilo de vida activo y saludable?
        Este proyecto busca descifrar los patrones que diferencian a quienes logran mantener su bienestar de quienes enfrentan
        dificultades para hacerlo.
    """),
    html.H4("UNA MIRADA BASADA EN DATOS"),
    html.P("""
        A partir del conjunto de datos "Fitness Classification Dataset", se realiza un análisis exploratorio para identificar
        relaciones significativas entre variables fisiológicas y hábitos de vida. El objetivo es comprender cómo la edad,
        la nutrición, la actividad y el descanso configuran el estado físico.
    """),
    html.H4("EL CAMINO DE LA INVESTIGACIÓN"),
    html.Ul([
        html.Li("Patrones de Comportamiento: observar la relación entre frecuencia de actividad y condición física."),
        html.Li("Preferencias Personales: analizar cómo hábitos de sueño y nutrición impactan el bienestar."),
        html.Li("Factores Contextuales: entender la influencia de variables demográficas en el rendimiento físico.")
    ]),
    html.H4("OBJETIVOS"),
    html.P("""
        Este estudio pretende generar una comprensión integral de lo que significa “estar en forma”, 
        valorando la interacción entre cuerpo, descanso y hábitos saludables. Los resultados aspiran a ofrecer
        una perspectiva que motive decisiones basadas en evidencia hacia una vida activa y equilibrada.
    """),
    html.Hr(),
    html.H3("CONTEXTO DEL ANÁLISIS"),
    html.P("""
        Se analizan variables fisiológicas (edad, peso, altura, frecuencia cardíaca, presión arterial) y de estilo de vida 
        (sueño, nutrición, actividad física), buscando relaciones que expliquen diferencias en el estado físico reportado.
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
                "Horas de sueño", "Calidad nutricional", "Índice de actividad física", "Estado físico (1 = bueno)"
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

# ===============================================================
# ETL Y VALIDACIÓN
# ===============================================================
etl = html.Div([
    html.H3("ETL, LIMPIEZA Y VALIDACIÓN ESTADÍSTICA", className="mt-4"),
    html.P("""
        Se identificaron y gestionaron los valores ausentes mediante imputación por mediana, verificando su estabilidad estadística.
        También se aplicaron pruebas de normalidad para comprender la naturaleza de las distribuciones numéricas.
    """),
    html.H4("RESUMEN DE VALORES FALTANTES"),
    dcc.Graph(figure=px.bar(nulos, x="Variable", y="Valores Nulos",
                            color_discrete_sequence=["#2874A6"],
                            title="Valores faltantes por variable")
              .update_layout(yaxis_title="Valores Nulos")),
    html.Hr(),
    html.H4("DISTRIBUCIÓN ANTES Y DESPUÉS DE LA IMPUTACIÓN"),
    dcc.Graph(figure=fig_pre),
    dcc.Graph(figure=fig_post),
    html.I(f"Prueba KS entre distribuciones: KS={ks_stat:.3f}, p-value={ks_pvalue:.3f}",
            style={"color": "#34495E"}),
    html.Br(),
    html.P(f"Filas duplicadas: {df.duplicated().sum()}"),
    html.Br(),
    html.H4("PRUEBA DE NORMALIDAD (SHAPIRO–WILK)"),
    html.P("Permite determinar si una variable numérica sigue una distribución normal (p > 0.05 indica normalidad)."),
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

# ===============================================================
# ANÁLISIS DESCRIPTIVO
# ===============================================================
corr_df = df.corr(numeric_only=True)["is_fit"].drop("is_fit").abs().sort_values(ascending=False).reset_index()
corr_df.columns = ["Variable", "Correlación"]

analisis = html.Div([
    html.H3("ANÁLISIS DESCRIPTIVO Y RELACIONAL", className="mt-4"),
    html.P("""
        Se examinan las relaciones entre las variables numéricas y el estado físico, 
        visualizando cómo cada factor contribuye al bienestar general.
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
    html.Div([
        dcc.Graph(
            figure=px.imshow(df.corr(numeric_only=True),
                             color_continuous_scale="Blues",
                             height=700, width=900)
        )
    ], style={"display": "flex", "justifyContent": "center"}),
    html.H4("VARIABLES MÁS CORRELACIONADAS CON EL ESTADO FÍSICO", className="mt-4"),
    dcc.Graph(figure=px.bar(corr_df, x="Variable", y="Correlación",
                            color="Correlación", color_continuous_scale="Blues",
                            title="Correlaciones absolutas con is_fit"))
])

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
    box.update_traces(boxpoints=False, hoverinfo="skip")
    return html.Div([dcc.Graph(figure=hist), dcc.Graph(figure=box)])

# ===============================================================
# CONCLUSIONES E INSIGHTS
# ===============================================================
conclusiones = html.Div([
    html.H3("CONCLUSIONES E INSIGHTS", className="mt-4"),

    html.H4("1. Hallazgos Clave"),
    html.P("""
        El análisis confirma que el estado físico está fuertemente asociado con la calidad de los hábitos cotidianos.
        Las personas físicamente activas presentan mayores horas de sueño, mejor nutrición y menor frecuencia cardíaca promedio.
        A nivel estadístico, se observan correlaciones significativas entre el índice de actividad y la calidad nutricional con el estado físico.
    """),

    html.H4("2. Interpretación de Resultados"),
    html.P("""
        Estos hallazgos refuerzan la premisa de que la condición física es un fenómeno multifactorial.
        No existe una sola variable que determine el bienestar, sino la interacción entre descanso, alimentación y actividad.
        La estabilidad de la distribución tras la imputación de valores faltantes demuestra que el tratamiento de los datos 
        fue adecuado y que las conclusiones derivadas son confiables.
    """),
    html.P("""
        Asimismo, los resultados sugieren que el sueño adecuado actúa como un regulador clave del rendimiento fisiológico.
        La evidencia empírica obtenida coincide con literatura previa que vincula el descanso con el equilibrio hormonal,
        la reparación muscular y la regulación metabólica.
    """),

    html.H4("3. Implicaciones y Reflexión Final"),
    html.P("""
        Este estudio evidencia la utilidad del análisis exploratorio como punto de partida para investigaciones más complejas
        en el ámbito de la salud y el comportamiento humano. Comprender los factores que inciden en el estado físico permite
        orientar intervenciones personalizadas que fomenten hábitos sostenibles.
    """),
    html.P("""
        En términos prácticos, los resultados pueden apoyar el diseño de programas de bienestar integrales,
        donde la recomendación no se limite a “hacer ejercicio”, sino que contemple el balance entre descanso,
        nutrición y actividad moderada. 
    """),
    html.P("""
        Finalmente, se reafirma que la ciencia de datos es una herramienta poderosa para traducir patrones invisibles
        en conocimiento aplicable. La integración de evidencia cuantitativa con interpretación contextual
        fortalece la capacidad de decisión en políticas públicas, salud preventiva y bienestar individual.
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
