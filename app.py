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
    box.update_traces(boxpoints=False, hoverinfo="skip")  # 🔹 sin etiquetas extra
    return html.Div([dcc.Graph(figure=hist), dcc.Graph(figure=box)])

# ===============================================================
# CONCLUSIONES
# ===============================================================
conclusiones = html.Div([
    html.H3("CONCLUSIONES E INSIGHTS", className="mt-4"),
    html.P("""
        El análisis revela una conexión clara entre el estado físico y hábitos saludables. 
        Las personas con mejor estado físico suelen presentar mayor calidad de sueño, niveles superiores de nutrición 
        y un índice de actividad más elevado, evidenciando la importancia del equilibrio integral entre cuerpo y mente.
    """),
    html.P("""
        Se observa que variables fisiológicas como la frecuencia cardíaca y la presión arterial tienden a mostrar valores 
        más estables en individuos físicamente activos, lo que sugiere un mejor funcionamiento cardiovascular asociado 
        a la constancia en la actividad física y a un descanso adecuado.
    """),
    html.P("""
        Las correlaciones encontradas, aunque moderadas, refuerzan la idea de que el bienestar físico es multifactorial:
        no depende de una sola variable, sino de la interacción entre hábitos, biología y estilo de vida. 
        Esto abre camino a futuras investigaciones que integren dimensiones psicológicas y socioeconómicas.
    """),
    html.P("""
        En síntesis, los resultados permiten inferir que promover hábitos consistentes de sueño y actividad física, 
        junto a una nutrición equilibrada, puede traducirse en una mejora tangible en el estado de salud general. 
        La evidencia sugiere que pequeñas variaciones sostenidas en el tiempo son más determinantes que esfuerzos aislados.
    """),
    html.P("""
        Finalmente, este estudio demuestra cómo el análisis de datos puede convertirse en una herramienta poderosa 
        para entender el bienestar humano, invitando a reflexionar sobre la manera en que la información empírica 
        puede guiar decisiones más saludables, tanto a nivel individual como colectivo.
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
