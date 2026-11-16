
from pathlib import Path
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------

st.set_page_config(
    page_title="Natalidad en Colombia 2015-2025",
    page_icon="📊",
    layout="wide"
)

# Ubicación de las carpetas del proyecto
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"


# --------------------------------------------------
# CARGA DE RESULTADOS
# --------------------------------------------------

@st.cache_data
def cargar_csv(nombre):
    ruta = CARPETA_RESULTADOS / nombre
    return pd.read_csv(ruta, encoding="utf-8-sig")


# --------------------------------------------------
# TÍTULO Y PRESENTACIÓN
# --------------------------------------------------

st.title("Natalidad en Colombia (2015-2025)")

st.subheader(
    "Análisis de la disminución de los nacimientos "
    "y sus características demográficas"
)

st.markdown("""
Esta aplicación presenta los resultados del análisis de los
nacimientos registrados en Colombia entre 2015 y 2025,
a partir de los microdatos del DANE.

Se examinan la evolución anual de los nacimientos y su
distribución según características demográficas.
""")

st.divider()


# --------------------------------------------------
# LECTURA DE LOS CINCO ARCHIVOS
# --------------------------------------------------

try:
    anual = cargar_csv("nacimientos_anuales.csv")
    edad = cargar_csv("nacimientos_por_edad.csv")
    educacion = cargar_csv("nacimientos_por_educacion.csv")
    area = cargar_csv("nacimientos_por_area.csv")
    sexo = cargar_csv("nacimientos_por_sexo.csv")

except FileNotFoundError as error:
    st.error(
        "No se encontró uno de los archivos CSV en la carpeta "
        "'resultados'. Revisa que los cinco archivos estén allí."
    )
    st.stop()


# --------------------------------------------------
# FUNCIONES PARA IDENTIFICAR COLUMNAS
# --------------------------------------------------

def columna_anio(datos):
    for columna in datos.columns:
        if "ANO" in columna.upper() or "AÑO" in columna.upper():
            return columna
    return datos.columns[0]


def columna_numerica(datos, excluir=None):
    excluir = excluir or []
    candidatas = [
        columna for columna in datos.columns
        if columna not in excluir
        and pd.api.types.is_numeric_dtype(datos[columna])
    ]

    if not candidatas:
        return None

    return candidatas[0]


def mostrar_grafica(datos, titulo):
    columna_categoria = datos.columns[0]
    columna_valor = columna_numerica(datos, [columna_categoria])

    if columna_valor is None:
        st.warning(f"No se encontró una columna numérica para {titulo}.")
        st.dataframe(datos, use_container_width=True)
        return

    st.subheader(titulo)
    st.bar_chart(
        datos.set_index(columna_categoria)[columna_valor]
    )


# --------------------------------------------------
# INDICADORES GENERALES
# --------------------------------------------------

col_anio = columna_anio(anual)
col_nacimientos = columna_numerica(anual, [col_anio])

if col_nacimientos is None:
    st.error("No se identificó la columna de nacimientos en el archivo anual.")
    st.stop()

anual[col_anio] = pd.to_numeric(anual[col_anio], errors="coerce")
anual[col_nacimientos] = pd.to_numeric(
    anual[col_nacimientos], errors="coerce"
)

anual = anual.dropna(subset=[col_anio, col_nacimientos])
anual = anual.sort_values(col_anio)

primer_anio = anual.iloc[0]
ultimo_anio = anual.iloc[-1]

total_periodo = anual[col_nacimientos].sum()
nacimientos_iniciales = primer_anio[col_nacimientos]
nacimientos_finales = ultimo_anio[col_nacimientos]

variacion = (
    (nacimientos_finales - nacimientos_iniciales)
    / nacimientos_iniciales
) * 100

st.header("Panorama general")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Nacimientos acumulados",
    f"{total_periodo:,.0f}"
)

c2.metric(
    f"Nacimientos en {int(ultimo_anio[col_anio])}",
    f"{nacimientos_finales:,.0f}"
)

c3.metric(
    "Variación entre primer y último año",
    f"{variacion:.1f}%"
)

st.caption(
    "Los indicadores corresponden a conteos de nacimientos, "
    "no a tasas de fecundidad."
)

st.divider()


# --------------------------------------------------
# EVOLUCIÓN ANUAL
# --------------------------------------------------

st.header("1. Evolución de los nacimientos")

grafica_anual = anual.copy()
grafica_anual[col_anio] = grafica_anual[col_anio].astype(int).astype(str)

st.line_chart(
    grafica_anual.set_index(col_anio)[col_nacimientos],
    x_label="Año",
    y_label="Número de nacimientos"
)
st.dataframe(anual, use_container_width=True)


# --------------------------------------------------
# CARACTERÍSTICAS DEMOGRÁFICAS
# --------------------------------------------------

st.header("2. Características de los nacimientos")

pestanas = st.tabs([
    "Edad de la madre",
    "Nivel educativo",
    "Área de nacimiento",
    "Sexo del nacido"
])

with pestanas[0]:
    mostrar_grafica(edad, "Nacimientos por edad de la madre")
    st.dataframe(edad, use_container_width=True)

with pestanas[1]:
    mostrar_grafica(educacion, "Nacimientos por nivel educativo")
    st.dataframe(educacion, use_container_width=True)

with pestanas[2]:
    mostrar_grafica(area, "Nacimientos por área de nacimiento")
    st.dataframe(area, use_container_width=True)

with pestanas[3]:
    mostrar_grafica(sexo, "Nacimientos por sexo")
    st.dataframe(sexo, use_container_width=True)


# --------------------------------------------------
# 3. MODELO VAR
# --------------------------------------------------

st.divider()
st.header("3. Modelo VAR: análisis y pronóstico")

st.markdown("""
El modelo de Vectores Autorregresivos (VAR) permite analizar
la evolución conjunta de los nacimientos en los grupos de
edad de las madres de 15-19, 20-24 y 25-29 años.
""")

st.subheader("Comparación de modelos")

metricas_var = pd.DataFrame([
    ["15-19 años", "VAR(1)", 304.06, 371.05],
    ["15-19 años", "VAR(2)", 329.95, 395.25],
    ["15-19 años", "VAR(12)", 373.13, 420.40],
    ["15-19 años", "Estacional", 474.00, 498.87],

    ["20-24 años", "VAR(1)", 517.14, 609.11],
    ["20-24 años", "VAR(2)", 571.19, 661.24],
    ["20-24 años", "VAR(12)", 530.02, 586.07],
    ["20-24 años", "Estacional", 530.92, 595.53],

    ["25-29 años", "VAR(1)", 507.63, 621.74],
    ["25-29 años", "VAR(2)", 542.93, 643.94],
    ["25-29 años", "VAR(12)", 493.03, 564.47],
    ["25-29 años", "Estacional", 263.67, 336.57],
], columns=["Grupo de edad", "Modelo", "MAE", "RMSE"])

st.dataframe(metricas_var, width="stretch", hide_index=True)

st.markdown("""
**Interpretación:**

- El MAE mide el error absoluto medio del pronóstico.
- El RMSE penaliza con mayor intensidad los errores grandes.
- VAR(1) presenta un comportamiento competitivo para los
  tres grupos, aunque no es el mejor en todos los casos.
- Se seleccionó VAR(1) como modelo conjunto para alimentar
  la simulación de Monte Carlo.
""")

st.subheader("Diagnósticos del modelo VAR(1)")

col1, col2, col3 = st.columns(3)

col1.metric("Estabilidad", "Sí")
col2.metric("Normalidad", "p = 0.587")
col3.metric("Autocorrelación residual", "p = 0.011")

st.markdown("""
**Interpretación de los diagnósticos:**

El modelo cumple la condición de estabilidad. La prueba de
normalidad no permite rechazar la hipótesis nula al 5 %.

Sin embargo, la prueba de autocorrelación residual sí
evidencia dependencia temporal restante. Por tanto, los
pronósticos deben interpretarse con cautela.
""")

# --------------------------------------------------
# 4. CADENAS DE MARKOV
# --------------------------------------------------

st.divider()
st.header("4. Cadenas de Markov")

st.markdown("""
Las cadenas de Markov permiten estudiar la probabilidad de
que los nacimientos de cada grupo de edad permanezcan en
un mismo estado o cambien a otro durante el mes siguiente.

Los estados utilizados son Bajo, Medio y Alto.
""")

estados = ["Bajo", "Medio", "Alto"]

matrices_markov = {
    "15-19 años": [
        [0.9302, 0.0698, 0.0000],
        [0.0889, 0.8222, 0.0889],
        [0.0000, 0.1163, 0.8837]
    ],
    "20-24 años": [
        [0.8837, 0.1163, 0.0000],
        [0.1364, 0.5909, 0.2727],
        [0.0000, 0.2955, 0.7045]
    ],
    "25-29 años": [
        [0.8140, 0.1628, 0.0233],
        [0.1364, 0.6136, 0.2500],
        [0.0682, 0.2045, 0.7273]
    ]
}


for grupo, valores in matrices_markov.items():
    st.subheader(f"Matriz de transición: {grupo}")

    matriz = pd.DataFrame(
        valores,
        index=estados,
        columns=estados
    )

    # Mostrar la matriz como tabla numérica
    st.table(matriz.style.format("{:.2%}"))

    st.markdown("""
    **Cómo interpretar la matriz:**

    - Las filas representan el estado del mes actual.
    - Las columnas representan el estado del mes siguiente.
    - La diagonal muestra la probabilidad de permanecer
      en el mismo estado.
    """)

    st.subheader(f"Probabilidad de permanencia: {grupo}")

    permanencia = pd.DataFrame({
        "Estado": estados,
        "Probabilidad": [
            valores[0][0],
            valores[1][1],
            valores[2][2]
        ]
    })

    st.bar_chart(
        permanencia,
        x="Estado",
        y="Probabilidad",
        x_label="Estado",
        y_label="Probabilidad"
    )

st.markdown("""
**Interpretación general:**

Los tres grupos presentan probabilidades relativamente
altas de permanecer en el mismo estado.

El grupo de 15-19 años presenta la mayor permanencia
en el estado Bajo (93.02 %). El grupo de 20-24 años
presenta la menor permanencia en el estado Medio (59.09 %).

Estas probabilidades describen la dinámica de los estados
mensuales observados durante el periodo de estudio.
""")


# --------------------------------------------------
# 5. SIMULACION MONTE CARLO
# --------------------------------------------------
st.divider()
st.header("5. Simulación Monte Carlo")

st.markdown("""
La simulación Monte Carlo permite explorar posibles
escenarios de evolución de los nacimientos mediante
5.000 simulaciones y un horizonte de 12 meses.

Se utilizó el modelo VAR(1) y remuestreo de residuos
en bloques de 3, 6 y 12 meses para representar
diferentes estructuras de dependencia temporal.
""")

resultados_mc = {
    "Bloques de 3 meses": [
        ["15-19 años", 4290.27, 4313.78, 1728.27, 6736.35],
        ["20-24 años", 9137.01, 9171.60, 5445.03, 12781.11],
        ["25-29 años", 9966.53, 9994.67, 6705.72, 13121.35]
    ],
    "Bloques de 6 meses": [
        ["15-19 años", 4256.45, 4236.48, 1981.86, 6637.95],
        ["20-24 años", 9107.09, 9078.49, 5803.79, 12374.07],
        ["25-29 años", 9938.38, 9937.93, 7035.94, 12828.55]
    ],
    "Bloques de 12 meses": [
        ["15-19 años", 4256.40, 4258.83, 3073.35, 5693.18],
        ["20-24 años", 9100.45, 9152.95, 7438.96, 11358.78],
        ["25-29 años", 9927.71, 9960.07, 8280.65, 12144.68]
    ]
}

st.subheader("Resultados para diciembre de 2026")

pestanas_mc = st.tabs(list(resultados_mc.keys()))

for pestana, (nombre, datos) in zip(
    pestanas_mc, resultados_mc.items()
):
    with pestana:
        tabla_mc = pd.DataFrame(
            datos,
            columns=[
                "Grupo de edad",
                "Promedio simulado",
                "Mediana",
                "Límite inferior 95%",
                "Límite superior 95%"
            ]
        )

        st.table(
    tabla_mc.style.format({
        "Promedio simulado": "{:,.2f}",
        "Mediana": "{:,.2f}",
        "Límite inferior 95%": "{:,.2f}",
        "Límite superior 95%": "{:,.2f}"
            }),
        )

        st.subheader("Mediana simulada por grupo de edad")

        st.bar_chart(
            tabla_mc,
            x="Grupo de edad",
            y="Mediana",
            x_label="Grupo de edad",
            y_label="Nacimientos simulados"
        )

st.markdown("""
**Interpretación:**

Las medianas simuladas para diciembre de 2026
son relativamente estables entre los tres tamaños
de bloque.

El grupo de 15-19 años presenta los menores valores
simulados, seguido por los grupos de 20-24 y
25-29 años.

Los intervalos del 95 % representan la variabilidad
de los escenarios simulados y cambian según el
tamaño de bloque utilizado.

**Limitaciones:**

En el bloque de 3 meses se obtuvieron seis valores
negativos entre las simulaciones del grupo de
15-19 años. Esto evidencia una limitación del
modelo lineal VAR, que puede producir resultados
negativos aunque los nacimientos no puedan serlo.

Los resultados de Markov se presentan como un
análisis independiente y no están acoplados
a las simulaciones Monte Carlo.
""")


# --------------------------------------------------
# 6. CONCLUSIONES GENERALES
# --------------------------------------------------
st.divider()
st.header("6. Conclusiones generales")

st.markdown("""
### 1. Evolución de los nacimientos en Colombia

Entre 2015 y 2025, los nacimientos registrados en
Colombia disminuyeron de 660.999 a 441.537,
lo que representa una reducción aproximada del 33,2 %.

Este resultado evidencia una disminución importante
de los nacimientos durante el periodo analizado.

### 2. Características demográficas

El análisis por edad de la madre, nivel educativo,
área de nacimiento y sexo del nacido permitió
examinar cómo se distribuyen los nacimientos
según diferentes características demográficas.

Los grupos de edad de 15-19, 20-24 y 25-29 años
presentan comportamientos diferenciados, por lo que
es importante analizarlos de manera independiente.

### 3. Resultados del modelo VAR

Se compararon modelos VAR(1), VAR(2) y VAR(12),
además de un modelo estacional de referencia.

El VAR(1) fue seleccionado como modelo conjunto
para la simulación Monte Carlo. Sin embargo,
no presentó el menor error para todos los grupos
de edad.

Aunque el modelo cumplió la condición de estabilidad,
los diagnósticos identificaron autocorrelación residual.
Por esta razón, sus resultados deben interpretarse con cautela.

### 4. Resultados de las cadenas de Markov

Las matrices de transición mostraron probabilidades
de permanencia relativamente altas en los estados
Bajo, Medio y Alto.

Se destacó la permanencia del grupo de 15-19 años
en el estado Bajo, con una probabilidad del 93,02 %.

Estos resultados permiten describir la continuidad
y los cambios entre estados mensuales.

### 5. Resultados de Monte Carlo

Las 5.000 simulaciones permitieron explorar escenarios
para diciembre de 2026.

Las medianas simuladas fueron:

- 15-19 años: aproximadamente 4.259 nacimientos.
- 20-24 años: aproximadamente 9.153 nacimientos.
- 25-29 años: aproximadamente 9.960 nacimientos.

Estos valores corresponden al escenario de bloques
de 12 meses y no deben interpretarse como cifras
observadas ni como predicciones exactas.

### 6. Limitaciones del estudio

- El análisis utiliza conteos de nacimientos y no tasas
  de fecundidad.
- Los modelos estadísticos presentan incertidumbre
  y no permiten establecer por sí solos relaciones
  causales.
- El modelo VAR conserva autocorrelación residual.
- - Algunas simulaciones produjeron valores negativos,
  una limitación de los modelos lineales aplicados
  a conteos.
- Las cadenas de Markov y Monte Carlo se desarrollaron
  como análisis independientes.

### Conclusión final

El análisis evidencia una reducción significativa
de los nacimientos registrados en Colombia entre
2015 y 2025. La combinación de herramientas
estadísticas permitió estudiar la evolución temporal,
las características demográficas, la transición entre
estados y posibles escenarios futuros.

Los resultados ofrecen una perspectiva cuantitativa
del comportamiento de los nacimientos y constituyen
un aporte para comprender los cambios demográficos
observados durante el periodo de estudio.
""")

# --------------------------------------------------
# 7. FUENTES DE INFORMACIÓN
# --------------------------------------------------

st.divider()
st.header("7. Fuentes de información")

st.markdown("""
**Fuente principal de información:**

Departamento Administrativo Nacional de Estadística (DANE).
Estadísticas Vitales: nacimientos en Colombia, periodo 2015-2025.

**Enlaces de consulta:**

- [DANE - Estadísticas de nacimientos](https://www.dane.gov.co/index.php/estadisticas-por-tema/salud/nacimientos-y-defunciones/nacimientos)

- [DANE - Microdatos de Estadísticas Vitales](https://microdatos.dane.gov.co/index.php/catalog/915)

**Elaboración propia:**

Procesamiento, organización, análisis estadístico y modelación
realizados en Python a partir de los datos de nacimientos.
""")
# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.divider()

st.markdown("""
**Fuente:** Microdatos de nacimientos del Departamento
Administrativo Nacional de Estadística (DANE).

**Periodo de estudio:** 2015-2025.
""")
