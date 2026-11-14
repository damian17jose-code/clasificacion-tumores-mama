import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------- Configuración de la página ----------
st.set_page_config(page_title="Clasificación de tumores de mama", layout="wide", initial_sidebar_state="collapsed")

ROJO, VERDE, AZUL, GRIS = "#b3261e", "#1b7f5c", "#1f4e9c", "#8a94a6"   # paleta única para toda la app

# ---------- Estilos (CSS) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp {
    font-family: 'Inter', sans-serif;
}footer {visibility: hidden;}
.stApp {background: #f5f7fb;}
.block-container {padding-top: 1.5rem; max-width: 1200px;}

/* Encabezado */
.hero {background: #14213d; color: white; padding: 1.8rem 2.2rem; border-radius: 12px;
       margin-bottom: 1.5rem; border-left: 6px solid #1f4e9c;}
.hero h1 {margin: 0; font-size: 1.9rem; font-weight: 700; color: white; letter-spacing: -.02em;}
.hero p {margin: .4rem 0 0 0; opacity: .85; font-size: 1rem;}

/* Barra lateral */
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e6e9f2;}
[data-testid="stSidebar"] h2 {font-weight: 700;}

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {gap: 8px; border-bottom: none; background: #e9edf7;
                                   padding: 6px; border-radius: 10px;}
.stTabs [data-baseweb="tab"] {height: 44px; border-radius: 10px; padding: 0 18px; font-weight: 600;
                              background: transparent;}
.stTabs [aria-selected="true"] {background: white !important; color: #1f4e9c !important;
                                box-shadow: 0 2px 8px rgba(0,0,0,.08);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* Botones */
div.stButton > button, div.stDownloadButton > button {border-radius: 12px; font-weight: 600;
    border: 1px solid #d5dbeb; background: white; padding: .6rem 1rem; transition: all .15s ease;}
div.stButton > button:hover, div.stDownloadButton > button:hover {border-color: #1f4e9c; color: #1f4e9c;
    transform: translateY(-2px); box-shadow: 0 6px 14px rgba(31,78,156,.15);}

/* Campos numéricos */
[data-testid="stNumberInput"] input {border-radius: 10px;}
[data-testid="stNumberInput"] label p {font-weight: 600; font-size: .85rem;}

/* Tarjetas */
.resultado {background: white; border-radius: 12px; padding: 1.4rem 1.6rem;
            box-shadow: 0 6px 24px rgba(30, 40, 90, .08);}
.res-modelo {font-size: .8rem; color: #6b7390; text-transform: uppercase; letter-spacing: .08em; font-weight: 600;}
.res-texto {font-size: 2.6rem; font-weight: 700; line-height: 1.1; letter-spacing: -.02em;}
.res-prob {font-size: 1.05rem; margin-top: .2rem; color: #3a4060;}
.barra {position: relative; height: 14px; border-radius: 8px; margin: 1.2rem 0 .3rem 0;
        background: linear-gradient(90deg, #1b7f5c 0%, #d9b84a 50%, #b3261e 100%);}
.marca {position: absolute; top: -5px; width: 6px; height: 24px; background: #222;
        border-radius: 3px; border: 2px solid white; box-shadow: 0 2px 6px rgba(0,0,0,.3);}
.escala {display: flex; justify-content: space-between; font-size: .75rem; color: #8a94a6;}
.kpi {background: white; border-radius: 16px; padding: 1rem 1.1rem; text-align: center;
      box-shadow: 0 4px 16px rgba(30, 40, 90, .07);}
.kpi .valor {font-size: 1.8rem; font-weight: 700; color: #14213d;}
.kpi .nombre {font-size: .8rem; color: #6b7390; font-weight: 500;}
h2, h3 {font-weight: 700 !important; letter-spacing: -.01em;}
</style>
""", unsafe_allow_html=True)


# ---------- Carga de los archivos guardados ----------
@st.cache_resource   # carga una sola vez, no en cada clic
def cargar_archivos():
    modelos = joblib.load("modelos.joblib")
    scaler = joblib.load("scaler.joblib")
    variables = joblib.load("variables.joblib")
    X_test, y_test = joblib.load("ejemplos.joblib")
    rangos = joblib.load("rangos.joblib")
    resultados = joblib.load("resultados.joblib")
    return modelos, scaler, variables, X_test, y_test, rangos, resultados


try:
    modelos, scaler, variables, X_test, y_test, rangos, resultados = cargar_archivos()
except FileNotFoundError:
    st.error("Faltan archivos guardados. Ejecuta de nuevo la celda del Colab que guarda los archivos y reinicia la aplicación.")
    st.stop()

# Unifico los nombres de los modelos (en el notebook uno dice "logística" y otro "Logística")
_mapa = {k.lower(): k for k in modelos}
resultados = {_mapa.get(k.lower(), k): v for k, v in resultados.items()}


# ---------- Nombres amigables para las 30 variables ----------
NOMBRES = {"radius": "Radio", "texture": "Textura", "perimeter": "Perímetro", "area": "Área",
           "smoothness": "Suavidad", "compactness": "Compacidad", "concavity": "Concavidad",
           "concave_points": "Puntos cóncavos", "symmetry": "Simetría", "fractal_dimension": "Dimensión fractal"}
GRUPOS = {"1": "Valor promedio", "2": "Error estándar", "3": "Peor valor"}   # el dataset trae las 30 medidas en 3 bloques de 10


def etiqueta(v, con_grupo=True):
    m = re.match(r"(.+?)(\d)$", v)   # separa "radius1" en ("radius", "1")
    if not m or m.group(1) not in NOMBRES:
        return v
    base, grupo = NOMBRES[m.group(1)], GRUPOS.get(m.group(2), "")
    return f"{base} ({grupo.lower()})" if con_grupo else base


def grupo_de(v):
    m = re.search(r"(\d)$", v)
    return m.group(1) if m else "1"


# ---------- Funciones ----------
def predecir(nombre_modelo, tabla):
    escalado = scaler.transform(tabla[variables])
    modelo = modelos[nombre_modelo]
    return modelo.predict(escalado), modelo.predict_proba(escalado)[:, 1]


def variables_fuera_de_rango(valores):
    return [v for v in variables if valores[v] < rangos.loc["min", v] or valores[v] > rangos.loc["max", v]]


TIPO_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def a_excel(tabla):
    buffer = io.BytesIO()
    tabla.to_excel(buffer, index=False, sheet_name="Datos")
    return buffer.getvalue()


def kpi(col, nombre, valor):
    col.markdown(f'<div class="kpi"><div class="valor">{valor}</div><div class="nombre">{nombre}</div></div>',
                 unsafe_allow_html=True)


def tarjeta_resultado(modelo, maligno, proba):
    color = ROJO if maligno else VERDE
    texto = "MALIGNO" if maligno else "BENIGNO"
    st.markdown(f"""
    <div class="resultado" style="border-left: 8px solid {color};">
      <div class="res-modelo">{modelo}</div>
      <div class="res-texto" style="color:{color};">{texto}</div>
      <div class="res-prob">Probabilidad de maligno: <b>{proba:.1f}%</b></div>
      <div class="barra"><div class="marca" style="left: calc({proba:.1f}% - 2px);"></div></div>
      <div class="escala"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>""", unsafe_allow_html=True)


def figura(ancho, alto):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="both", alpha=.15)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    return fig, ax


# ---------- Valores iniciales y casos de ejemplo ----------
promedios = X_test.mean()
caso_benigno = X_test[y_test == 0].iloc[0]
caso_maligno = X_test[y_test == 1].iloc[0]


def cargar_promedios():
    for v in variables:
        st.session_state[v] = float(promedios[v])
    st.session_state["ejemplo"] = None


def cargar_caso(fila, diagnostico):
    valores = {v: float(fila[v]) for v in variables}
    for v in variables:
        st.session_state[v] = valores[v]
    st.session_state["ejemplo"] = (valores, diagnostico)


if "inicializado" not in st.session_state:
    cargar_promedios()
    st.session_state["inicializado"] = True


# ---------- Encabezado ----------
st.markdown("""
<div class="hero">
  <h1>Clasificación de tumores de mama</h1>
  <p>Comparación de cuatro modelos de clasificación sobre el dataset Breast Cancer Wisconsin (Diagnostic).</p>
</div>
""", unsafe_allow_html=True)

col_modelo, col_aviso = st.columns([1, 2], gap="large")
with col_modelo:
    modelo_elegido = st.selectbox("Modelo de clasificación", list(modelos.keys()))
with col_aviso:
    st.markdown('<div style="padding-top: 1.9rem; color: #6b7390; font-size: .9rem;">'
                'Ejercicio académico. Este resultado no reemplaza un diagnóstico médico.</div>',
                unsafe_allow_html=True)

tab_manual, tab_csv, tab_resultados = st.tabs(["Predicción individual", "Carga de archivo", "Resultados de los modelos"])


# ---------- Pestaña 1: predicción individual ----------
with tab_manual:
    c1, c2, c3 = st.columns(3)
    c1.button("Cargar caso benigno real", on_click=cargar_caso, args=(caso_benigno, "Benigno"), width="stretch")
    c2.button("Cargar caso maligno real", on_click=cargar_caso, args=(caso_maligno, "Maligno"), width="stretch")
    c3.button("Restablecer a valores promedio", on_click=cargar_promedios, width="stretch")

    # La predicción se calcula en vivo con los valores actuales de los campos
    valores = {v: st.session_state[v] for v in variables}
    caso = pd.DataFrame([valores])
    clases, probas = predecir(modelo_elegido, caso)
    proba_maligno = probas[0] * 100

    izq, der = st.columns([1, 1.15], gap="large")

    with izq:
        tarjeta_resultado(modelo_elegido, clases[0] == 1, proba_maligno)

        nombres = list(modelos.keys())
        resultado_modelos = [predecir(n, caso) for n in nombres]

        ejemplo = st.session_state.get("ejemplo")
        if ejemplo is not None and all(abs(valores[v] - ejemplo[0][v]) < 1e-5 for v in variables):
            st.info(f"Diagnóstico real de este caso de ejemplo: **{ejemplo[1]}**")

        fuera = variables_fuera_de_rango(valores)
        if fuera:
            with st.expander(f"{len(fuera)} valor(es) fuera del rango del dataset"):
                st.write("La predicción puede no ser confiable para: " + ", ".join(etiqueta(v) for v in fuera))

    with der:
        probas_modelos = [r[1][0] * 100 for r in resultado_modelos]
        fig, ax = figura(6, 3.2)
        colores = [ROJO if p >= 50 else VERDE for p in probas_modelos]
        barras = ax.barh(nombres, probas_modelos, color=colores, height=0.55)
        ax.axvline(50, linestyle="--", color=GRIS, linewidth=1)
        ax.set_xlim(0, 112)
        ax.invert_yaxis()
        ax.set_xlabel("Probabilidad de maligno (%)")
        ax.set_title("Probabilidad de maligno por modelo", loc="left", fontweight="bold")
        for barra, p in zip(barras, probas_modelos):
            ax.text(p + 1.5, barra.get_y() + barra.get_height() / 2, f"{p:.1f}%", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Medidas del tumor")

    sub_tabs = st.tabs([f"{nombre}" for nombre in GRUPOS.values()])
    for (cod, _), sub in zip(GRUPOS.items(), sub_tabs):
        with sub:
            cols = st.columns(5)
            vars_grupo = [v for v in variables if grupo_de(v) == cod]
            for i, v in enumerate(vars_grupo):
                cols[i % 5].number_input(
                    etiqueta(v, con_grupo=False), min_value=0.0, format="%.4f", key=v,
                    help=f"{v} · Rango del dataset: {rangos.loc['min', v]:.4g} a {rangos.loc['max', v]:.4g}")


# ---------- Pestaña 2: carga de archivo (Excel o CSV) ----------
with tab_csv:
    st.write("Cargue un archivo Excel (.xlsx) o CSV con las 30 variables (una fila por tumor) para clasificar varios casos a la vez. "
             "Se utiliza el modelo seleccionado en la parte superior.")
    st.download_button("Descargar archivo de ejemplo (Excel)", a_excel(X_test.head(5)), "ejemplo.xlsx", TIPO_XLSX)

    archivo = st.file_uploader("Archivo Excel o CSV", type=["xlsx", "csv"])
    if archivo is not None:
        try:
            datos = pd.read_excel(archivo) if archivo.name.lower().endswith("xlsx") else pd.read_csv(archivo)
        except Exception:
            datos = None
        if datos is None or len(datos) == 0:
            st.error("No pude leer filas del archivo. Revisa que tenga una fila por tumor.")
        else:
            faltantes = [v for v in variables if v not in datos.columns]
            if faltantes:
                st.error("Al archivo le faltan estas columnas: " + ", ".join(faltantes))
            else:
                numeros = datos[variables].apply(pd.to_numeric, errors="coerce")
                invalidas = numeros.isnull()
                if invalidas.values.any():
                    filas_malas, cols_malas = np.where(invalidas.values)
                    detalle = [f"fila {int(f) + 2}, columna {variables[int(c)]}" for f, c in zip(filas_malas[:5], cols_malas[:5])]
                    st.error("Hay celdas vacías o con texto en lugar de números. Primeras encontradas: " + "; ".join(detalle))
                else:
                    clases, probas = predecir(modelo_elegido, numeros)
                    fuera_filas = ((numeros < rangos.loc["min"]) | (numeros > rangos.loc["max"])).any(axis=1)

                    k1, k2, k3, k4 = st.columns(4)
                    kpi(k1, "Tumores analizados", len(clases))
                    kpi(k2, "Clasificados malignos", int((clases == 1).sum()))
                    kpi(k3, "Clasificados benignos", int((clases == 0).sum()))
                    kpi(k4, "Filas con valores fuera de rango", int(fuera_filas.sum()))
                    st.write("")

                    salida = pd.DataFrame({
                        "Fila": np.arange(1, len(clases) + 1),
                        "Predicción": ["Maligno" if c == 1 else "Benigno" for c in clases],
                        "Probabilidad de Maligno (%)": (probas * 100).round(1),
                        "¿Fuera de rango?": ["Sí" if f else "No" for f in fuera_filas],
                    })
                    st.dataframe(
                        salida, hide_index=True, width="stretch",
                        column_config={"Probabilidad de Maligno (%)": st.column_config.ProgressColumn(
                            "Probabilidad de Maligno (%)", min_value=0, max_value=100, format="%.1f")})
                    st.download_button("Descargar resultados (Excel)", a_excel(salida), "predicciones.xlsx", TIPO_XLSX)


# ---------- Pestaña 3: resultados de los modelos ----------
with tab_resultados:
    tabla = pd.DataFrame(resultados).T.round(4)

    st.subheader("Comparación de métricas")
    st.caption("Evaluación sobre el conjunto de prueba. Se resalta el mejor valor de cada métrica.")
    st.dataframe(tabla.style.highlight_max(color="#d6e4f5").format("{:.4f}"), width="stretch")

    st.subheader(f"Matriz de confusión: {modelo_elegido}")
    y_pred = modelos[modelo_elegido].predict(scaler.transform(X_test[variables]))
    cm = confusion_matrix(y_test, y_pred)
    col_cm, _ = st.columns([1, 1])
    with col_cm:
        fig2, ax2 = figura(4.5, 4)
        ax2.grid(False)
        ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Benigno", "Maligno"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Benigno", "Maligno"])
        ax2.set_xlabel("Predicción"); ax2.set_ylabel("Valor real")
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=18,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
