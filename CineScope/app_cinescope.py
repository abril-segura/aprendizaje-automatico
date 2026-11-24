"""CineScope Studio: interfaz cinematografica para comparar dos CNN.

Coloca este archivo junto a las carpetas 'datos' y 'modelos' de CineScope.
"""

from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# 1) CONFIGURACION Y ARCHIVOS
st.set_page_config(
    page_title="CineScope Studio | El ojo del director",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
ESTILOS = BASE / "cinescope_style.css"
if ESTILOS.is_file():
    st.html(ESTILOS)

MODELOS = {
    "V1": BASE / "modelos" / "cinescope_cnn.keras",
    "V2": BASE / "modelos" / "cinescope_v2_224.keras",
}
CARPETA_PRUEBA = BASE / "datos" / "modelo_v2" / "prueba"
CARPETA_TRAIN = BASE / "datos" / "modelo_v2" / "entrenamiento"

for ruta in (CARPETA_PRUEBA, CARPETA_TRAIN):
    if not ruta.is_dir():
        st.error(f"No encuentro la carpeta de CineScope: {ruta}")
        st.stop()

clases = sorted(p.name for p in CARPETA_TRAIN.iterdir() if p.is_dir())
if len(clases) != 3:
    st.error("Se esperaban exactamente 3 clases de CineScope en el entrenamiento de V2.")
    st.stop()


def bonito(nombre: str) -> str:
    return nombre.replace("_", " ").capitalize()


@st.cache_resource(show_spinner="Cargando el modelo de CineScope...")
def cargar_modelo(ruta: str):
    return tf.keras.models.load_model(ruta, compile=False)


def predecir(version: str, imagen: Image.Image, archivo_prueba: Path | None):
    ruta = MODELOS[version]
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo {version}: {ruta}")
    modelo = cargar_modelo(str(ruta))

    original = ImageOps.exif_transpose(imagen).convert("RGB")

    if version == "V1":
        # Para fotos del dataset usamos la misma copia de prueba de 128x128.
        # Para fotos subidas reproducimos el preprocesamiento de V1.
        preparada = None
        if archivo_prueba is not None:
            copia = (
                BASE / "datos" / "modelo_v1" / "prueba"
                / archivo_prueba.parent.name / archivo_prueba.name
            )
            if copia.is_file():
                with Image.open(copia) as foto:
                    preparada = foto.convert("RGB")
        if preparada is None:
            preparada = ImageOps.pad(
                original,
                (128, 128),
                method=Image.Resampling.LANCZOS,
                color="black",
            )
        matriz = np.asarray(preparada, dtype=np.float32)
    else:
        # Igual que image_dataset_from_directory(..., pad_to_aspect_ratio=True)
        # de la V2: conserva proporcion y crea entrada 224x224.
        redimensionar = tf.keras.layers.Resizing(
            224, 224, interpolation="bilinear", pad_to_aspect_ratio=True
        )
        matriz = redimensionar(
            np.asarray(original, dtype=np.float32)
        ).numpy()

    lote = np.expand_dims(matriz, axis=0)
    probabilidades = np.asarray(modelo.predict(lote, verbose=0)[0], dtype=float)
    if len(probabilidades) != len(clases):
        raise ValueError("Las salidas del modelo no coinciden con las categorías.")
    return probabilidades


def tarjeta_resultado(version: str, probabilidades: np.ndarray, etiqueta_real=None):
    indice_ganador = int(np.argmax(probabilidades))
    ganador = clases[indice_ganador]
    confianza = float(probabilidades[indice_ganador]) * 100
    etiqueta = bonito(ganador)

    barras = []
    for indice in np.argsort(probabilidades)[::-1]:
        indice = int(indice)
        valor = max(0.0, min(100.0, float(probabilidades[indice]) * 100))
        nombre = escape(bonito(clases[indice]))
        barras.append(
            f'<div class="cs-prob-line"><span>{nombre}</span>'
            f'<strong>{valor:.1f}%</strong></div>'
            f'<div class="cs-track"><div class="cs-fill" style="width:{valor:.2f}%"></div></div>'
        )

    estado = ""
    if etiqueta_real is not None:
        acierto = ganador == etiqueta_real
        estado = (
            '<div class="cs-check cs-ok">✓ Coincide con la etiqueta</div>'
            if acierto else
            '<div class="cs-check cs-bad">✕ Difiere de la etiqueta</div>'
        )

    score_prueba = "40.74%" if version == "V1" else "87.04%"
    descriptor = "CNN creada desde cero · 128 px" if version == "V1" else "MobileNetV2 · 224 px"
    st.html(
        f'<div class="cs-result-card">'
        f'<div class="cs-result-top"><span class="cs-overline">MODELO {version}</span>'
        f'<span class="cs-tiny">PRUEBA GLOBAL: {score_prueba}</span></div>'
        f'<div class="cs-result-name">{escape(etiqueta)}</div>'
        f'<div class="cs-confidence">{confianza:.1f}% <span>probabilidad estimada</span></div>'
        f'<div class="cs-prob-title">DISTRIBUCIÓN DE PREDICCIONES</div>'
        f'{"".join(barras)}{estado}'
        f'<div class="cs-model-meta">{escape(descriptor)}</div>'
        f'</div>'
    )
    return ganador


# 2) CABECERA CINEMATOGRAFICA
st.html("""
<section class="cs-hero">
  <div class="cs-brand"><span class="cs-mark">▣</span> CINESCOPE <span class="cs-brand-muted">/ STUDIO</span></div>
  <div class="cs-eyebrow">AI FILM ANALYSIS <span class="cs-dot"></span> VISUAL INTELLIGENCE</div>
  <h1>El ojo del <em>director.</em></h1>
  <p>Una mirada a la composición cinematográfica a través de redes neuronales convolucionales.</p>
  <div class="cs-hero-line"></div>
</section>
""")

# Selector de modelo encima del fotograma (sin barra lateral).
selector_izq, selector_der = st.columns([2.5, 1], gap="medium")
with selector_izq:
    st.html('<div class="cs-label">MODO DE ANÁLISIS</div>')
    modo = st.segmented_control(
        "Modo de análisis",
        ["V2", "V1", "Comparar"],
        default="V2",
        required=True,
        label_visibility="collapsed",
        width="stretch",
    )
with selector_der:
    st.html('<div class="cs-live"><span class="cs-live-dot"></span> MODELOS CARGADOS BAJO DEMANDA<br><small>V1 40.74% &nbsp;·&nbsp; V2 87.04% EN PRUEBA</small></div>')

# 3) CUERPO: IMAGEN GRANDE Y RESULTADOS AL LADO
izquierda, derecha = st.columns([1.35, 1], gap="large", vertical_alignment="top")

imagen = None
archivo_prueba = None
etiqueta_real = None

with izquierda:
    with st.container(border=True):
        st.html('<div class="cs-section-title"><span>01 /</span> FOTOGRAMA <small>SELECCIÓN</small></div>')
        origen = st.segmented_control(
            "Origen de la imagen",
            ["Colección de prueba", "Subir fotografía"],
            default="Colección de prueba",
            required=True,
            label_visibility="collapsed",
            width="stretch",
        )

        if origen == "Colección de prueba":
            ejemplos = sorted(CARPETA_PRUEBA.glob("*/*.jpg"))
            if not ejemplos:
                st.warning("No se encontraron imágenes en datos/modelo_v2/prueba.")
            else:
                archivo_prueba = st.selectbox(
                    "Selecciona una escena",
                    ejemplos,
                    format_func=lambda p: f"{bonito(p.parent.name)}  /  {p.name}",
                )
                etiqueta_real = archivo_prueba.parent.name
                with Image.open(archivo_prueba) as foto:
                    imagen = foto.convert("RGB")
        else:
            archivo = st.file_uploader(
                "Sube un fotograma (JPG, PNG o WEBP)", type=["jpg", "jpeg", "png", "webp"]
            )
            if archivo is not None:
                try:
                    with Image.open(archivo) as foto:
                        imagen = ImageOps.exif_transpose(foto).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    st.error("No se pudo leer el archivo. Prueba otra fotografía.")

        if imagen is not None:
            st.image(imagen, width="stretch")
            ancho, alto = imagen.size
            chip = (
                f'ETIQUETA ORIGINAL: {escape(bonito(etiqueta_real))}'
                if etiqueta_real else 'IMAGEN APORTADA POR TI'
            )
            st.html(
                f'<div class="cs-frame-meta"><span>FRAME / {ancho} × {alto} PX</span>'
                f'<span>{chip}</span></div>'
            )
        else:
            st.html('<div class="cs-empty">Selecciona o sube un fotograma para iniciar el análisis.</div>')

with derecha:
    with st.container(border=True):
        st.html('<div class="cs-section-title"><span>02 /</span> ANÁLISIS <small>RED NEURONAL</small></div>')
        if imagen is None:
            st.html('<div class="cs-empty">Esperando imagen de entrada...</div>')
        else:
            versiones = ["V1", "V2"] if modo == "Comparar" else [modo]
            for version in versiones:
                try:
                    resultado = predecir(version, imagen, archivo_prueba)
                    tarjeta_resultado(version, resultado, etiqueta_real)
                except (FileNotFoundError, ValueError, OSError) as exc:
                    st.error(f"No se pudo analizar con {version}: {exc}")
                except Exception as exc:
                    st.error(f"Falló la predicción de {version}: {exc}")
            st.html(
                '<div class="cs-disclaimer">Las probabilidades son estimaciones del modelo, '
                'no certezas. En el modo comparación ambos analizan el mismo fotograma.</div>'
            )

st.html("""
<footer class="cs-footer">
  <span>CINESCOPE — THE DIRECTOR'S EYE</span>
  <span>COMPUTER VISION / CNN RESEARCH PROJECT</span>
</footer>
""")
