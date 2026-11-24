from pathlib import Path
import numpy as np # para trabajr con los números que representan una imagen
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config( # configuracion pantalla principal
    page_title="CineScope | El ojo del director",
    page_icon="🎬",
    layout="wide",
)

BASE = Path(__file__).resolve().parent # identifica la carpeta del archivo pro si se abre en otra terminal

RUTA_MODELO = BASE / "modelos" / "cinescope_cnn.keras"

RUTA_ENTRENAMIENTO = BASE / "datos" / "modelo_v1" / "entrenamiento"

RUTA_PRUEBA = BASE / "datos" / "modelo_v1" / "prueba"

if not RUTA_MODELO.is_file():
    st.error("No se encontró el modelo entrenado de CineScope.")
    st.stop()

if not RUTA_ENTRENAMIENTO.is_dir():
    st.error("No se encontró la carpeta de entrenamiento.")
    st.stop()

clases = sorted(
    carpeta.name
    for carpeta in RUTA_ENTRENAMIENTO.iterdir()
    if carpeta.is_dir()
)

if len(clases) != 3:
    st.error("Se esperaban exactamente tres categorías.")
    st.stop()

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(RUTA_MODELO)

modelo = cargar_modelo() # se recupera la CNN desde el archivo .keras

if modelo.output_shape[-1] != len(clases):
    st.error("Las categorías no coinciden con las salidas del modelo.")
    st.stop()

def nombre_mejorado(nombre):
    return nombre.replace("_", " ").capitalize()

st.markdown("#### AI FIL ANALYSIS / VERSION 01")

st.title("CINESCOPE")

st.subheader("l ojo del director")

st.write(
    "Inteligencia Artificial que analiza fotografías "
    "cinematográficas y reconoce el tipo de plano utilizado"
)

st.divider()

izquierda, derecha = st.columns([1.3, 1], gap="large")

# Elegir una imagen
imagen = None

with izquierda:

    st.subheader("01 / Seleccionar imagen")

    origen = st.radio( # permite seleccionar entre dos opciones
        "Origen de la imagen",
        ["Imagen de prueba", "Subir fotografía"], 
        horizontal=True
    )

    if origen == "Imagen de prueba":

        ejemplos = sorted(
            RUTA_PRUEBA.glob("*/*.jpg")
        )

        if ejemplos:

            elegido = st.selectbox(
                "Selecciona una imagen",
                ejemplos,
                format_func=lambda ruta: (
                    f"{nombre_mejorado(ruta.parent.name)} - {ruta.name}"
                )
            )

            imagen = Image.open(elegido).convert("RGB")

        else:
            st.warning("No encontramos imágenes de prueba.")

    else:

        archivo = st.file_uploader( # permite sleccionar archivos desde la computadora
            "Selecciona una imagen JPG, PNG o WEBP",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if archivo is not None:

            try:
                imagen = ImageOps.exif_transpose(
                    Image.open(archivo)
                ).convert("RGB")

            except (UnidentifiedImageError, OSError):
                st.error("No se pudo abriri esta imagen")

    if imagen is not None:
        st.image( # muestra la imagen en pantalla
            imagen,
            caption="Imagen sleccionada",
            width="stretch"
        )

## Hcaer que CNN reconozca la imagen
with derecha:

    st.subheader("02 / Análisis de la CNN")

    if imagen is None:

        st.info("Selecciona una imagen para comenzar.")

    else:

        # se prepara la imagen
        preparada = ImageOps.pad(
            imagen,
            (128, 128),
            method=Image.Resampling.LANCZOS,
            color="black"
        )

        # convertir los píxeles en numeros 
        matriz =np.array(preparada, dtype=np.float32)

        #añadir dimensión de los lotes
        lote = np.expand_dims(matriz, axis=0)

        # obtener las predicciones
        predicciones = modelo.predict(
            lote,
            verbose=0 # evita qe se imprima una barra de progeso innecesaria
        )[0] # selecciona el primer resultado del lote, o sea, la imagen

        # identifiicar la categoria ganadora
        ganador = int(np.argmax(predicciones)) # encuentra la posición del número más grande, la ganadora de las 3 categorias 

        st.metric(
            "Plano cinematográfico detectado",
            nombre_mejorado(clases[ganador]) # convierte ahora sí en el nombre de la categoria 
        )

        st.caption(
            f"Puntuación del modelo: "
            f"{predicciones[ganador] * 100:.1f}%"
        )

        st.divider()

        st.write("### Probabilidades por categoría")

        orden = np.argsort(predicciones)[::-1]

        for indice in orden:
            indice = int(indice)

            probabilidad = float(predicciones[indice])

            st.write(
                f"{nombre_mejorado(clases[indice])}: "
                f"{probabilidad * 100:.1f}%"
            )

            st.progress(probabilidad) # barras de cada prediccion

        st.caption(
            "Estas puntuaciones representan las predicciones "
            "del modelo, no una grantía de clasificación correcta"
        )

        if origen == "Imagen de prueba":
            st.caption(
                "Etiqueta original del dataset: "
                + nombre_mejorado(elegido.parent.name)
            )