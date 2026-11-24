from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(
    page_title="CineScope | El ojo del director",
    page_icon="🎬",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

MODELOS = {
    "V1": BASE / "modelos" / "cinescope_cnn.keras",
    "V2": BASE / "modelos" / "cinescope_v2_224.keras"
}

RUTA_ENTRENAMIENTO = BASE / "datos/modelo_v2/entrenamiento"
RUTA_PRUEBA = BASE / "datos/modelo_v2/prueba"

clases = sorted(
    carpeta.name
    for carpeta in RUTA_ENTRENAMIENTO.iterdir()
    if carpeta.is_dir()
)

if len(clases) != 3:
    st.error("No se encontraron las tres categorías de CineScope.")
    st.stop()

def nombre_bonito(nombre):
    return nombre.replace("_", " ").capitalize()


st.sidebar.title("CineScope Studio")

modo = st.sidebar.radio(
    "Selecciona el modo de análisis",
    [
        "CineScope V1",
        "CineScope V2",
        "Comparar V1 y V2"
    ],
    index=1
)

st.sidebar.divider()

st.sidebar.write(
    "V1: CNN construida desde cero."
)

st.sidebar.write(
    "V2: MobileNetV2 con aprendizaje por transferencia."
)

st.sidebar.caption(
    "Ambos modelos fueron entrenados para reconocer "
    "tres tipos de planos cinematográficos."
)


@st.cache_resource
def cargar_modelo(ruta):
    return tf.keras.models.load_model(
        ruta,
        compile=False
    )


def obtener_modelo(version):

    ruta = MODELOS[version]

    if not ruta.is_file():
        st.error(
            f"No se encontró el modelo {version}: {ruta}"
        )
        st.stop()

    return cargar_modelo(str(ruta))


def preparar_imagen(imagen, version, ruta_prueba=None):

    imagen = ImageOps.exif_transpose(
        imagen
    ).convert("RGB")

    if version == "V1":

        # Si es una fotografía de prueba, utilizar
        # su copia original de entrenamiento V1
        preparada = None

        if ruta_prueba is not None:

            ruta_v1 = (
                BASE
                / "datos/modelo_v1/prueba"
                / ruta_prueba.parent.name
                / ruta_prueba.name
            )

            if ruta_v1.is_file():
                with Image.open(ruta_v1) as archivo:
                    preparada = archivo.convert("RGB")

        if preparada is None:

            preparada = ImageOps.pad(
                imagen,
                (128, 128),
                method=Image.Resampling.LANCZOS,
                color="black"
            )

        matriz = np.asarray(
            preparada,
            dtype=np.float32
        )

    else:

        # Mismo tipo de ajuste utilizado
        # durante el entrenamiento de V2
        redimensionar = tf.keras.layers.Resizing(
            224,
            224,
            interpolation="bilinear",
            pad_to_aspect_ratio=True
        )

        matriz = redimensionar(
            np.asarray(imagen, dtype=np.float32)
        ).numpy()

    lote = np.expand_dims(matriz, axis=0)

    return lote


def analizar_fotografia(imagen, version, ruta_prueba=None):

    modelo = obtener_modelo(version)

    lote = preparar_imagen(
        imagen,
        version,
        ruta_prueba
    )

    predicciones = modelo.predict(
        lote,
        verbose=0
    )[0]

    ganador = int(np.argmax(predicciones))

    return predicciones, ganador


def mostrar_resultados(version, imagen, ruta_prueba=None):

    predicciones, ganador = analizar_fotografia(
        imagen,
        version,
        ruta_prueba
    )

    st.subheader(f"CineScope {version}")

    st.metric(
        "Plano detectado",
        nombre_bonito(clases[ganador])
    )

    st.caption(
        f"Confianza estimada del modelo: "
        f"{predicciones[ganador] * 100:.2f}%"
    )

    st.write("Probabilidades por categoría")

    orden = np.argsort(predicciones)[::-1]

    for indice in orden:

        indice = int(indice)

        probabilidad = float(predicciones[indice])

        st.write(
            f"{nombre_bonito(clases[indice])}: "
            f"{probabilidad * 100:.2f}%"
        )

        st.progress(probabilidad)

    return ganador


st.markdown("#### AI FILM ANALYSIS / CINESCOPE STUDIO")

st.title("CINESCOPE")

st.subheader("El ojo del director")

st.write(
    "Analiza fotografías cinematográficas mediante "
    "redes neuronales convolucionales."
)

st.divider()

izquierda, derecha = st.columns(
    [1, 1.5],
    gap="large"
)

imagen = None
ruta_prueba = None

with izquierda:

    st.subheader("01 / Selecciona una fotografía")

    origen = st.radio(
        "Origen de la imagen",
        ["Fotografías de prueba", "Subir fotografía"],
        horizontal=True
    )

    if origen == "Fotografías de prueba":

        ejemplos = sorted(
            RUTA_PRUEBA.glob("*/*.jpg")
        )

        if ejemplos:

            ruta_prueba = st.selectbox(
                "Selecciona un fotograma",
                ejemplos,
                format_func=lambda ruta: (
                    f"{nombre_bonito(ruta.parent.name)} "
                    f"- {ruta.name}"
                )
            )

            with Image.open(ruta_prueba) as archivo:
                imagen = archivo.convert("RGB")

        else:
            st.warning(
                "No encontramos fotografías de prueba."
            )

    else:

        archivo = st.file_uploader(
            "Sube una fotografía",
            type=["jpg", "jpeg", "png"]
        )

        if archivo is not None:

            try:
                with Image.open(archivo) as foto:
                    imagen = ImageOps.exif_transpose(
                        foto
                    ).convert("RGB")

            except (UnidentifiedImageError, OSError):
                st.error(
                    "No se pudo abrir la fotografía."
                )

with derecha:

    st.subheader("02 / Fotograma cinematográfico")

    if imagen is not None:

        st.image(
            imagen,
            caption="Fotograma seleccionado",
            width="stretch"
        )

    else:
        st.info(
            "Selecciona una fotografía para comenzar."
        )


st.divider()

st.subheader("03 / Resultados del análisis")

if imagen is not None:

    if modo == "CineScope V1":

        mostrar_resultados(
            "V1",
            imagen,
            ruta_prueba
        )

    elif modo == "CineScope V2":

        mostrar_resultados(
            "V2",
            imagen,
            ruta_prueba
        )

    else:

        columna_v1, columna_v2 = st.columns(2, gap="large")

        with columna_v1:

            resultado_v1 = mostrar_resultados(
                "V1",
                imagen,
                ruta_prueba
            )

        with columna_v2:

            resultado_v2 = mostrar_resultados(
                "V2",
                imagen,
                ruta_prueba
            )

        st.divider()

        if resultado_v1 == resultado_v2:
            st.info(
                "Las dos versiones predicen el mismo tipo de plano."
            )
        else:
            st.warning(
                "Las dos versiones predicen tipos de plano diferentes."
            )

    if ruta_prueba is not None:

        st.caption(
            "Etiqueta original del dataset: "
            + nombre_bonito(ruta_prueba.parent.name)
        )

    st.caption(
        "Las probabilidades representan estimaciones "
        "de las redes neuronales y pueden ser incorrectas."
    )

else:

    st.info(
        "Selecciona una imagen para visualizar sus resultados."
    )
