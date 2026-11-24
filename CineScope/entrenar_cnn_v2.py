
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path

# Para que los experimentos sean más reproducibles
tf.keras.utils.set_random_seed(42)

# Carpeta principal de CineScope
BASE = Path(__file__).resolve().parent

# Dataset con nuestras imágenes originales
DATOS = BASE / "datos" / "modelo_v2"

# Carpeta donde guardaremos el nuevo modelo
MODELOS = BASE / "modelos"
MODELOS.mkdir(exist_ok=True)

RUTA_MODELO = MODELOS / "cinescope_v2_224.keras"

# Identificar las tres categorías
clases = sorted(
    carpeta.name
    for carpeta in (DATOS / "entrenamiento").iterdir()
    if carpeta.is_dir()
)

print("Categorías:", clases)

# Función que carga las imágenes
def cargar_imagenes(grupo, mezclar=False):

    return tf.keras.utils.image_dataset_from_directory(
        DATOS / grupo,
        image_size=(224, 224),
        batch_size=8,
        label_mode="int",
        class_names=clases,
        pad_to_aspect_ratio=True,
        shuffle=mezclar,
        seed=42
    )

entrenamiento = cargar_imagenes(
    "entrenamiento",
    mezclar=True
)

validacion = cargar_imagenes("validacion")

prueba = cargar_imagenes("prueba")


# Cargar MobileNetV2 con conocimientos previos
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Congelar sus pesos originales
base.trainable = False

# Construir nuestra nueva CNN
entrada = layers.Input(shape=(224, 224, 3))

x = layers.RandomFlip("horizontal")(entrada)

# Convertir los píxeles al rango de -1 a 1
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

# Extraer características con MobileNetV2
x = base(x, training=False)

# Resumir los patrones visuales
x = layers.GlobalAveragePooling2D()(x)

# Reducir el riesgo de sobreajuste
x = layers.Dropout(0.3)(x)

# Clasificar los tres tipos de plano
salida = layers.Dense(
    len(clases),
    activation="softmax"
)(x)

modelo = tf.keras.Model(
    inputs=entrada,
    outputs=salida
)

modelo.summary()


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(RUTA_MODELO),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    )
]


print("\nComenzando entrenamiento de CineScope V2...")

historial = modelo.fit(
    entrenamiento,
    validation_data=validacion,
    epochs=25,
    callbacks=callbacks
)

# Recuperar el mejor modelo guardado
mejor_modelo = tf.keras.models.load_model(
    RUTA_MODELO
)

# Medir el resultado de validación
_, val_accuracy = mejor_modelo.evaluate(
    validacion,
    verbose=0
)

print(
    f"\nExactitud de validación: "
    f"{val_accuracy * 100:.2f}%"
)

# Evaluación final con fotografías reservadas
perdida, accuracy = mejor_modelo.evaluate(
    prueba,
    verbose=0
)

print("\n===== CINESCOPE V2 =====")
print(f"Exactitud de prueba: {accuracy * 100:.2f}%")
print(f"Pérdida de prueba: {perdida:.4f}")
print("Modelo guardado en:", RUTA_MODELO)
