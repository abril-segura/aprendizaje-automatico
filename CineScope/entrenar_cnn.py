import tensorflow as tf
from tensorflow.keras import layers # INTERFAZ PARA CONSTRUIR REDES NEURONALES : LAYES CONTIENE LAS CAPAS QUE SE UTILIZARAN
from pathlib import Path

tf.keras.utils.set_random_seed(42)

print("CineScope:iniciando nuestra CNN")

# Acceso a las carpetas que preparamos anteriormente
carpeta = Path("datos/modelo_v1")

entrenamiento = tf.keras.utils.image_dataset_from_directory( # lee imagenes de una carpeta y obtiene sus etiquetas a partir de sus subcarpetas
    carpeta / "entrenamiento",
    image_size=(128, 128),
    batch_size=16, # agrupar por lote, para que CNN calcule qué tanto se equivoca y actualice sus parametros durante el entrenamiento
    label_mode="int",
    shuffle=True,
    seed=42
)

clases = entrenamiento.class_names # mantiene un orden consistente de los conjuntos 

validacion = tf.keras.utils.image_dataset_from_directory(
    carpeta / "validacion",
    image_size=(128, 128),
    batch_size=16,
    label_mode="int",
    class_names=clases,
    shuffle=False
)

prueba = tf.keras.utils.image_dataset_from_directory(
    carpeta / "prueba",
    image_size=(128, 128),
    batch_size=16,
    label_mode="int",
    class_names=clases,
    shuffle=False
)

print("Categorías que aprenderá CineScope")
print(clases)

if len(clases) != 3:
    raise ValueError("CineScope necesita exactamente tres categorías.")

# Contruir la CNN
modelo = tf.keras.Sequential([ # permite crear una red donde las capas se ejecutan secuencialmente, una tras otra 

    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255), # normalizar los valores de los pixeles a un rango de 0 a 1

    layers.RandomFlip("horizontal"), # puede voltear las imagenes horizontalmente para aumentar la cantidad de datos de entrenamiento

    layers.Conv2D(16, (3, 3), activation="relu", padding="same"), # capa convolucional, detecta patrones en las imagenes
    layers.MaxPooling2D((2, 2)), # capa de agrupamiento, reduce la cantidad de pixeles a procesar

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"), # se elevan los filtros para detectar patrones más complejos
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(), # aplana la imagen a un vector de 1 dimension, o sea, una lista de numeros

    layers.Dense(32, activation="relu"), # capa densa, conecta todos los pixeles de la imagen con todas las neuronas de la capa (32 neuronas en este caso)

    layers.Dropout(0.3), # capa de abandono, evita que la CNN se sobreentrene

    layers.Dense(3, activation="softmax") # capa de salida, devuelve la probabilidad de cada categoría / existen tres categorías, por eso 3 neuronas / softmax devuelve un valor entre 0 y 1 para cada categoría, y la suma de los tres valores es 1
])

modelo.summary() # imprime un resumen de la CNN

modelo.compile(
    optimizer="adam", # algoritmo de optimización, ajusta los pesos de la CNN para minimizar el error
    loss="sparse_categorical_crossentropy", # función de pérdida, mide qué tan bien está funcionando la CNN
    metrics=["accuracy"] # métrica de evaluación, mide qué tan bien está funcionando la CNN
)

print("\nComenznado el entreamiento de CineScope...")

historial = modelo.fit( # instrucción para entrenar la CNN, recibe los datos de entrenamiento y validación, y el número de épocas
    entrenamiento,
    validation_data=validacion,
    epochs=10 # epochs es un recorrido completo por el conjunto de entrenamiento, en cada epoca va ajustar los parametros para mejorar las predicciones
)

perdida, exactitud = modelo.evaluate(prueba) # evalua la CNN con el conjunto de prueba, devuelve la pérdida y la exactitud

print("\nRESULTADOS FINALES DE CINESCOPE")
print(f"Exactitud en prueba: {exactitud * 100:.2f}%")
print(f"Pérdida en prueba: {perdida:.4f}")

# Guardar el modelo entrenado
carpeta_modelos = Path("modelos")
carpeta_modelos.mkdir(exist_ok=True)

modelo.save(carpeta_modelos / "cinescope_cnn.keras") # guarda el modelo entrenado en un archivo .keras

print("\nModelo guardado correctamente.")