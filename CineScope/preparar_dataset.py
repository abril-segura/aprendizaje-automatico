from datasets import load_dataset # obtener las imagenes de Hugging Face (donde esta el dataset)
from pathlib import Path # manejar y crear rutas de carpetas
from random import Random # mezclar imagenes
from PIL import Image, ImageOps # procesar y ajustar imagenes
import hashlib # detectar imagenes identicas

# Categoriías que queremos obtener
categorias = {
    "closeUp": "primer plano",
    "mediumShot": "plano medio",
    "longShot": "plano general",
}

# Dónde se guardaran las imágenes
destino = Path("datos") / "modelo_v1"

if destino.exists():
    raise SystemExit(
        "La carpeta modelo_v1 ya existe. "
        "Revise sus archivos antes de volver a preparar los datos"
    )

imagenes = {
    clase: [] for clase in categorias.values()
}

# Cargar el dataset
datos = load_dataset(
    "szymonrucinski/types-of-film-shots",
    split="test",
    streaming=True
)

# Recorrer el dataset
huellas = set()  # Para evitar duplicados

for fila in datos:

    etiqueta = datos.features["label"].int2str(fila["label"])

    if etiqueta not in categorias:
        continue

    clase = categorias[etiqueta]

    # Ajustar las imagenes a un mismo tamaño y formato
    try:
        imagen = fila["image"].convert("RGB")

        imagen = ImageOps.pad(
            imagen,
            (128, 128),
            method=Image.Resampling.LANCZOS,
            color="black"
        )
    except (OSError, ValueError):
        print("Se omitió una imagen que no pudo procesarse.")
        continue

    # Evitar duplicados, funciona como huella digital, si tienen exactamente la misma cantidad de pixeles
    huella = hashlib.sha256(imagen.tobytes()).hexdigest()

    if huella in huellas:
        continue

    huellas.add(huella)
    imagenes[clase].append(imagen)

# Contar las imagenes
print("\nImágenes encontradas:")

for clase, lista in imagenes.items():
    print(clase, ":", len(lista))

# Equilibramos la cantidad de imagenes por categoria, seleccionando la cantidad mínima 120
minimo = min(len(lista) for lista in imagenes.values())

if minimo < 30:
    raise SystemExit(
        "Hay menos de 30 imágenes en alguna categoría. "
        "Necesitamos revisar los datos antes de continuar."
    )

por_clase = min(120, minimo)

print("\nUtilizaremos", por_clase, "imágenes por categoría.")

# Mezcla y dicisión de datos
azar = Random(42)

for clase, lista in imagenes.items():

    azar.shuffle(lista)

    seleccionadas = lista[:por_clase]

    n_entrenamiento = int(por_clase * 0.70)
    n_validacion = int(por_clase * 0.15)

    partes = {
        "entrenamiento": seleccionadas[:n_entrenamiento],
        "validacion": seleccionadas[
            n_entrenamiento:n_entrenamiento + n_validacion
        ],
        "prueba": seleccionadas[
            n_entrenamiento + n_validacion:
        ]
    }

    # Guardar las imágenes en carpetas separadas
    for grupo, fotos in partes.items():

        carpeta = destino / grupo / clase
        carpeta.mkdir(parents=True, exist_ok=True)

        for numero, foto in enumerate(fotos, start=1):

            archivo = carpeta / f"{clase}_{numero:03d}.jpg"
            foto.save(archivo, quality=90)

        print(clase, "-", grupo, ":", len(fotos))

print("\nDataset de CineScope preparado correctamente.")