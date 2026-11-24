from datasets import load_dataset
from pathlib import Path

datos = load_dataset(
    "szymonrucinski/types-of-film-shots",
    split="test",
    streaming=True
)

print("Categorías disponibles:")
print(datos.features["label"].names)

categorias = { ## Nombre en el dataset : Nombre en el proyecto
    "closeUp": "primer plano",
    "mediumShot": "plano medio",
    "longShot": "plano general",
}

conteos = { ## creación del contador por categoría
    "primer plano": 0,
    "plano medio": 0,
    "plano general": 0,
}

carpeta = Path("datos") / "ejemplos" ## Creación de la carpeta para guardar las imágenes
carpeta.mkdir(parents=True, exist_ok=True)

for fila in datos: ## Iteración sobre el dataset, POR CADA IMÁGEN
    nombre = datos.features["label"].int2str(fila["label"])

    if nombre not in categorias: ## Si la categoría no está en el diccionario de categorías, se omite la imagen
        continue

    clase = categorias[nombre]

    if conteos[clase] >= 3: ## Si ya se han guardado 3 imágenes de esa categoría, se omite la imagen
        continue

    conteos[clase] += 1

    imagen = fila["image"].convert("RGB")
    archivo = carpeta / f"{clase}_{conteos[clase]}.jpg"

    imagen.save(archivo)
    print("Guardada:", archivo)

    if sum(conteos.values()) == 9: ## Si ya se han guardado 3 imágenes de cada categoría, se termina el bucle
        break

    print("Imágenes por categoría:", conteos)
          