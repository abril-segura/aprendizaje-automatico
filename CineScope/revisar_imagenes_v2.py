
from pathlib import Path
from PIL import Image, ImageDraw
import random

# 1. Ubicar nuestras fotografías
BASE = Path(__file__).resolve().parent

DATOS = BASE / "datos" / "modelo_v2"
DESTINO = BASE / "revision_v2"

DESTINO.mkdir(exist_ok=True)

# 2. Buscar las categorías
ENTRENAMIENTO = DATOS / "entrenamiento"

categorias = sorted(
    carpeta.name
    for carpeta in ENTRENAMIENTO.iterdir()
    if carpeta.is_dir()
)

# 3. Revisar cantidades
for grupo in ["entrenamiento", "validacion", "prueba"]:

    print("\nCONJUNTO:", grupo)

    for categoria in categorias:

        carpeta = DATOS / grupo / categoria
        archivos = sorted(carpeta.glob("*.jpg"))

        print(categoria, ":", len(archivos))

# 4. Crear mosaicos de entrenamiento
azar = random.Random(42)

for categoria in categorias:

    carpeta = ENTRENAMIENTO / categoria
    archivos = sorted(carpeta.glob("*.jpg"))

    seleccionadas = azar.sample(
        archivos,
        min(9, len(archivos))
    )

    mosaico = Image.new(
        "RGB",
        (930, 690),
        color="white"
    )

    dibujo = ImageDraw.Draw(mosaico)

    for i, archivo in enumerate(seleccionadas):

        with Image.open(archivo) as original:
            imagen = original.convert("RGB")

        imagen.thumbnail(
            (290, 185),
            Image.Resampling.LANCZOS
        )

        columna = i % 3
        fila = i // 3

        x = columna * 310 + 10
        y = fila * 230 + 10

        mosaico.paste(imagen, (x, y))

        dibujo.text(
            (x, y + 190),
            archivo.name,
            fill="black"
        )

    salida = DESTINO / f"revision_{categoria}.jpg"

    mosaico.save(salida, quality=95)

    print("Mosaico creado:", salida)

print("\nRevisión preparada correctamente.")
