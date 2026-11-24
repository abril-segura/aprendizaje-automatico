
from pathlib import Path
from datasets import load_dataset
from PIL import ImageOps, Image
from io import BytesIO
import hashlib

# 1. Localizar las imágenes de prueba
BASE = Path(__file__).resolve().parent

PRUEBA = BASE / "datos/modelo_v1/prueba"
DESTINO = BASE / "datos/originales/prueba"

# 2. Crear una lista de huellas digitales
buscadas = {}

for archivo in PRUEBA.glob("*/*.jpg"):

    contenido = archivo.read_bytes()

    huella = hashlib.sha256(contenido).hexdigest()

    buscadas[huella] = archivo

print("Fotografías que intentaremos recuperar:", len(buscadas))

# 3. Cargar las imágenes originales
datos = load_dataset(
    "szymonrucinski/types-of-film-shots",
    split="test",
    streaming=True
)

recuperadas = 0

# 4. Buscar cada imagen
for fila in datos:

    try:
        original = fila["image"].convert("RGB")

        # Reproducir el procesamiento anterior
        reducida = ImageOps.pad(
            original,
            (128, 128),
            method=Image.Resampling.LANCZOS,
            color="black"
        )

        memoria = BytesIO()

        reducida.save(
            memoria,
            format="JPEG",
            quality=90
        )

        huella = hashlib.sha256(
            memoria.getvalue()
        ).hexdigest()

        if huella not in buscadas:
            continue

        archivo_anterior = buscadas.pop(huella)

        carpeta = DESTINO / archivo_anterior.parent.name
        carpeta.mkdir(parents=True, exist_ok=True)

        nuevo_archivo = carpeta / archivo_anterior.name

        original.save(nuevo_archivo, quality=95)

        recuperadas += 1

        print(
            "Recuperada:",
            nuevo_archivo.name,
            original.size
        )

        if not buscadas:
            break

    except (OSError, ValueError):
        continue

print("\nFotografías recuperadas:", recuperadas)
print("Fotografías pendientes:", len(buscadas))
