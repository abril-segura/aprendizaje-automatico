
from pathlib import Path
from datasets import load_dataset
from PIL import ImageOps, Image
from io import BytesIO
import hashlib

# Carpetas del proyecto
BASE = Path(__file__).resolve().parent

ANTERIOR = BASE / "datos/modelo_v1"
NUEVO = BASE / "datos/modelo_v2"

# Buscar las fotografías de los tres grupos
archivos = []

for grupo in ["entrenamiento", "validacion", "prueba"]:
    archivos.extend(
        (ANTERIOR / grupo).glob("*/*.jpg")
    )

# Calcular sus huellas digitales
buscadas = {}

for archivo in archivos:
    huella = hashlib.sha256(
        archivo.read_bytes()
    ).hexdigest()

    buscadas[huella] = archivo

print("Fotografías por recuperar:", len(buscadas))

# Acceder nuevamente a los originales
datos = load_dataset(
    "szymonrucinski/types-of-film-shots",
    split="test",
    streaming=True
)

recuperadas = 0

for fila in datos:

    if not buscadas:
        break

    try:
        original = fila["image"].convert("RGB")

        # Reproducir la reducción del primer dataset
        reducida = ImageOps.pad(
            original,
            (128, 128),
            method=Image.Resampling.LANCZOS,
            color="black"
        )

        memoria = BytesIO()
        reducida.save(memoria, format="JPEG", quality=90)

        huella = hashlib.sha256(
            memoria.getvalue()
        ).hexdigest()

        if huella not in buscadas:
            continue

        anterior = buscadas.pop(huella)

        # Conservar su grupo y categoría
        relativa = anterior.relative_to(ANTERIOR)
        destino = NUEVO / relativa

        destino.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Guardar la imagen a su resolución original
        original.save(destino, quality=95)

        recuperadas += 1

        print(
            f"Recuperada {recuperadas}: "
            f"{destino.name} - {original.size}"
        )

    except (OSError, ValueError):
        continue

print("\nRESULTADOS")
print("Recuperadas:", recuperadas)
print("Pendientes:", len(buscadas))

if buscadas:
    print("\nAlgunas imágenes no coincidieron.")
    print("Conserva el dataset anterior y revisa el resultado.")
else:
    print("\nDataset original recuperado correctamente.")
