import os
import cv2
import numpy as np
from glob import glob

# Configuración
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
script_dir = os.path.dirname(os.path.abspath(__file__))

def agregar_ruido_gaussiano(imagen, media=0, sigma=15):
    """Agrega ruido gaussiano a la imagen."""
    ruido = np.random.normal(media, sigma, imagen.shape).astype(np.uint8)
    imagen_ruido = cv2.add(imagen, ruido)
    return imagen_ruido

def ajustar_brillo(imagen, factor=1.2):
    """Ajusta el brillo de la imagen."""
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[..., 2] = hsv[..., 2] * factor
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def aplicar_erosion(imagen, kernel_size=3):
    """Aplica erosión a la imagen."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(imagen, kernel, iterations=1)

def aplicar_dilatacion(imagen, kernel_size=3):
    """Aplica dilatación a la imagen."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(imagen, kernel, iterations=1)

def guardar_variaciones(nombre_base, imagen, transformaciones):
    """Guarda las variaciones de la imagen en el directorio actual."""
    for nombre, img in transformaciones.items():
        ruta_salida = os.path.join(script_dir, f"{nombre_base}_{nombre}.jpg")
        cv2.imwrite(ruta_salida, img)
        print(f"Guardada variación: {os.path.basename(ruta_salida)}")

def main():
    # Obtener todas las imágenes en el directorio actual
    imagenes = []
    for ext in valid_extensions:
        imagenes.extend(glob(os.path.join(script_dir, f"*{ext}")))

    if not imagenes:
        print("No se encontraron imágenes para procesar")
        return

    print(f"Encontradas {len(imagenes)} imágenes para procesar")

    for ruta_imagen in imagenes:
        nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
        print(f"\nProcesando: {nombre_base}")
        
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error al cargar la imagen: {ruta_imagen}")
            continue

        # Definir las transformaciones a aplicar
        transformaciones = {
            "rot90": cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE),
            "rot180": cv2.rotate(imagen, cv2.ROTATE_180),
            "rot270": cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE),
            "flip_h": cv2.flip(imagen, 1),
            "flip_v": cv2.flip(imagen, 0),
            "bright": ajustar_brillo(imagen),
            "eroded": aplicar_erosion(imagen),
            "dilated": aplicar_dilatacion(imagen),
            "noisy": agregar_ruido_gaussiano(imagen)
        }

        guardar_variaciones(nombre_base, imagen, transformaciones)

    print("\nAumentación de datos completada.")

if __name__ == "__main__":
    main()