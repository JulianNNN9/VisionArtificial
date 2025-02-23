import cv2
import numpy as np
import pandas as pd

def extraer_caracteristicas(imagen, nombre):
    """Extrae características de la imagen y devuelve un diccionario con los valores."""
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    if imagen.shape[0] > 100 and imagen.shape[1] > 200:
        if len(imagen.shape) == 3:  # Imagen en color (BGR)
            pixel_valor = tuple(imagen[100, 200])  
        else:  # Imagen en escala de grises
            pixel_valor = (imagen[100, 200], None, None)  
    else:
        pixel_valor = (None, None, None)

    matriz_reducida = img_gray[:3, :3] if img_gray.shape[0] >= 3 and img_gray.shape[1] >= 3 else None
    determinante = np.linalg.det(matriz_reducida) if matriz_reducida is not None else None
    multiplicacion = np.sum(img_gray * 2)
    suma = np.sum(img_gray + 50)
    caracteristicas = {
        "Nombre": nombre,
        "Pixel_100_200_B": pixel_valor[0],
        "Pixel_100_200_G": pixel_valor[1],
        "Pixel_100_200_R": pixel_valor[2],
        "Promedio_Intensidad": np.mean(img_gray),
        "Desviacion_Estandar": np.std(img_gray),
        "Determinante": determinante,
        "Multiplicacion": multiplicacion,
        "Suma": suma
    }
    return caracteristicas

def guardar_caracteristicas_csv(caracteristicas_lista, archivo_salida="VisionArtificial/Vision Artificial/Laboratorio 1/Resultados/caracteristicas_imagenes.csv"):
    """Guarda una lista de características extraídas en un archivo CSV."""
    if not isinstance(caracteristicas_lista, list) or not all(isinstance(item, dict) for item in caracteristicas_lista):
        print("Error: La entrada debe ser una lista de diccionarios.")
        return
    
    df = pd.DataFrame(caracteristicas_lista)
    df.to_csv(archivo_salida, index=False)
    print(f"Características guardadas en {archivo_salida}")

