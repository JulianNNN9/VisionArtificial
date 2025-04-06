import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage import img_as_ubyte
import math

'''
METODOS

- Algoritmo Momentos de Hu
- Algoritmo Optical Flow (Flujo Óptico)
- Algoritmo Laplaciano de Gauss (LoG)
- Algoritmo AKAZE
- Algoritmo SIFT
- Algoritmo SURF
- Algoritmo ORB
- Algoritmo HOG (Histogram of Oriented Gradients)
- Algoritmo KAZE
- Transformada de HOUG
    - Detección de Rectas
    - Detección de Circunferencias
- Métodos Estadísticos
    - Estadísticos de Primer Orden
        - Media
        - Varianza
        - Desviación
        - Entropía
    - Estadísticos de Segundo Orden
        - Homogeneidad
        - Contraste
        - Disimilaridad
        - Media
        - Desviación Estándar
        - Entropía
        - Energía
- Métodos Geométricos (No hay informacion en diapositivas)
- Métodos basados en modelos (No hay informacion en diapositivas)
- Métodos basados en tratamiento de señal (No hay informacion en diapositivas)

'''

def momentos_de_hu(imagen, umbral=127, suavizado=False, usar_canny=False, titulo='Imagen'):
    """
    Procesa una imagen (ya cargada) para binarizar, calcular momentos y Momentos de Hu.

    Parámetros:
    - imagen (np.ndarray): Imagen cargada en escala de grises.
    - umbral (int): Valor de umbralización.
    - suavizado (bool): Aplica suavizado GaussianBlur si es True.
    - usar_canny (bool): Aplica Canny antes del cálculo de momentos si es True.
    - titulo (str): Título para la visualización.

    Retorna:
    - hu_moments: Momentos de Hu calculados.
    """
    if imagen is None:
        print("La imagen es inválida.")
        return None

    # Opcional: suavizado
    if suavizado:
        imagen = cv2.GaussianBlur(imagen, (5, 5), 0)

    # Umbralización
    _, imagen_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)

    # Opcional: usar bordes de Canny
    if usar_canny:
        imagen_binaria = cv2.Canny(imagen_binaria, 100, 200)

    # Calcular los momentos
    moments = cv2.moments(imagen_binaria)

    # Calcular los Momentos de Hu
    hu_moments = cv2.HuMoments(moments)

    # Mostrar los Momentos de Hu
    print(f"\nMomentos de Hu para {titulo}:")
    for i in range(7):
        print(f"  Momento {i+1}: {hu_moments[i][0]}")

    # Visualización
    plt.title(titulo)
    plt.imshow(imagen_binaria, cmap='gray')
    plt.axis('off')
    plt.show()

    return hu_moments

def analizar_imagen_grises(imagen: np.ndarray) -> dict:
    """
    Analiza una imagen en escala de grises y calcula estadísticas básicas.

    Parámetros:
        imagen (np.ndarray): Imagen en escala de grises como matriz NumPy.

    Retorna:
        dict: Diccionario con media, varianza, desviación estándar y entropía.
    """
    if imagen is None or len(imagen.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises (matriz 2D).")

    # 1. Media
    media = np.mean(imagen)

    # 2. Varianza
    varianza = np.var(imagen)

    # 3. Desviación estándar
    desviacion_estandar = np.std(imagen)

    # 4. Entropía
    imagen_float = np.float32(imagen) + 1e-5  # Evita log(0)
    entropia = -np.sum(imagen_float * np.log(imagen_float))

    return {
        "media": media,
        "varianza": varianza,
        "desviacion_estandar": desviacion_estandar,
        "entropia": entropia
    }