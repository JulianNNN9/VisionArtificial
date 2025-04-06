'''
METODOS

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

print("a")

import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_imagen_desde_objeto(imagen, umbral=127, suavizado=False, usar_canny=False, titulo='Imagen'):
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


import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas(imagen, umbral_canny1=50, umbral_canny2=150, aperture_size=7, umbral_hough=150, titulo='Líneas Detectadas'):
    """
    Detecta líneas rectas en una imagen usando Canny + Transformada de Hough.

    Parámetros:
    - imagen: Imagen en escala de grises (np.ndarray)
    - umbral_canny1: Umbral inferior para Canny
    - umbral_canny2: Umbral superior para Canny
    - aperture_size: Tamaño del kernel Sobel (3, 5 o 7)
    - umbral_hough: Umbral mínimo de acumulación en la transformada de Hough
    - titulo: Título del gráfico de salida

    Retorna:
    - edges: Imagen de bordes detectados
    - lines: Líneas detectadas (parámetros rho y theta)
    """
    if imagen is None:
        print("La imagen es inválida.")
        return None, None

    edges = cv2.Canny(imagen, umbral_canny1, umbral_canny2, apertureSize=aperture_size)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, umbral_hough)

    imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            x1 = int(rho * np.cos(theta) + 1000 * (-np.sin(theta)))
            y1 = int(rho * np.sin(theta) + 1000 * (np.cos(theta)))
            x2 = int(rho * np.cos(theta) - 1000 * (-np.sin(theta)))
            y2 = int(rho * np.sin(theta) - 1000 * (np.cos(theta)))
            cv2.line(imagen_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.imshow(edges, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

    return edges, lines


def detectar_circulos(imagen, umbral_binario=107, usar_umbral=True,
                      dp=1, min_dist=20, param1=50, param2=30, min_radius=50, max_radius=80,
                      titulo='Círculos Detectados'):
    """
    Detecta círculos en una imagen usando Canny + Transformada de Hough.

    Parámetros:
    - imagen: Imagen en escala de grises (np.ndarray)
    - umbral_binario: Valor para binarizar la imagen antes de detectar bordes
    - usar_umbral: Si True, aplica umbralización antes de Canny
    - dp, min_dist, param1, param2, min_radius, max_radius: Parámetros de HoughCircles
    - titulo: Título del gráfico

    Retorna:
    - imagen_resultado: Imagen con círculos dibujados
    - circles: Lista de círculos detectados
    """
    if imagen is None:
        print("La imagen es inválida.")
        return None, None

    imagen_procesada = imagen.copy()
    if usar_umbral:
        _, imagen_procesada = cv2.threshold(imagen_procesada, umbral_binario, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(imagen_procesada, 50, 150, apertureSize=7)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                                param1=param1, param2=param2,
                                minRadius=min_radius, maxRadius=max_radius)

    imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(imagen_color, (x, y), r, (0, 255, 0), 4)
            cv2.circle(imagen_color, (x, y), 2, (0, 0, 255), 3)

    plt.imshow(imagen_color, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

    print(circles)
    print("Los dos primeros números son las coordenadas (x, y) del centro del círculo.")
    print("El tercer número es el radio del círculo.")

    return imagen_color, circles
