import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage import img_as_ubyte
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
import math

'''
FUNCIONES

Descriptores de Textura
├── HOG
│   └── Histogram of Oriented Gradients
├── Momentos_de_Hu
└── Estadísticos
    ├── Primer_Orden
    │   ├── Media
    │   ├── Varianza
    │   ├── Desviación
    │   └── Entropía
    └── Segundo_Orden
        ├── Homogeneidad
        ├── Contraste
        ├── Disimilaridad
        ├── Media
        ├── Desviación_Estándar
        ├── Entropía
        └── Energía

Detección de Bordes
├── Laplaciano_de_Gauss
└── Optical_Flow
    └── Flujo_Óptico Farneback

Detección de Formas
├── Transformada_de_Hough
│   ├── Detección_de_Rectas
│   └── Detección_de_Circunferencias
└── Segmentación
    └── GrabCut (Basado en Grafos)

Métodos Avanzados de Características
├── SIFT
├── SURF (No se puede implementar)
├── ORB
├── KAZE
└── AKAZE

'''
def momentos_de_hu(imagen, umbral=127, suavizado=False, usar_canny=False):
    if imagen is None:
        return None

    if suavizado:
        imagen = cv2.GaussianBlur(imagen, (5, 5), 0)

    if usar_canny:
        imagen = cv2.Canny(imagen, 100, 200)
    else:
        _, imagen = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)

    moments = cv2.moments(imagen)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments


def aplicar_sift_con_preprocesamiento(imagen: np.ndarray) -> dict:
    """
    Aplica el algoritmo SIFT a una imagen con preprocesamiento mediante
    filtros gaussianos y diferencia de gaussianas.

    Parámetros:
        imagen (np.ndarray): Imagen a procesar en formato BGR o escala de grises.

    Retorna:
        dict: Diccionario que contiene:
            - 'imagen_con_keypoints': Imagen con los puntos clave dibujados
            - 'keypoints': Lista de puntos clave detectados
            - 'descriptors': Matriz de descriptores asociados
    """
    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif len(imagen.shape) == 2:
        imagen_gris = imagen
    else:
        raise ValueError("Formato de imagen no válido. Se espera imagen BGR o escala de grises.")

    # 1. Definir valores de sigma para el filtro Gaussiano
    sigma_values = [0.5, 2, 5.5, 8, 11, 15.5, 20, 24.5]
    imagenes_filtradas = {}

    # 2. Aplicar el filtro gaussiano con distintos valores de sigma
    for sigma in sigma_values:
        ksize = int(2 * (sigma * 3) + 1)
        ksize += 1 if ksize % 2 == 0 else 0  # Asegurar impar
        imagen_filtrada = cv2.GaussianBlur(imagen_gris, (ksize, ksize), sigma)
        imagenes_filtradas[sigma] = imagen_filtrada

    # 3. Calcular diferencias de gaussianas (DoG)
    diferencias_dog = []
    for i in range(len(sigma_values) - 1):
        sigma1 = sigma_values[i]
        sigma2 = sigma_values[i + 1]
        dog = cv2.absdiff(imagenes_filtradas[sigma1], imagenes_filtradas[sigma2])
        diferencias_dog.append(dog)

    # 4. Crear el detector SIFT
    sift = cv2.SIFT_create()

    # 5. Aplicar SIFT sobre la imagen original
    keypoints, descriptors = sift.detectAndCompute(imagen_gris, None)

    # 6. Dibujar los puntos clave
    imagen_con_keypoints = cv2.drawKeypoints(
        imagen_gris, 
        keypoints, 
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return {
        "imagen_con_keypoints": imagen_con_keypoints,
        "keypoints": keypoints,
        "descriptors": descriptors
    }

def laplaciano_de_gauss(imagen: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Aplica el operador Laplaciano de Gauss para detectar bordes en una imagen.

    Parámetros:
        imagen (np.ndarray): Imagen de entrada (escala de grises o BGR).
        kernel_size (int): Tamaño del kernel gaussiano (debe ser impar y > 1).

    Retorna:
        np.ndarray: Imagen procesada con el operador Laplaciano de Gauss (valores absolutos en uint8).
    """
    if imagen is None:
        raise ValueError("La imagen no puede ser None.")
    
    # Convertir a escala de grises si la imagen es BGR
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavizado gaussiano
    imagen_suavizada = cv2.GaussianBlur(imagen, (kernel_size, kernel_size), 0)

    # Aplicar el operador Laplaciano
    laplaciano = cv2.Laplacian(imagen_suavizada, cv2.CV_64F)

    # Convertir a valores absolutos y a uint8
    laplaciano = np.uint8(np.absolute(laplaciano))

    return laplaciano

def segmentar_grabcut(imagen: np.ndarray, rect: tuple, iteraciones: int = 5) -> dict:
    """
    Aplica el algoritmo GrabCut para segmentar el objeto principal de una imagen.

    Parámetros:
        imagen (np.ndarray): Imagen de entrada en formato BGR.
        rect (tuple): Rectángulo de inicio para la segmentación (x, y, ancho, alto).
        iteraciones (int): Número de iteraciones para el algoritmo (por defecto 5).

    Retorna:
        dict: Diccionario con:
            - 'mascara': Máscara resultante de la segmentación.
            - 'imagen_segmentada': Imagen con el fondo eliminado (solo primer plano).
    """
    if imagen is None:
        raise ValueError("La imagen no puede ser None.")
    if len(imagen.shape) != 3 or imagen.shape[2] != 3:
        raise ValueError("La imagen debe estar en formato BGR (3 canales).")

    # Inicializar la máscara y los modelos de fondo/primer plano
    mask = np.zeros(imagen.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Aplicar GrabCut
    cv2.grabCut(imagen, mask, rect, bgd_model, fgd_model, iteraciones, cv2.GC_INIT_WITH_RECT)

    # Crear la máscara binaria: 1 para primer plano, 0 para fondo
    mask_binaria = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')

    # Aplicar la máscara sobre la imagen original
    imagen_segmentada = imagen * mask_binaria[:, :, np.newaxis]

    return {
        "mascara": mask_binaria,
        "imagen_segmentada": imagen_segmentada
    }

def extraer_kaze(imagen: np.ndarray) -> dict:
    """
    Extrae puntos clave y descriptores usando el algoritmo KAZE.

    Parámetros:
        imagen (np.ndarray): Imagen de entrada (color o escala de grises).

    Retorna:
        dict: Diccionario que contiene:
            - 'keypoints': Lista de puntos clave detectados.
            - 'descriptors': Matriz de descriptores (np.ndarray).
            - 'imagen_con_keypoints': Imagen con puntos clave dibujados (np.ndarray).
    """
    if imagen is None:
        raise ValueError("La imagen no puede ser None.")

    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif len(imagen.shape) == 2:
        imagen_gris = imagen
    else:
        raise ValueError("Formato de imagen no válido. Se espera imagen BGR o en escala de grises.")

    # Crear el detector KAZE
    kaze = cv2.KAZE_create()

    # Detectar puntos clave y calcular descriptores
    keypoints, descriptors = kaze.detectAndCompute(imagen_gris, None)

    # Dibujar puntos clave en una copia de la imagen original
    imagen_con_keypoints = cv2.drawKeypoints(imagen_gris, keypoints, None, color=(0, 255, 0))

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "imagen_con_keypoints": imagen_con_keypoints
    }

def extraer_akaze(imagen: np.ndarray) -> dict:
    """
    Detecta puntos clave y extrae descriptores utilizando el algoritmo AKAZE.

    Parámetros:
        imagen (np.ndarray): Imagen de entrada (BGR o escala de grises).

    Retorna:
        dict: Diccionario con:
            - 'keypoints': Lista de puntos clave (cv2.KeyPoint).
            - 'descriptors': np.ndarray con descriptores binarios.
            - 'imagen_con_keypoints': Imagen con los puntos clave dibujados.
    """
    if imagen is None:
        raise ValueError("La imagen no puede ser None.")

    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif len(imagen.shape) == 2:
        imagen_gris = imagen
    else:
        raise ValueError("Formato de imagen no válido.")

    # Crear el detector AKAZE
    akaze = cv2.AKAZE_create()

    # Detectar puntos clave y calcular descriptores
    keypoints, descriptors = akaze.detectAndCompute(imagen_gris, None)

    # Dibujar los puntos clave en una copia de la imagen
    imagen_con_keypoints = cv2.drawKeypoints(
        imagen_gris, keypoints, None, color=(0, 255, 0)
    )

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "imagen_con_keypoints": imagen_con_keypoints
    }

def extraer_orb(imagen: np.ndarray) -> dict:
    """
    Detecta puntos clave y extrae descriptores usando el algoritmo ORB.

    Parámetros:
        imagen (np.ndarray): Imagen de entrada (BGR o en escala de grises).

    Retorna:
        dict: Diccionario con:
            - 'keypoints': Lista de objetos cv2.KeyPoint.
            - 'descriptors': np.ndarray con los descriptores.
            - 'imagen_con_keypoints': Imagen con los puntos clave dibujados.
    """
    if imagen is None:
        raise ValueError("La imagen no puede ser None.")

    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif len(imagen.shape) == 2:
        imagen_gris = imagen
    else:
        raise ValueError("Formato de imagen no válido.")

    # Crear el detector ORB
    orb = cv2.ORB_create()

    # Detectar puntos clave y calcular descriptores
    keypoints, descriptors = orb.detectAndCompute(imagen_gris, None)

    # Dibujar puntos clave con orientación y escala
    imagen_con_keypoints = cv2.drawKeypoints(
        imagen_gris, keypoints, None, color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "imagen_con_keypoints": imagen_con_keypoints
    }

def extraer_caracteristicas_hog(imagen: np.ndarray) -> np.ndarray:
    """
    Extrae el descriptor HOG (Histogram of Oriented Gradients) de una imagen.

    Parámetros:
        imagen (np.ndarray): Imagen en formato BGR (color) o escala de grises.

    Retorna:
        np.ndarray: Vector de características HOG como un arreglo 1D.
    """
    if imagen is None:
        raise ValueError("La imagen no puede ser None.")

    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif len(imagen.shape) == 2:
        imagen_gris = imagen
    else:
        raise ValueError("Formato de imagen no válido. Se espera imagen BGR o en escala de grises.")

    # Crear el descriptor HOG con parámetros por defecto
    hog = cv2.HOGDescriptor()

    # Calcular el descriptor
    caracteristicas = hog.compute(imagen_gris)

    # Aplanar el resultado para facilidad de uso
    return caracteristicas.flatten()

def metodos_estadisticos_primer_orden(imagen: np.ndarray) -> dict:
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

def metodos_estadisticos_segundo_orden(imagen: np.ndarray) -> dict:
    """
    Calcula propiedades de textura basadas en la matriz de co-ocurrencia (GLCM).

    Parámetros:
        imagen (np.ndarray): Imagen en escala de grises (matriz 2D).

    Retorna:
        dict: Diccionario con contraste, homogeneidad, disimilitud, energía,
              correlación, media, desviación estándar y entropía de la GLCM.
    """
    if imagen is None or len(imagen.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises (matriz 2D).")

    # Convertir imagen a tipo adecuado para GLCM
    imagen_ubyte = img_as_ubyte(imagen)

    # Calcular GLCM con desplazamiento de 1 píxel en dirección horizontal
    glcm = graycomatrix(imagen_ubyte, distances=[1], angles=[0], symmetric=True, normed=True)

    # Propiedades estadísticas de la GLCM
    contraste = graycoprops(glcm, prop='contrast')[0, 0]
    homogeneidad = graycoprops(glcm, prop='homogeneity')[0, 0]
    disimilitud = graycoprops(glcm, prop='dissimilarity')[0, 0]
    energia = graycoprops(glcm, prop='energy')[0, 0]
    correlacion = graycoprops(glcm, prop='correlation')[0, 0]

    # Media y desviación estándar de la GLCM
    media_glcm = np.mean(glcm)
    desviacion_glcm = np.std(glcm)

    # Entropía de la GLCM
    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]  # Evitar log(0)
    entropia_glcm = -np.sum(glcm_flat * np.log(glcm_flat))

    return {
        "contraste": contraste,
        "homogeneidad": homogeneidad,
        "disimilitud": disimilitud,
        "energia": energia,
        "correlacion": correlacion,
        "media_glcm": media_glcm,
        "desviacion_glcm": desviacion_glcm,
        "entropia_glcm": entropia_glcm
    }

def detectar_lineas_Hough(imagen, umbral_canny1=50, umbral_canny2=150, aperture_size=7, umbral_hough=150, titulo='Líneas Detectadas'):
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


def detectar_circulos_Hough(imagen, umbral_binario=107, usar_umbral=True,
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