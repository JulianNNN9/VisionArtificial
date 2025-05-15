import cv2
import numpy as np
from skimage.feature import hog

def extraer_caracteristicas_hog(image):
    """
    Extrae las características HOG (Histogram of Oriented Gradients) de una imagen.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Características HOG como un vector numpy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    features, hog_image = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features  # Devuelve un array de características HOG


def metodos_estadisticos_primer_orden(image):
    """
    Calcula los métodos estadísticos de primer orden: media, varianza, contraste, etc.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Diccionario con valores de las estadísticas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    variance = np.var(gray)
    contrast = np.std(gray)
    entropy = -np.sum((gray/255) * np.log2(gray/255 + 1e-6))  # Aproximación de entropía
    return {"mean": mean, "variance": variance, "contrast": contrast, "entropy": entropy}


def momentos_de_hu(image):
    """
    Calcula los momentos de Hu (invariantes a transformación).
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Valores de los momentos de Hu.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    huMoments = cv2.HuMoments(moments).flatten()
    return huMoments


def laplaciano_de_gauss(image):
    """
    Aplica el filtro Laplaciano de Gauss para detectar bordes.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Imagen procesada con bordes detectados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.sum(np.abs(laplacian))  # Devuelve la suma absoluta de los valores del laplaciano


def detectar_circulos_Hough(image):
    """
    Detecta círculos en la imagen utilizando la Transformada de Hough.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Número de círculos detectados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return len(circles[0])  # Devuelve la cantidad de círculos detectados
    return 0


def segmentar_grabcut(image):
    """
    Segmenta una imagen utilizando el algoritmo GrabCut.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Área de la región segmentada.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)  # Rectángulo de inicialización
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1)  # Reemplazar fondo con 0
    return np.sum(mask2)  # Devuelve el área de la región segmentada


def extraer_orb(image):
    """
    Extrae puntos clave utilizando el detector ORB (Oriented FAST and Rotated BRIEF).
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Número de puntos clave detectados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return len(keypoints)  # Devuelve el número de puntos clave


def ToGrayScale(image):
    """
    Convierte una imagen a escala de grises.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Imagen en escala de grises.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)  # Devuelve el valor promedio de los píxeles en la imagen en escala de grises


def umbralizacion_adaptativa(image):
    """
    Aplica umbralización adaptativa para segmentar la imagen.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Número de píxeles binarizados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    return np.sum(thresholded)  # Devuelve el número de píxeles binarizados


def segmentacion_kmeans(image, k=3):
    """
    Segmenta una imagen utilizando el algoritmo K-means basado en color.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :param k: Número de clusters.
    :return: Suma de los valores de los píxeles de la imagen segmentada.
    """
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape(image.shape)
    return np.sum(segmented_image)  # Devuelve la suma de los valores de los píxeles


def detectar_bordes_Canny(image):
    """
    Detecta bordes utilizando el algoritmo de Canny.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Número de bordes detectados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Umbrales de Canny
    return np.sum(edges)  # Devuelve el número de bordes detectados


def SharpenImage(image):
    """
    Aplica un filtro de nitidez a la imagen para resaltar detalles.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Valor promedio de los píxeles de la imagen mejorada.
    """
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.mean(sharpened)  # Devuelve el valor promedio de los píxeles de la imagen mejorada
import cv2
import numpy as np

def calcular_histograma_color(image):
    """
    Calcula el histograma de color de una imagen y devuelve el color dominante.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: El color dominante de la imagen basado en el histograma.
    """
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Dividir la imagen en los canales H, S, V
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # Calcular el histograma para el canal Hue (H)
    hist_h = cv2.calcHist([h_channel], [0], None, [180], [0, 180])  # Histograma para tonalidades
    hist_h = hist_h / hist_h.sum()  # Normalizar el histograma
    
    # Encontrar el valor dominante de tonalidad (valor máximo en el histograma)
    dominant_hue = np.argmax(hist_h)  # La tonalidad con mayor frecuencia
    
    # Calcular el histograma para el canal Saturation (S)
    hist_s = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
    hist_s = hist_s / hist_s.sum()  # Normalizar el histograma
    
    # Encontrar el valor dominante de saturación
    dominant_saturation = np.argmax(hist_s)
    
    # Calcular el histograma para el canal Value (V)
    hist_v = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
    hist_v = hist_v / hist_v.sum()  # Normalizar el histograma
    
    # Encontrar el valor dominante de luminosidad
    dominant_value = np.argmax(hist_v)
    
    return dominant_hue, dominant_saturation, dominant_value


def detectar_tonalidades(image):
    """
    Detecta la tonalidad (hue) predominante en una imagen utilizando el espacio de color HSV.
    
    :param image: Imagen de entrada en formato np.array (BGR).
    :return: Valor promedio del canal Hue (tonalidad) de la imagen.
    """
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extraer solo el canal Hue (H)
    hue_channel = hsv[:, :, 0]
    
    # Calcular el valor promedio de la tonalidad (Hue)
    mean_hue = np.mean(hue_channel)
    
    return mean_hue  # Devuelve el valor promedio de la tonalidad (Hue)
