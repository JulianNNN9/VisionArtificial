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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.array([np.mean(laplacian), np.std(laplacian)])

def detectar_circulos_Hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return np.array([len(circles[0]), np.mean(circles[0][:, 2])])  # Número de círculos y radio promedio
    return np.array([0, 0])

def segmentar_grabcut(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1)
    return np.array([np.sum(mask2), np.mean(mask2)])

def extraer_orb(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is not None:
        return np.array([len(keypoints), np.mean(descriptors)])
    return np.array([len(keypoints), 0])

def ToGrayScale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.array([np.mean(gray), np.std(gray)])

def umbralizacion_adaptativa(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    return np.array([np.mean(thresholded), np.std(thresholded)])

def segmentacion_kmeans(image, k=3):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return np.array([np.mean(center), np.std(center)])

def detectar_bordes_Canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.array([np.sum(edges), np.mean(edges)])

def SharpenImage(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.array([np.mean(sharpened), np.std(sharpened)])

def calcular_histograma_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.array([np.mean(h), np.mean(s), np.mean(v)])

def detectar_tonalidades(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    return np.array([np.mean(h), np.std(h)])