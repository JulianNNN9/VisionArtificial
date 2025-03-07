import cv2
import numpy as np

def ChangeSize(img, tamanioX, tamanioY):
    # Cambiar el tamaño de la imagen (Reducción y Amplificación)
    resized_img = cv2.resize(img, (tamanioX, tamanioY))  # Cambiar el tamaño a 400x400 píxeles

    return resized_img

def InterpolationLinear(img, tamanioX, tamanioY):

    # Cambiar tamaño usando diferentes métodos de interpolación
    resized_bilinear = cv2.resize(img, (tamanioX, tamanioY), interpolation=cv2.INTER_LINEAR)  # Bilineal

    return resized_bilinear
    

def InterpolationNearest(img, tamanioX, tamanioY):

    resized_nearest = cv2.resize(img, (tamanioX, tamanioY), interpolation=cv2.INTER_NEAREST)  # Vecino más cercano

    return resized_nearest


def And_Op(img1, img2, umbral):
    
    img11 = ToGrayScale(img1)
    _, img1_bin = cv2.threshold(img11, umbral, 255, cv2.THRESH_BINARY)

    img21 = ToGrayScale(img2)
    _, img2_bin = cv2.threshold(img21, umbral, 255, cv2.THRESH_BINARY)

    and_img = cv2.bitwise_and(img1_bin, img2_bin)

    return and_img

def Or_Op(img1, img2, umbral):

    img11 = ToGrayScale(img1)
    _, img1_bin = cv2.threshold(img11, umbral, 255, cv2.THRESH_BINARY)

    img21 = ToGrayScale(img2)
    _, img2_bin = cv2.threshold(img21, umbral, 255, cv2.THRESH_BINARY)

    or_img = cv2.bitwise_or(img1_bin, img2_bin)

    return or_img

def Not_Op(img, umbral):

    _, img_bin = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)

    not_img = cv2.bitwise_not(img_bin)

    return not_img

# Suma de imágenes
def Sum_Op (img, valor):
    sum_img = cv2.add(img, valor)
    return sum_img

# Resta de imágenes
def Diff_Op (ruta, valor):
    diff_img = cv2.subtract(ruta, valor)
    return diff_img

# Multiplicación de imágenes
def Mult_Op (ruta, valor):
    mult_img = cv2.multiply(ruta, valor)
    return mult_img

# División de imágenes
def Div_Op (ruta, valor):
    div_img = cv2.divide(ruta, valor)
    return div_img

# Cargar una imagen en color
def ToBinary (img, umbral):
   

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralizar la imagen para convertirla en blanco y negro (binaria)
    _, binary_img = cv2.threshold(gray_img, umbral, 255, cv2.THRESH_BINARY)

    return binary_img

def ToGrayScale (img):

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img

def RotateImage (img):
   
    # Obtener las dimensiones de la imagen
    (h, w) = img.shape[:2]

    # Establecer el centro de la imagen para rotarla
    center = (w // 2, h // 2)

    # Crear la matriz de rotación (por ejemplo, 45 grados)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)

    # Rotar la imagen
    rotated_img = cv2.warpAffine(img, M, (w, h))

    return rotated_img

def SharpenImage (img):
  

    # Crear un kernel para nítidez
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    # Aplicar el filtro de nítidez el segundo parametro de profundidad de la imagen
    # de salida ddepth
    sharpened_image = cv2.filter2D(img, 0, sharpen_kernel)

    return sharpened_image

def RelieveImage (img):

    # Crear un kernel para el filtro de relieve
    emboss_kernel = np.array([[-2, -1, 0],
                            [-1,  1, 1],
                            [ 0,  1, 2]])

    # Aplicar el filtro de relieve
    embossed_image = cv2.filter2D(img, -1, emboss_kernel)

    return embossed_image

def BlurFilter(img):

    # Aplicar el filtro de desenfoque
    blurred_image = cv2.blur(img, (50, 50))
    return blurred_image

def CannyEdgeDetection(img):
        # Cargar la imagen en escala de grises
    image = ToGrayScale(img)
    # Aplicar el filtro de detección de bordes (Canny)
    edges = cv2.Canny(image, 100, 200)
    return edges

def GaussianBlur(img):
        # Cargar la imagen
   
    # Aplicar el filtro de desenfoque gaussiano
    gaussian_blurred_image = cv2.GaussianBlur(img, (51, 51), 0)

    return gaussian_blurred_image


def cargar_imagen(ruta):
    """Carga una imagen en escala de grises."""
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

def umbralizar_imagen(image, threshold=127):
    """Convierte la imagen en binaria mediante umbralización."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def aplicar_apertura(image, kernel_size=(7, 7)):
    """Aplica la operación morfológica de apertura para eliminar ruido."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def aplicar_cierre(image, kernel_size=(7, 7)):
    """Aplica la operación morfológica de cierre para rellenar agujeros."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def detectar_contornos(image):
    """Detecta contornos en la imagen binaria cerrada."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def dibujar_contornos(image, contours):
    """Dibuja los contornos sobre la imagen original."""
    return cv2.drawContours(np.copy(image), contours, -1, (0, 255, 0), 2)

def calcular_caracteristicas(contours):
    """Calcula áreas y perímetros de los contornos detectados."""
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    return areas, perimeters

def aplicar_gradiente(image, kernel_size=(3, 3)):
    """Aplica la operación morfológica de gradiente para resaltar los bordes."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)