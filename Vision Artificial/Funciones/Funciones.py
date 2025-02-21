import cv2
import matplotlib.pyplot as plt
import numpy as np

def ChangeSize(ruta, tamanioX, tamanioY):
    img = cv2.imread(ruta)

    # Cambiar el tamaño de la imagen (Reducción y Amplificación)
    resized_img = cv2.resize(img, (tamanioX, tamanioY))  # Cambiar el tamaño a 400x400 píxeles

    return resized_img

def InterpolationLinear(ruta, tamanioX, tamanioY):
    img = cv2.imread(ruta)

    # Cambiar tamaño usando diferentes métodos de interpolación
    resized_bilinear = cv2.resize(img, (tamanioX, tamanioY), interpolation=cv2.INTER_LINEAR)  # Bilineal

    return resized_bilinear
    

def InterpolationNearest(ruta, tamanioX, tamanioY):
    img = cv2.imread(ruta)

    resized_nearest = cv2.resize(img, (tamanioX, tamanioY), interpolation=cv2.INTER_NEAREST)  # Vecino más cercano

    return resized_nearest


def And_Op(ruta1, ruta2, umbral):
    
    img1 = cv2.imread(ruta1, cv2.IMREAD_GRAYSCALE)
    _, img1_bin = cv2.threshold(img1, umbral, 255, cv2.THRESH_BINARY)

    img2 = cv2.imread(ruta2, cv2.IMREAD_GRAYSCALE)
    _, img2_bin = cv2.threshold(img2, umbral, 255, cv2.THRESH_BINARY)

    and_img = cv2.bitwise_and(img1_bin, img2_bin)

    return and_img

def Or_Op(ruta1, ruta2, umbral):

    img1 = cv2.imread(ruta1, cv2.IMREAD_GRAYSCALE)
    _, img1_bin = cv2.threshold(img1, umbral, 255, cv2.THRESH_BINARY)

    img2 = cv2.imread(ruta2, cv2.IMREAD_GRAYSCALE)
    _, img2_bin = cv2.threshold(img2, umbral, 255, cv2.THRESH_BINARY)

    or_img = cv2.bitwise_or(img1_bin, img2_bin)

    return or_img

def Not_Op(ruta, umbral):

    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)

    not_img = cv2.bitwise_not(img_bin)

    return not_img

# Suma de imágenes
def Sum_Op (ruta, valor):
    sum_img = cv2.add(ruta, valor)
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
def ToBinary (ruta, umbral):
    img = cv2.imread(ruta)

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralizar la imagen para convertirla en blanco y negro (binaria)
    _, binary_img = cv2.threshold(gray_img, umbral, 255, cv2.THRESH_BINARY)

    return binary_img

def ToGrayScale (ruta):
    img = cv2.imread(ruta)

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img

def RotateImage ():
    img = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg')

    # Obtener las dimensiones de la imagen
    (h, w) = img.shape[:2]

    # Establecer el centro de la imagen para rotarla
    center = (w // 2, h // 2)

    # Crear la matriz de rotación (por ejemplo, 45 grados)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)

    # Rotar la imagen
    rotated_img = cv2.warpAffine(img, M, (w, h))

def BlurFilter(ruta):
    # Cargar la imagen
    image = cv2.imread(ruta)

    # Aplicar el filtro de desenfoque
    blurred_image = cv2.blur(image, (50, 50))
    return blurred_image

def CannyEdgeDetection(ruta):
        # Cargar la imagen en escala de grises
    image = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    # Aplicar el filtro de detección de bordes (Canny)
    edges = cv2.Canny(image, 100, 200)
    return edges

def GaussianBlur(ruta):
        # Cargar la imagen
    image = cv2.imread(ruta)

    # Aplicar el filtro de desenfoque gaussiano
    gaussian_blurred_image = cv2.GaussianBlur(image, (51, 51), 0)

