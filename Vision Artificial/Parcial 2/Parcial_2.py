import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage import img_as_ubyte
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
import math
from matplotlib.widgets import Button
import os
import sys

#Descriptores de Textura
from funciones.funciones import extraer_caracteristicas_hog #SOLO TEXTO - RETORNA ARRAY
from funciones.funciones import momentos_de_hu #SOLO TEXTO - RETORNA TEXTO
from funciones.funciones import metodos_estadisticos_primer_orden #SOLO TEXTO - RETORNA DICCIONARIO CON TEXTO
from funciones.funciones import metodos_estadisticos_segundo_orden #SOLO TEXTO - RETORNA DICCIONARIO CON TEXTO

#Detección de Bordes
from funciones.funciones import laplaciano_de_gauss #GRAFICA - RETORNA IMAGEN 
from funciones.funciones import flujo_optico_farneback #GRAFICA - RETORNA IMAGEN

#Detección de Formas
from funciones.funciones import detectar_lineas_Hough #GRAFICA - RETORNA IMAGEN Y TEXTO
from funciones.funciones import detectar_circulos_Hough #GRAFICA Y TEXTO - RETORNA IMAGEN Y TEXTO
from funciones.funciones import segmentar_grabcut #GRAFICA - RETORNA DICCIONARIO CON IMAGEN Y MASCARA

#Métodos Avanzados de Características
from funciones.funciones import aplicar_sift_con_preprocesamiento #GRAFICA - IMPRIME IMAGEN Y RETORNA TEXTO
from funciones.funciones import extraer_orb #GRAFICA Y TEXTO - RETORNA DICCIONARIO CON IMAGEN Y TEXTO
from funciones.funciones import extraer_kaze #GRAFICA Y TEXTO - RETORNA DICCIONARIO CON IMAGEN Y TEXTO
from funciones.funciones import extraer_akaze #GRAFICA Y TEXTO - RETORNA DICCIONARIO CON IMAGEN Y TEXTO

def procesar_imagenes(imagenes):
     for idx, img in enumerate(imagenes):
        # Convertir a escala de grises si es necesario
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # 1. Laplaciano de Gauss
        laplaciano = laplaciano_de_gauss(img)

        # 2. Detección de líneas
        imagen_lineas, _ = detectar_lineas_Hough(img_gray)

        # 3. Detección de círculos
        imagen_circulos, _ = detectar_circulos_Hough(img_gray)

        # 4. Segmentación con GrabCut
        alto, ancho = img.shape[:2]
        rect = (int(ancho * 0.25), int(alto * 0.25), int(ancho * 0.5), int(alto * 0.5))
        resultado_grabcut = segmentar_grabcut(img, rect)
        imagen_grabcut = resultado_grabcut["imagen_segmentada"]

        # Secciones organizadas (solo 2: Laplaciano y Formas)
        secciones = [
            ([laplaciano], ["laplaciano_de_gauss"], "Detección de Bordes - Laplaciano de Gauss"),
            (
                [imagen_lineas, imagen_circulos, imagen_grabcut],
                ["detectar_lineas_Hough", "detectar_circulos_Hough", "segmentar_grabcut"],
                "Detección y Segmentación de Formas"
            )
        ]

        for imagenes_seccion, nombres, titulo in secciones:
            mostrar_imagenes(imagenes_seccion, nombres, titulo)

def mostrar_imagenes(imagenes, nombres, titulo):
    num_imagenes = len(imagenes)
    fig, axes = plt.subplots(1, num_imagenes, figsize=(5 * num_imagenes, 5))
    manager = plt.get_current_fig_manager()

    #intentar poner en pantalla completa
    try:
        manager.window.state('zoomed')  # Para TkAgg (Windows y algunos sistemas)
    except AttributeError:
        try:
            manager.full_screen_toggle()  # Para otros backends
        except AttributeError:
            pass
    
    #Si solo hay una imagen se castea a lista para poderla iterar
    if num_imagenes == 1:
        axes = [axes]
    
    for ax, img, nombre in zip(axes, imagenes, nombres): #agrupa elementos de las tres listas por posición en una sola estructura iterable
        #Si la imagen es en escala de grises
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else: #Si la imagen es a color
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(nombre, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold', color='darkblue')
    
    ax_next = plt.axes([0.8, 0.01, 0.1, 0.05])
    btn_next = Button(ax_next, 'Next')
    btn_next.on_clicked(lambda event: plt.close(fig))
    
    plt.show()

def obtener_imagenes_de_carpeta(ruta_carpeta):
    """
    Carga todas las imágenes de una carpeta y las devuelve como una lista de arrays de NumPy.

    :param ruta_carpeta: str, ruta de la carpeta donde están las imágenes.
    :return: list, lista de imágenes en formato numpy.ndarray.
    """
    # Define las extensiones de archivos que se considerarán imágenes
    extensiones_imagenes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    imagenes = []  # Lista donde se almacenarán las imágenes cargadas

    # Verificar si la carpeta existe
    if not os.path.isdir(ruta_carpeta):  # Comprueba si la ruta proporcionada es un directorio válido
        print(f"La ruta '{ruta_carpeta}' no es una carpeta válida.")  # Mensaje de error si no existe
        return []  # Retorna una lista vacía

    # Recorrer los archivos dentro de la carpeta
    for archivo in os.listdir(ruta_carpeta):  # Itera sobre todos los archivos en la carpeta
        ruta_imagen = os.path.join(ruta_carpeta, archivo)  # Obtiene la ruta completa del archivo
        
        # Verificar si la extensión del archivo está en la lista de imágenes permitidas
        if os.path.splitext(archivo)[1].lower() in extensiones_imagenes:
            img = cv2.imread(ruta_imagen)  # Lee la imagen con OpenCV
            
            if img is not None:  # Verifica que la imagen se haya cargado correctamente
                imagenes.append(img)  # Agrega la imagen a la lista
            else:
                print(f"Advertencia: No se pudo cargar la imagen {archivo}")  # Mensaje si la imagen no se carga

    return imagenes  # Retorna la lista de imágenes cargadas


# Cargar imágenes de prueba
imagenes = obtener_imagenes_de_carpeta("VisionArtificial\\Vision Artificial\\Parcial 2\\images")
procesar_imagenes(imagenes)