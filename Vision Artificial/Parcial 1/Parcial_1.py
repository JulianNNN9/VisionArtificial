import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import get_backend

# Ajustar rutas para importar funciones personalizadas
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

#Filtros (FALTA CANNY)
from Funciones.Funciones import SharpenImage
from Funciones.Funciones import RelieveImage
from Funciones.Funciones import BlurFilter
from Funciones.Funciones import CannyEdgeDetection
from Funciones.Funciones import GaussianBlur

#Filtros morfologicos
from Funciones.Funciones import aplicar_apertura
from Funciones.Funciones import aplicar_cierre
from Funciones.Funciones import aplicar_gradiente

#Cambios de Tonalidad (FALTAN ULTIMAS DOS)
from Funciones.Funciones import ToBinary
from Funciones.Funciones import ToGrayScale
from Funciones.Funciones import umbralizar_imagen
from Funciones.Funciones import umbralizacion_adaptativa

#Operaciones Lógicas (FALTAN TODAS)
from Funciones.Funciones import And_Op
from Funciones.Funciones import Or_Op
from Funciones.Funciones import Not_Op

#Operaciones Matemáticas
from Funciones.Funciones import Sum_Op
from Funciones.Funciones import Diff_Op
from Funciones.Funciones import Mult_Op
from Funciones.Funciones import Div_Op

#Ampliación y Reducción
from Funciones.Funciones import ChangeSize
from Funciones.Funciones import InterpolationLinear
from Funciones.Funciones import InterpolationNearest

#Segmentación (FALTA CANNY)
from Funciones.Funciones import segmentacion_kmeans
from Funciones.Funciones import segmentacion_watershed
from Funciones.Funciones import detectar_bordes_Canny

def procesar_imagenes(imagenes):
    for idx, img in enumerate(imagenes):
        secciones = [
            ([ChangeSize(img, 400, 400), InterpolationLinear(img, 400, 400), InterpolationNearest(img, 400, 400)], ["ChangeSize", "InterpolationLinear", "InterpolationNearest"], "Ampliación y reducción"),
            ([Sum_Op(img, 50), Diff_Op(img, 50), Mult_Op(img, 2), Div_Op(img, 2)], ["Sum_Op", "Diff_Op", "Mult_Op", "Div_Op"], "Operaciones matemáticas"),
            ([ToGrayScale(img), ToBinary(img, 127), umbralizar_imagen(ToGrayScale(img)), umbralizacion_adaptativa(ToGrayScale(img))], ["ToGrayScale", "ToBinary", "umbralizar_imagen", "umbralizacion_adaptativa"], "Cambios de tonalidad"),
            ([SharpenImage(img), RelieveImage(img), BlurFilter(img), GaussianBlur(img), CannyEdgeDetection(img)], ["SharpenImage", "RelieveImage", "BlurFilter", "GaussianBlur", "CannyEdgeDetection"], "Filtros"),
            ([aplicar_apertura(ToBinary(img, 127)), aplicar_cierre(ToBinary(img, 127)), aplicar_gradiente(ToBinary(img, 127))], ["aplicar_apertura", "aplicar_cierre", "aplicar_gradiente"], "Operaciones morfológicas"),
            ([And_Op(img, img, 127), Or_Op(img, img, 127), Not_Op(ToGrayScale(img), 127)], ["And_Op", "Or_Op", "Not_Op"], "Operaciones lógicas"),
            ([segmentacion_kmeans(img), segmentacion_watershed(img), detectar_bordes_Canny(img)], ["segmentacion_kmeans", "segmentacion_watershed", "detectar_bordes_Canny"], "Segmentación")
        ]
        
        for imagenes_seccion, nombres, titulo in secciones:
            mostrar_imagenes(imagenes_seccion, nombres, titulo)

def mostrar_imagenes(imagenes, nombres, titulo):
    num_imagenes = len(imagenes)
    fig, axes = plt.subplots(1, num_imagenes, figsize=(5 * num_imagenes, 5))
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')  # Para TkAgg (Windows y algunos sistemas)
    except AttributeError:
        try:
            manager.full_screen_toggle()  # Para otros backends
        except AttributeError:
            pass
    
    if num_imagenes == 1:
        axes = [axes]
    
    for ax, img, nombre in zip(axes, imagenes, nombres):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
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
imagenes = obtener_imagenes_de_carpeta("VisionArtificial\Vision Artificial\Parcial 1\images")
procesar_imagenes(imagenes)

