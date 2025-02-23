import os
import cv2
import sys
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from Funciones.Funciones import ChangeSize
from Funciones.Funciones import InterpolationLinear
from Funciones.Funciones import InterpolationNearest
from Funciones.Funciones import And_Op
from Funciones.Funciones import Or_Op
from Funciones.Funciones import Not_Op
from Funciones.Funciones import Sum_Op
from Funciones.Funciones import Diff_Op
from Funciones.Funciones import Mult_Op
from Funciones.Funciones import Div_Op
from Funciones.Funciones import ToBinary
from Funciones.Funciones import ToGrayScale
from Funciones.Funciones import RotateImage
from Funciones.Funciones import SharpenImage
from Funciones.Funciones import RelieveImage
from Funciones.Funciones import BlurFilter
from Funciones.Funciones import CannyEdgeDetection
from Funciones.Funciones import GaussianBlur
from Extraccion.extracciones import extraer_caracteristicas
from Extraccion.extracciones import guardar_caracteristicas_csv


def obtener_imagenes_de_carpeta(ruta_carpeta):
    """
    Carga todas las imágenes de una carpeta y las devuelve como una lista de arrays de NumPy.

    :param ruta_carpeta: str, ruta de la carpeta donde están las imágenes.
    :return: list, lista de imágenes en formato numpy.ndarray.
    """
    extensiones_imagenes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    imagenes = []

    # Verificar si la carpeta existe
    if not os.path.isdir(ruta_carpeta):
        print(f"La ruta '{ruta_carpeta}' no es una carpeta válida.")
        return []

    # Recorrer los archivos de la carpeta
    for archivo in os.listdir(ruta_carpeta):
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        
        # Verificar si la extensión es de imagen
        if os.path.splitext(archivo)[1].lower() in extensiones_imagenes:
            img = cv2.imread(ruta_imagen)  # Leer imagen con OpenCV
            
            if img is not None:  # Verificar que la imagen se cargó correctamente
                imagenes.append(img)
            else:
                print(f"Advertencia: No se pudo cargar la imagen {archivo}")

    return imagenes


def aplicar_procesamiento(imagenes):
    """
    Procesa todas las imágenes de un array y las devuelve como una lista de imagenes en arrays de NumPy.

    :param imagenes: arrays de Numpy, array donde están las imágenes a procesar.
    :return: arrays de Numpy, lista de imágenes procesadas por cada uno de los metodos.
    """
    resultados = []
    
    for imagen in imagenes:

        procesadas = []

        procesadas.append(("ChangeSize", ChangeSize(imagen, 500, 500)))
        procesadas.append(("InterpolationLinear", InterpolationLinear(imagen, 500, 500)))
        procesadas.append(("InterpolationNearest", InterpolationNearest(imagen, 500, 500)))
        procesadas.append(("And_Op", And_Op(imagen, imagenes[0], 127)))
        procesadas.append(("Or_Op", Or_Op(imagen, imagenes[0], 127)))
        procesadas.append(("Not_Op", Not_Op(imagen, 127)))
        procesadas.append(("Sum_Op", Sum_Op(imagen, 50)))
        procesadas.append(("Diff_Op", Diff_Op(imagen, 50)))
        procesadas.append(("Mult_Op", Mult_Op(imagen, 1.5)))
        procesadas.append(("Div_Op", Div_Op(imagen, 2)))
        procesadas.append(("ToBinary", ToBinary(imagen, 127)))
        procesadas.append(("ToGrayScale", ToGrayScale(imagen)))
        procesadas.append(("RotateImage", RotateImage(imagen)))
        procesadas.append(("SharpenImage", SharpenImage(imagen)))
        procesadas.append(("RelieveImage", RelieveImage(imagen)))
        procesadas.append(("BlurFilter", BlurFilter(imagen)))
        procesadas.append(("CannyEdgeDetection", CannyEdgeDetection(imagen)))
        procesadas.append(("GaussianBlur", GaussianBlur(imagen)))
        
        resultados.append(procesadas)
    
    return resultados

def imprimir_resultados(resultados, num_columnas=6):
    """
    Muestra cada imagen original con sus variaciones en una ventana independiente.
    
    :param resultados: Lista de listas. Cada sublista contiene tuplas (etiqueta, imagen).
    :param num_columnas: Número de columnas en la cuadrícula de cada ventana.
    """
    for i, procesadas in enumerate(resultados):
        num_imagenes = len(procesadas)
        num_filas = (num_imagenes + num_columnas - 1) // num_columnas  # Calcular filas necesarias

        fig, axes = plt.subplots(num_filas, num_columnas, figsize=(15, num_filas * 3))
        axes = axes.flatten()  # Convertir en una lista plana para iterar más fácilmente

        # Agregar imágenes procesadas
        for index, (etiqueta, imagen) in enumerate(procesadas):
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) #Esto es necesario ya que sino muestra casi todas las imagenes en tonalidades verdes, debido a matplotlib
            axes[index].imshow(imagen_rgb)
            axes[index].set_title(etiqueta, fontsize=10)
            axes[index].axis("off")

        # Deshabilitar los ejes sobrantes si hay más subplots que imágenes
        for j in range(index + 1, len(axes)):
            axes[j].axis("off")

        # Mostrar la ventana con el título de la imagen original
        fig.suptitle(f"Variaciones de Imagen {i + 1}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

def main():

    imagenes = obtener_imagenes_de_carpeta("VisionArtificial/Vision Artificial/Laboratorio 1/imagenes")
    
    resultados = aplicar_procesamiento(imagenes)

    #imprimir_resultados(resultados)

    caracteristicas_lista = []
    for i, procesadas in enumerate(resultados):
        for etiqueta, img in procesadas:
            caracteristicas_lista.append(extraer_caracteristicas(img, etiqueta))

    guardar_caracteristicas_csv(caracteristicas_lista)

if __name__ == "__main__":
    main()