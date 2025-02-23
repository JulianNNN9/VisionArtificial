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
from Extraccion.Extracciones import extraer_caracteristicas
from Extraccion.Extracciones import guardar_caracteristicas_csv
from Extraccion.Extracciones import graficar_comparacion


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


def aplicar_procesamiento(imagenes):
    """
    Aplica distintos métodos de procesamiento a cada imagen de la lista.

    :param imagenes: list, lista de imágenes en formato numpy.ndarray.
    :return: list, lista de listas con imágenes procesadas y sus etiquetas.
    """

    resultados = []  # Lista que almacenará las imágenes procesadas con sus etiquetas
    
    for imagen in imagenes:  # Itera sobre cada imagen en la lista
        procesadas = []  # Lista de transformaciones aplicadas a la imagen

        # Aplicar diferentes transformaciones a la imagen
        procesadas.append(("ChangeSize", ChangeSize(imagen, 500, 500)))  # Redimensionado a 500x500
        procesadas.append(("InterpolationLinear", InterpolationLinear(imagen, 500, 500)))  # Interpolación lineal
        procesadas.append(("InterpolationNearest", InterpolationNearest(imagen, 500, 500)))  # Interpolación por vecino más cercano
        procesadas.append(("And_Op", And_Op(imagen, imagenes[0], 127)))  # Operación lógica AND con la primera imagen
        procesadas.append(("Or_Op", Or_Op(imagen, imagenes[0], 127)))  # Operación lógica OR con la primera imagen
        procesadas.append(("Not_Op", Not_Op(imagen, 127)))  # Operación lógica NOT
        procesadas.append(("Sum_Op", Sum_Op(imagen, 50)))  # Incrementa la intensidad en 50
        procesadas.append(("Diff_Op", Diff_Op(imagen, 50)))  # Reduce la intensidad en 50
        procesadas.append(("Mult_Op", Mult_Op(imagen, 1.5)))  # Multiplica la intensidad por 1.5
        procesadas.append(("Div_Op", Div_Op(imagen, 2)))  # Divide la intensidad por 2
        procesadas.append(("ToBinary", ToBinary(imagen, 127)))  # Conversión a binario usando un umbral de 127
        procesadas.append(("ToGrayScale", ToGrayScale(imagen)))  # Conversión a escala de grises
        procesadas.append(("RotateImage", RotateImage(imagen)))  # Rotación de la imagen
        procesadas.append(("SharpenImage", SharpenImage(imagen)))  # Aplicación de un filtro de enfoque (sharpening)
        procesadas.append(("RelieveImage", RelieveImage(imagen)))  # Aplicación de un filtro de relieve
        procesadas.append(("BlurFilter", BlurFilter(imagen)))  # Aplicación de un filtro de desenfoque
        procesadas.append(("CannyEdgeDetection", CannyEdgeDetection(imagen)))  # Detección de bordes con Canny
        procesadas.append(("GaussianBlur", GaussianBlur(imagen)))  # Aplicación de un desenfoque gaussiano
        
        resultados.append(procesadas)  # Agrega la lista de imágenes procesadas a los resultados
    
    return resultados  # Retorna la lista con todas las transformaciones aplicadas


def imprimir_resultados(resultados, num_columnas=6):
    """
    Muestra cada imagen original junto con sus versiones procesadas en una cuadrícula.

    :param resultados: list, lista de listas de tuplas (etiqueta, imagen).
    :param num_columnas: int, número de columnas en la cuadrícula de visualización.
    """
    for i, procesadas in enumerate(resultados):  # Itera sobre los resultados
        num_imagenes = len(procesadas)  # Número total de imágenes procesadas
        num_filas = (num_imagenes + num_columnas - 1) // num_columnas  # Calcula la cantidad de filas necesarias

        # Crea una figura con subgráficos organizados en filas y columnas
        fig, axes = plt.subplots(num_filas, num_columnas, figsize=(15, num_filas * 3))
        axes = axes.flatten()  # Convierte la matriz de ejes en una lista

        for index, (etiqueta, imagen) in enumerate(procesadas):  # Itera sobre las imágenes procesadas
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB para visualización
            axes[index].imshow(imagen_rgb)  # Muestra la imagen en la posición correspondiente
            axes[index].set_title(etiqueta, fontsize=10)  # Coloca un título con la etiqueta
            axes[index].axis("off")  # Oculta los ejes

        # Deshabilita los ejes sobrantes si hay más subplots que imágenes
        for j in range(index + 1, len(axes)):
            axes[j].axis("off")

        # Título general de la figura
        fig.suptitle(f"Variaciones de Imagen {i + 1}", fontsize=14, fontweight="bold")
        plt.tight_layout()  # Ajusta el espaciado de la figura
        plt.show()  # Muestra la figura en pantalla


def main():
    """
    Función principal del script. Carga imágenes, aplica procesamiento, muestra resultados
    y guarda características en un archivo CSV.
    """
    # Carga las imágenes desde la carpeta especificada
    imagenes = obtener_imagenes_de_carpeta("Vision Artificial/Laboratorio 1/imagenes")
    
    # Aplica distintos métodos de procesamiento a las imágenes
    resultados = aplicar_procesamiento(imagenes)

    # Muestra los resultados procesados en una cuadrícula
    imprimir_resultados(resultados)

    # Extrae características de las imágenes procesadas y almacena los datos en una lista
    caracteristicas_lista = []
    for i, procesadas in enumerate(resultados):
        for etiqueta, img in procesadas:
            caracteristicas_lista.append(extraer_caracteristicas(i+1, img, etiqueta))  # Extrae características

    # Guarda las características extraídas en un archivo CSV
    guardar_caracteristicas_csv(caracteristicas_lista)

    # Genera gráficos comparativos de las características extraídas
    graficar_comparacion()

if __name__ == "__main__":
    main()