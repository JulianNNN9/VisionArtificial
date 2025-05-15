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
import csv
from datetime import datetime
import tempfile
import cv2
import numpy as np
import csv
import os
from datetime import datetime


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


def guardar_resultados_en_csv(resultados, nombre_funcion):
    """
    Guarda los resultados en un archivo CSV.
    
    :param resultados: Lista de resultados a guardar (valores numéricos).
    :param nombre_funcion: Nombre de la función utilizada, para el nombre del archivo CSV.
    """
    # Crear la carpeta de salida si no existe
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    
    # Nombre del archivo CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'resultados/{nombre_funcion}_{timestamp}.csv'
    
    # Guardar los resultados en el archivo CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir los encabezados
        writer.writerow(["Imagen", "Resultado"])
        for i, resultado in enumerate(resultados):
            writer.writerow([f"Imagen_{i+1}", resultado])
    print(f"Guardado {nombre_funcion} en {csv_filename}")


def procesar_imagenes(imagenes):
    """
    Aplica todas las funciones a una lista de imágenes y guarda los resultados en archivos CSV.
    
    :param imagenes: Lista de imágenes (rutas de imágenes) a procesar.
    """
    # Resultados de cada función
    resultados_hog = []
    resultados_momentos_hu = []
    resultados_estadisticos = []
    resultados_laplaciano = []
    resultados_circulos_hough = []
    resultados_grabcut = []
    resultados_orb = []
    resultados_gray = []
    resultados_umbral_adaptativo = []
    resultados_kmeans = []
    resultados_bordes_canny = []
    resultados_sharpen = []
    resultados_histograma_color = []
    resultados_tonalidades = []

     # Procesar cada imagen
    for img_path in imagenes:
        # Leer la imagen
        image = cv2.imread(img_path)
        
        # 1. Extraer características HOG
        hog_result = extraer_caracteristicas_hog(image)
        resultados_hog.append(hog_result)
        
        # 2. Métodos estadísticos primer orden
        estadisticos_result = metodos_estadisticos_primer_orden(image)
        resultados_estadisticos.append(estadisticos_result)
        
        # 3. Momentos de Hu
        hu_moments_result = momentos_de_hu(image)
        resultados_momentos_hu.append(hu_moments_result)
        
        # 4. Laplaciano de Gauss
        laplaciano_result = laplaciano_de_gauss(image)
        resultados_laplaciano.append(laplaciano_result)
        
        # 5. Detectar círculos Hough
        circulos_result = detectar_circulos_Hough(image)
        resultados_circulos_hough.append(circulos_result)
        
        # 6. Segmentar con GrabCut
        grabcut_result = segmentar_grabcut(image)
        resultados_grabcut.append(grabcut_result)
        
        # 7. Extraer ORB
        orb_result = extraer_orb(image)
        resultados_orb.append(orb_result)
        
        # 8. Convertir a escala de grises
        gray_result = ToGrayScale(image)
        resultados_gray.append(gray_result)
        
        # 9. Umbralización adaptativa
        umbral_result = umbralizacion_adaptativa(image)
        resultados_umbral_adaptativo.append(umbral_result)
        
        # 10. Segmentación K-means
        kmeans_result = segmentacion_kmeans(image)
        resultados_kmeans.append(kmeans_result)
        
        # 11. Detectar bordes Canny
        canny_result = detectar_bordes_Canny(image)
        resultados_bordes_canny.append(canny_result)
        
        # 12. Filtro de agudización
        sharpen_result = SharpenImage(image)
        resultados_sharpen.append(sharpen_result)
        
        # 13. Calcular histograma de color
        histograma_result = calcular_histograma_color(image)
        resultados_histograma_color.append(histograma_result)
        
        # 14. Detectar tonalidad
        tonalidad_result = detectar_tonalidades(image)
        resultados_tonalidades.append(tonalidad_result)
    
    # Guardar los resultados en archivos CSV
    guardar_resultados_en_csv(resultados_hog, "HOG")
    guardar_resultados_en_csv(resultados_momentos_hu, "Momentos_Hu")
    guardar_resultados_en_csv(resultados_estadisticos, "Estadisticos_Primer_Orden")
    guardar_resultados_en_csv(resultados_laplaciano, "Laplaciano_Gauss")
    guardar_resultados_en_csv(resultados_circulos_hough, "Circulos_Hough")
    guardar_resultados_en_csv(resultados_grabcut, "GrabCut")
    guardar_resultados_en_csv(resultados_orb, "ORB")
    guardar_resultados_en_csv(resultados_gray, "GrayScale")
    guardar_resultados_en_csv(resultados_umbral_adaptativo, "Umbralizacion_Adaptativa")
    guardar_resultados_en_csv(resultados_kmeans, "KMeans")
    guardar_resultados_en_csv(resultados_bordes_canny, "Bordes_Canny")
    guardar_resultados_en_csv(resultados_sharpen, "SharpenImage")
    guardar_resultados_en_csv(resultados_histograma_color, "Histograma_Color")
    guardar_resultados_en_csv(resultados_tonalidades, "Tonalidades")

# Cargar imágenes de prueba
imagenes = obtener_imagenes_de_carpeta("VisionArtificial\\Vision Artificial\\Parcial 2\\images")
procesar_imagenes(imagenes)
