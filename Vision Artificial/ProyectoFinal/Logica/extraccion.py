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
from funciones import (
    extraer_caracteristicas_hog,
    metodos_estadisticos_primer_orden,
    momentos_de_hu,
    laplaciano_de_gauss,
    detectar_circulos_Hough,
    segmentar_grabcut,
    extraer_orb,
    ToGrayScale,
    umbralizacion_adaptativa,
    segmentacion_kmeans,
    detectar_bordes_Canny,
    SharpenImage,
    calcular_histograma_color,
    detectar_tonalidades
)


def obtener_imagenes_de_carpeta(ruta_carpeta):
    """
    Carga todas las imágenes de una carpeta y las devuelve como una lista de arrays de NumPy.

    :param ruta_carpeta: str, ruta de la carpeta donde están las imágenes.
    :return: list, lista de imágenes en formato numpy.ndarray.
    """
    extensiones_imagenes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    imagenes = []

    try:
        # Normalizar ruta
        ruta_carpeta = os.path.normpath(ruta_carpeta)
        print(f"Escaneando directorio: {ruta_carpeta}")

        if not os.path.isdir(ruta_carpeta):
            print(f"Error: '{ruta_carpeta}' no es un directorio válido")
            return []

        for archivo in os.listdir(ruta_carpeta):
            try:
                # Obtener ruta completa usando unión de rutas apropiada
                ruta_imagen = os.path.join(ruta_carpeta, archivo)
                
                # Verificar extensión del archivo
                if os.path.splitext(archivo)[1].lower() in extensiones_imagenes:
                    print(f"Intentando cargar: {archivo}")
                    img = cv2.imdecode(
                        np.fromfile(ruta_imagen, dtype=np.uint8), 
                        cv2.IMREAD_COLOR
                    )
                    
                    if img is not None:
                        print(f"Imagen cargada exitosamente: {archivo}")
                        imagenes.append(img)
                    else:
                        print(f"Error al cargar la imagen: {archivo}")
            
            except Exception as e:
                print(f"Error procesando {archivo}: {str(e)}")
                continue

        print(f"Total de imágenes cargadas: {len(imagenes)}")
        return imagenes

    except Exception as e:
        print(f"Error al escanear directorio: {str(e)}")
        return []


def guardar_resultados_en_csv(resultados, nombre_funcion):
    """
    Guarda los resultados en un archivo CSV.
    
    :param resultados: Lista de resultados a guardar (pueden ser números, arrays o diccionarios).
    :param nombre_funcion: Nombre de la función utilizada, para el nombre del archivo CSV.
    """
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'resultados/{nombre_funcion}_{timestamp}.csv'
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Verificar si hay resultados
        if not resultados or len(resultados) == 0:
            print(f"No hay resultados para guardar en {nombre_funcion}")
            return
            
        # Determinar el tipo de resultado y escribir los encabezados apropiados
        if isinstance(resultados[0], dict):
            # Para resultados tipo diccionario (como estadísticos)
            headers = ["Imagen"] + list(resultados[0].keys())
            writer.writerow(headers)
            for i, resultado in enumerate(resultados):
                row = [f"Imagen_{i+1}"] + list(resultado.values())
                writer.writerow(row)
        
        elif isinstance(resultados[0], np.ndarray):
            # Para resultados tipo array
            try:
                # Convertir a lista si es necesario
                first_result = resultados[0].tolist() if isinstance(resultados[0], np.ndarray) else resultados[0]
                
                # Crear encabezados basados en la longitud del primer resultado
                if isinstance(first_result, list):
                    headers = ["Imagen"] + [f"Valor_{i}" for i in range(len(first_result))]
                else:
                    headers = ["Imagen", "Valor"]
                
                writer.writerow(headers)
                
                # Escribir cada fila
                for i, resultado in enumerate(resultados):
                    valores = resultado.tolist() if isinstance(resultado, np.ndarray) else [resultado]
                    if isinstance(valores, list):
                        row = [f"Imagen_{i+1}"] + valores
                    else:
                        row = [f"Imagen_{i+1}", valores]
                    writer.writerow(row)
            
            except Exception as e:
                print(f"Error al procesar resultados para {nombre_funcion}: {str(e)}")
                return
        
        else:
            # Para resultados simples (números)
            writer.writerow(["Imagen", "Valor"])
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
    for image in imagenes:
        
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
# ...existing code...

# Modificar el manejo de rutas al final del archivo
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagenes = os.path.join(ruta_script, '..', 'img')
ruta_imagenes = os.path.normpath(ruta_imagenes)  # Normalizar separadores de ruta

print(f"Buscando imágenes en: {ruta_imagenes}")
imagenes = obtener_imagenes_de_carpeta(ruta_imagenes)

if not imagenes:
    print("No se cargaron imágenes. Verifique que la ruta sea correcta y que existan las imágenes.")
else:
    print(f"Se cargaron {len(imagenes)} imágenes exitosamente")
    procesar_imagenes(imagenes)
