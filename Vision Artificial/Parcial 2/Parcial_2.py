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

#Descriptores de Textura
from funciones.funciones import extraer_caracteristicas_hog #SOLO TEXTO - RETORNA ARRAY
from funciones.funciones import momentos_de_hu #SOLO TEXTO - RETORNA TEXTO
from funciones.funciones import metodos_estadisticos_primer_orden #SOLO TEXTO - RETORNA DICCIONARIO CON TEXTO
from funciones.funciones import metodos_estadisticos_segundo_orden #SOLO TEXTO - RETORNA DICCIONARIO CON TEXTO

#Detección de Bordes
from funciones.funciones import laplaciano_de_gauss #GRAFICA - RETORNA IMAGEN 

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

def guardar_en_csv(imagenes):
    # Crear archivo CSV con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"resultados_textuales_{timestamp}.csv"
    
    # Encabezados del CSV basados en los datos textuales
    headers = [
        "imagen_num",
        # Descriptores de Textura
        "hog_vector_shape",
        "momentos_hu",
        "estadisticos_primer_orden",
        "estadisticos_segundo_orden",
        # Detección de Formas (solo datos)
        "lineas_hough_count",
        "circulos_hough_data",
        # Métodos Avanzados
        "sift_keypoints_count",
        "sift_descriptors_shape",
        "orb_keypoints_count",
        "orb_descriptors_shape",
        "kaze_keypoints_count",
        "kaze_descriptors_shape",
        "akaze_keypoints_count",
        "akaze_descriptors_shape"
    ]
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        
        for idx, img in enumerate(imagenes):
            # Convertir a escala de grises para métodos que lo requieren
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            row_data = {"imagen_num": idx}
            
            # 1. Descriptores de Textura --------------------------------------
            # HOG - Retorna array (solo guardamos forma)
            hog_features = extraer_caracteristicas_hog(img)
            row_data["hog_vector_shape"] = str(hog_features.shape)
            
            # Momentos de Hu - Retorna array de momentos
            hu_moments = momentos_de_hu(img_gray)
            row_data["momentos_hu"] = str([float(f"{m[0]:.4e}") for m in hu_moments])
            
            # Estadísticos primer orden - Diccionario
            stats_primer_orden = metodos_estadisticos_primer_orden(img_gray)
            row_data["estadisticos_primer_orden"] = str(stats_primer_orden)
            
            # Estadísticos segundo orden - Diccionario
            stats_segundo_orden = metodos_estadisticos_segundo_orden(img_gray)
            row_data["estadisticos_segundo_orden"] = str(stats_segundo_orden)
            
            # 2. Detección de Formas ------------------------------------------
            # Líneas Hough - Retorna (edges, lines) -> solo nos interesa lines
            _, lines = detectar_lineas_Hough(img_gray)
            row_data["lineas_hough_count"] = len(lines) if lines is not None else 0
            
            # Círculos Hough - Retorna (imagen, circles) -> solo circles
            _, circles = detectar_circulos_Hough(img_gray)
            if circles is not None:
                row_data["circulos_hough_data"] = str([(x, y, r) for (x, y, r) in circles])
            else:
                row_data["circulos_hough_data"] = "None"
            
            # 3. Métodos Avanzados --------------------------------------------
            # SIFT - Necesita ruta de archivo
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                cv2.imwrite(tmp.name, img)
                sift_kp, sift_desc = aplicar_sift_con_preprocesamiento(tmp.name)
                row_data["sift_keypoints_count"] = len(sift_kp) if sift_kp is not None else 0
                row_data["sift_descriptors_shape"] = str(sift_desc.shape) if sift_desc is not None else "None"
            
            # ORB - Diccionario con keypoints y descriptors
            orb_result = extraer_orb(img)
            row_data["orb_keypoints_count"] = len(orb_result["keypoints"])
            row_data["orb_descriptors_shape"] = str(orb_result["descriptors"].shape) if orb_result["descriptors"] is not None else "None"
            
            # KAZE - Diccionario con keypoints y descriptors
            kaze_result = extraer_kaze(img)
            row_data["kaze_keypoints_count"] = len(kaze_result["keypoints"])
            row_data["kaze_descriptors_shape"] = str(kaze_result["descriptors"].shape) if kaze_result["descriptors"] is not None else "None"
            
            # AKAZE - Diccionario con keypoints y descriptors
            akaze_result = extraer_akaze(img)
            row_data["akaze_keypoints_count"] = len(akaze_result["keypoints"])
            row_data["akaze_descriptors_shape"] = str(akaze_result["descriptors"].shape) if akaze_result["descriptors"] is not None else "None"
            
            # Escribir fila en CSV
            writer.writerow(row_data)
    
    print(f"Datos textuales guardados en {csv_filename}")

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
imagenes = obtener_imagenes_de_carpeta("Parcial 2/images")
procesar_imagenes(imagenes)
#guardar_en_csv(imagenes)