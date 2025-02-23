import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extraer_caracteristicas(numero_imagen, imagen, nombre):
    """Extrae características de la imagen y devuelve un diccionario con los valores."""
    
    # Convertir la imagen a escala de grises si es una imagen en color (BGR)
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen

    # Verificar si la imagen tiene al menos 100 filas y 200 columnas para extraer un píxel específico
    if imagen.shape[0] > 100 and imagen.shape[1] > 200:
        if len(imagen.shape) == 3:  # Si la imagen es en color (BGR)
            pixel_valor = tuple(imagen[100, 200])  # Extraer el valor del píxel en la posición (100, 200)
        else:  # Si la imagen es en escala de grises
            pixel_valor = (imagen[100, 200], None, None)  # Solo hay un canal, los otros valores quedan en None
    else:
        pixel_valor = (None, None, None)  # Si la imagen es muy pequeña, no se extrae el píxel

    # Extraer una submatriz 3x3 de la imagen en escala de grises si tiene al menos esas dimensiones
    matriz_reducida = img_gray[:3, :3] if img_gray.shape[0] >= 3 and img_gray.shape[1] >= 3 else None

    # Calcular el determinante de la matriz 3x3 si es válida
    determinante = np.linalg.det(matriz_reducida) if matriz_reducida is not None else None

    # Realizar operaciones matemáticas sobre la imagen en escala de grises
    multiplicacion = np.sum(img_gray * 2)  # Multiplicación de todos los píxeles por 2 y suma
    suma = np.sum(img_gray + 50)  # Suma de todos los píxeles con un incremento de 50 unidades

    # Crear un diccionario con las características extraídas
    caracteristicas = {
        "Numero_Imagen": numero_imagen,
        "Nombre": nombre,
        "Pixel_100_200_B": pixel_valor[0],  # Canal Azul
        "Pixel_100_200_G": pixel_valor[1],  # Canal Verde
        "Pixel_100_200_R": pixel_valor[2],  # Canal Rojo
        "Promedio_Intensidad": np.mean(img_gray),  # Promedio de intensidad en escala de grises
        "Desviacion_Estandar": np.std(img_gray),  # Desviación estándar de la intensidad
        "Determinante": determinante,  # Determinante de la matriz 3x3
        "Multiplicacion": multiplicacion,  # Resultado de la operación de multiplicación
        "Suma": suma  # Resultado de la operación de suma
    }
    return caracteristicas  # Retornar el diccionario con las características de la imagen

def guardar_caracteristicas_csv(caracteristicas_lista, archivo_salida="Vision Artificial/Laboratorio 1/Resultados/caracteristicas_imagenes.csv"):
    """Guarda una lista de características extraídas en un archivo CSV."""

    # Verificar que la entrada sea una lista de diccionarios antes de procesarla
    if not isinstance(caracteristicas_lista, list) or not all(isinstance(item, dict) for item in caracteristicas_lista):
        print("Error: La entrada debe ser una lista de diccionarios.")  # Mensaje de error si la entrada no es válida
        return
    
    # Crear un DataFrame de pandas a partir de la lista de características
    df = pd.DataFrame(caracteristicas_lista)

    # Guardar el DataFrame en un archivo CSV sin incluir el índice
    df.to_csv(archivo_salida, index=False)
    print(f"Características guardadas en {archivo_salida}")  # Mensaje de confirmación

def graficar_comparacion():
    """Genera gráficos de comparación entre dos imágenes usando métricas extraídas del archivo CSV."""

    # Ruta del archivo CSV que contiene las características de las imágenes
    data_path = "Vision Artificial/Laboratorio 1/Resultados/caracteristicas_imagenes.csv"

    # Leer los datos del CSV en un DataFrame de pandas
    df = pd.read_csv(data_path)

    # Filtrar los datos para obtener solo las filas correspondientes a cada imagen
    img1 = df[df["Numero_Imagen"] == 1]
    img2 = df[df["Numero_Imagen"] == 2]

    # Lista de métricas que se compararán gráficamente
    metricas = ["Promedio_Intensidad", "Desviacion_Estandar", "Determinante", "Multiplicacion", "Suma"]

    # Generar un gráfico para cada métrica
    for metrica in metricas:
        plt.figure(figsize=(12, 5))  # Configurar el tamaño del gráfico
        plt.plot(img1["Nombre"], img1[metrica], label="Imagen 1", marker="o")  # Graficar los valores de la Imagen 1
        plt.plot(img2["Nombre"], img2[metrica], label="Imagen 2", marker="s")  # Graficar los valores de la Imagen 2
        plt.xticks(rotation=90)  # Rotar las etiquetas en el eje X para mejor visualización
        plt.ylabel(metrica)  # Etiqueta del eje Y con el nombre de la métrica
        plt.title(f"Comparación de {metrica} entre Imagen 1 y Imagen 2")  # Título del gráfico
        plt.legend()  # Agregar leyenda para identificar cada imagen
        plt.grid(True)  # Agregar una cuadrícula para facilitar la lectura
        plt.show()  # Mostrar el gráfico en pantalla
