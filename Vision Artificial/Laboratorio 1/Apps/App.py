import os
import cv2
import Funciones.Funciones as funcs


def obtener_imagenes_de_carpeta(ruta_carpeta):
    extensiones_imagenes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    imagenes = []

    # Verificar si la carpeta existe
    if not os.path.isdir(ruta_carpeta):
        print(f"La ruta '{ruta_carpeta}' no es una carpeta válida.")
        return []

    # Recorrer los archivos de la carpeta
    for archivo in os.listdir(ruta_carpeta):
        if os.path.splitext(archivo)[1].lower() in extensiones_imagenes:
            imagenes.append(os.path.join(ruta_carpeta, archivo))

    return imagenes


def aplicar_procesamiento(imagenes):
        
    nombres_funciones = [
        "ChangeSize",
        "InterpolationLinear",
        "InterpolationNearest",
        "And_Op",
        "Or_Op",
        "Not_Op",
        "Sum_Op",
        "Diff_Op",
        "Mult_Op",
        "Div_Op",
        "ToBinary",
        "ToGrayScale",
        "RotateImage",
        "SharpenImage",
        "RelieveImage",
        "BlurFilter",
        "CannyEdgeDetection",
        "GaussianBlur"
    ]

  
    resultados = []
    
    for imagen in imagenes:
        procesadas = []
        procesadas.append(("ChangeSize", funcs.ChangeSize(imagen, 500, 500)))
        procesadas.append(("InterpolationLinear", funcs.InterpolationLinear(imagen, 500, 500)))
        procesadas.append(("InterpolationNearest", funcs.InterpolationNearest(imagen, 500, 500)))
        
        procesadas.append(("And_Op", funcs.And_Op(imagen, imagenes[0], 127)))
        procesadas.append(("Or_Op", funcs.Or_Op(imagen, imagenes[0], 127)))
        procesadas.append(("Not_Op", funcs.Not_Op(imagen, 127)))
        procesadas.append(("Sum_Op", funcs.Sum_Op(imagen, 50)))
        procesadas.append(("Diff_Op", funcs.Diff_Op(imagen, 50)))
        procesadas.append(("Mult_Op", funcs.Mult_Op(imagen, 1.5)))
        procesadas.append(("Div_Op", funcs.Div_Op(imagen, 2)))
        procesadas.append(("ToBinary", funcs.ToBinary(imagen, 127)))
        procesadas.append(("ToGrayScale", funcs.ToGrayScale(imagen)))
        procesadas.append(("RotateImage", funcs.RotateImage(imagen)))
        procesadas.append(("SharpenImage", funcs.SharpenImage(imagen)))
        procesadas.append(("RelieveImage", funcs.RelieveImage(imagen)))
        procesadas.append(("BlurFilter", funcs.BlurFilter(imagen)))
        procesadas.append(("CannyEdgeDetection", funcs.CannyEdgeDetection(imagen)))
        procesadas.append(("GaussianBlur", funcs.GaussianBlur(imagen)))
        
        resultados.append(procesadas)
    
    return resultados


def mostrar_imagenes(resultados):
    for procesadas in resultados:
        columnas = []
        for nombre, imagen in procesadas:
            if imagen is not None:
                columnas.append(imagen)
        
        if columnas:
            imagenes_concatenadas = np.hstack(columnas)
            cv2.imshow("Imagenes Procesadas", imagenes_concatenadas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()







