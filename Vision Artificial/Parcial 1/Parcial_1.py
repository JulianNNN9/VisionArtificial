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
from Funciones.Funciones import umbralizar_imagen
from Funciones.Funciones import aplicar_apertura
from Funciones.Funciones import aplicar_cierre
from Funciones.Funciones import detectar_contornos
from Funciones.Funciones import dibujar_contornos
from Funciones.Funciones import calcular_caracteristicas
from Funciones.Funciones import aplicar_gradiente
from Funciones.Funciones import detectar_bordes_Canny
from Funciones.Funciones import segmentacion_kmeans
from Funciones.Funciones import segmentacion_watershed
from Funciones.Funciones import umbralizacion_adaptativa
