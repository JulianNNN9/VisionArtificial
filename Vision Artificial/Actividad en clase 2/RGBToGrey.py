import cv2
import matplotlib.pyplot as plt

# Cargar una imagen en color
img = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg')

# Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mostrar la imagen resultante
plt.imshow(gray_img)
plt.axis('off')  # Ocultar los ejes
plt.show()