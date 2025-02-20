import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg')

# Cambiar el tamaño de la imagen (Reducción y Amplificación)
resized_img = cv2.resize(img, (400, 400))  # Cambiar el tamaño a 400x400 píxeles

# Mostrar la imagen redimensionada
plt.imshow(resized_img)
plt.axis('off')  # Ocultar los ejes
plt.show()