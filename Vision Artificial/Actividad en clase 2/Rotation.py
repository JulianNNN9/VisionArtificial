import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg')


# Obtener las dimensiones de la imagen
(h, w) = img.shape[:2]

# Establecer el centro de la imagen para rotarla
center = (w // 2, h // 2)

# Crear la matriz de rotación (por ejemplo, 45 grados)
M = cv2.getRotationMatrix2D(center, 90, 1.0)

# Rotar la imagen
rotated_img = cv2.warpAffine(img, M, (w, h))

# Mostrar la imagen rotada
plt.imshow(rotated_img)
plt.axis('off')  # Ocultar los ejes
plt.show()