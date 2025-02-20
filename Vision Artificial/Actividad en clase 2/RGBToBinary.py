import cv2
import matplotlib.pyplot as plt

# Cargar una imagen en color
img = cv2.imread('Actividad en clase 2/Imagenes/Papas.png')

# Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralizar la imagen para convertirla en blanco y negro (binaria)
_, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2, 1, figsize=(15, 5))

# Escala de grises
axes[0].imshow(gray_img, cmap='gray')
axes[0].set_title("Escala de Grises")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(binary_img, cmap='gray')
axes[1].set_title("Blanco y Negro")
axes[1].axis('off')

plt.show()