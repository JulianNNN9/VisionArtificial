import cv2
import matplotlib.pyplot as plt

# Cargar la imagen

img = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg')

# Cambiar tamaño usando diferentes métodos de interpolación
resized_bilinear = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)  # Bilineal
resized_nearest = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)  # Vecino más cercano

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2, 1, figsize=(15, 5))

# Escala de grises
axes[0].imshow(resized_bilinear, cmap='gray')
axes[0].set_title("Interpolación Bilineal")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(resized_nearest, cmap='gray')
axes[1].set_title("Interpolación Vecino más cercano")
axes[1].axis('off')

plt.show()
