import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imágenes en escala de grises
image1 = cv2.imread('Filtrado Morfologico/Imagenes/Formas.png', 0)
image2 = cv2.imread('Filtrado Morfologico/Imagenes/Letras.png', 0)
image3 = cv2.imread('Filtrado Morfologico/Imagenes/MujerNegra.jpeg', 0)
image4 = cv2.imread('Filtrado Morfologico/Imagenes/RuidoLibros.jpeg', 0)

# Crear un kernel de 5x5
#kernel = np.ones((5, 5), np.uint8)

# Crear un kernel cruzado de 5x5
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
'''
array([[0, 1, 0],
       [1, 1, 1],
       [0, 1, 0], dtype=uint8)
'''

# Aplicar gradiente a cada imagen
processed_images = []
for img in [image1, image2, image3, image4]:
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_cross)
    processed_images.append((img, gradient))

# Mostrar resultados utilizando matplotlib
fig, axes = plt.subplots(4, 2, figsize=(10, 10))

titles = ['Imagen Original', 'Bordes Detectados con Gradiente']

for i, (img, gradient) in enumerate(processed_images):
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(f'{titles[0]} {i+1}')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(gradient, cmap='gray')
    axes[i, 1].set_title(f'{titles[1]} {i+1}')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()