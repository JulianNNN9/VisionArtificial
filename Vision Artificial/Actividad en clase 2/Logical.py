import cv2
import matplotlib.pyplot as plt

# Cargar dos imágenes binarizadas (deben ser del mismo tamaño)
img1 = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Actividad en clase 2/Imagenes/Paisaje.jpg', cv2.IMREAD_GRAYSCALE)

# Convertir a binario (umbralizar si es necesario)
_, img1_bin = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
_, img2_bin = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

# Operación AND
and_img = cv2.bitwise_and(img1_bin, img2_bin)

# Operación OR
or_img = cv2.bitwise_or(img1_bin, img2_bin)

# Operación NOT
not_img1 = cv2.bitwise_not(img1_bin)
not_img2 = cv2.bitwise_not(img2_bin)

# Mostrar las imágenes resultantes

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2,2 , figsize=(15, 5))

axes[0][0].imshow(and_img, cmap='gray')
axes[0][0].set_title("AND")
axes[0][0].axis('off')

axes[0][1].imshow(or_img, cmap='gray')
axes[0][1].set_title("OR")
axes[0][1].axis('off')

axes[1][0].imshow(not_img1, cmap='gray')
axes[1][0].set_title("NOT IMAGEN UNO")
axes[1][0].axis('off')

axes[1][1].imshow(not_img2, cmap='gray')
axes[1][1].set_title("NOT IMAGEN DOS")
axes[1][1].axis('off')

plt.show()