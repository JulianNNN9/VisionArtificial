import cv2
import matplotlib.pyplot as plt

# Cargar dos imágenes (deben tener el mismo tamaño)
img1 = cv2.imread('Actividad en clase 2/Imagenes/Vaca.jpg')
img2 = cv2.imread('Actividad en clase 2/Imagenes/Paisaje.jpg')

# Suma de imágenes
sum_img = cv2.add(img1, img2)

# Resta de imágenes
diff_img = cv2.subtract(img1, img2)

# Multiplicación de imágenes
mult_img = cv2.multiply(img1, img2)

# División de imágenes
div_img = cv2.divide(img1, img2)

# Mostrar las imágenes resultantes
fig, axes = plt.subplots(2,2 , figsize=(15, 5))

axes[0][0].imshow(sum_img, cmap='gray')
axes[0][0].set_title("Suma")
axes[0][0].axis('off')

axes[0][1].imshow(diff_img, cmap='gray')
axes[0][1].set_title("Resta")
axes[0][1].axis('off')

axes[1][0].imshow(mult_img, cmap='gray')
axes[1][0].set_title("Multiplicación")
axes[1][0].axis('off')

axes[1][1].imshow(div_img, cmap='gray')
axes[1][1].set_title("División")
axes[1][1].axis('off')

plt.show()