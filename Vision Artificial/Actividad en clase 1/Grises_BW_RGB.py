import cv2
import matplotlib.pyplot as plt

# Leer la imagen en escala de grises
img_gray = cv2.imread('Actividad en clase 1/Imagenes/Teemo_18.jpg', cv2.IMREAD_GRAYSCALE)

# Leer la imagen en blanco y negro (binaria) usando un umbral
_, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

# Leer la imagen original en RGB (se asume que la imagen es RGB)
img_rgb = cv2.imread('Actividad en clase 1/Imagenes/Teemo_18.jpg')

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Escala de grises
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title("Escala de Grises")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(img_binary, cmap='gray')
axes[1].set_title("Blanco y Negro")
axes[1].axis('off')

# Imagen RGB
axes[2].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
axes[2].set_title("RGB")
axes[2].axis('off')

plt.show()