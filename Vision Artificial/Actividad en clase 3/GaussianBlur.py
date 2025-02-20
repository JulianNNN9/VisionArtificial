import cv2
import matplotlib.pyplot as plt


# Cargar la imagen
image = cv2.imread('Actividad en clase 3/Imagenes/Orquideas.jpg')

# Aplicar el filtro de desenfoque gaussiano
gaussian_blurred_image = cv2.GaussianBlur(image, (51, 51), 0)

# Mostrar la imagen filtrada
# cv2.imshow('Desenfoque Gaussiano', gaussian_blurred_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.title('Desenfoque Gaussiano')
plt.imshow(gaussian_blurred_image)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen filtrada
cv2.imwrite('Actividad en clase 3/Imagenes/imagen_gaussian_blur.jpg', gaussian_blurred_image)