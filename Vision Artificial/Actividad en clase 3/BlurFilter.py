import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('Actividad en clase 3/Imagenes/Orquideas.jpg')

# Aplicar el filtro de desenfoque
blurred_image = cv2.blur(image, (50, 50))

plt.title('Desenfoque')
plt.imshow(blurred_image)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen filtrada
cv2.imwrite('Actividad en clase 3/Imagenes/imagen_desenfocada.jpg', blurred_image)