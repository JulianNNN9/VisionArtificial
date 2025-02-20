import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('Actividad en clase 3/Imagenes/Orquideas.jpg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mostrar la imagen en escala de grises
# cv2.imshow('Escala de Grises', gray_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Guardar la imagen filtrada
cv2.imwrite('imagen_gris.jpg', gray_image)

plt.title('Escala de Grises')
plt.imshow(gray_image)
plt.axis('off')  # Ocultar los ejes
plt.show()