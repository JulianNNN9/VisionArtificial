import cv2
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread('Actividad en clase 3/Imagenes/Orquideas.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar el filtro de detección de bordes (Canny)
edges = cv2.Canny(image, 100, 200)

# Mostrar la imagen filtrada
# cv2.imshow('Detección de Bordes (Canny)', edges)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.title('Detección de Bordes (Canny)')
plt.imshow(edges)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen filtrada
cv2.imwrite('Actividad en clase 3/Imagenes/imagen_bordes.jpg', edges)