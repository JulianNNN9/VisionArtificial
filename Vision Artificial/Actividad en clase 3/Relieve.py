import cv2
import numpy as np
import matplotlib.pyplot as plt


# Cargar la imagen
image = cv2.imread('Actividad en clase 3/Imagenes/Orquideas.jpg')

# Crear un kernel para el filtro de relieve
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])

# Aplicar el filtro de relieve
embossed_image = cv2.filter2D(image, -1, emboss_kernel)

# Mostrar la imagen filtrada
# cv2.imshow('Relieve', embossed_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Guardar la imagen filtrada
cv2.imwrite('Actividad en clase 3/Imagenes/imagen_relieve.jpg', embossed_image)

plt.title('Relieve')
plt.imshow(embossed_image)
plt.axis('off')  # Ocultar los ejes
plt.show()