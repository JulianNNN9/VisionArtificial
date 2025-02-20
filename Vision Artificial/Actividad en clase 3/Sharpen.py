import cv2
import numpy as np
import matplotlib.pyplot as plt


# Cargar la imagen
image = cv2.imread('Actividad en clase 3/Imagenes/Orquideas.jpg')

# Crear un kernel para nítidez
sharpen_kernel = np.array([[-2, -2, -2],
                           [-2,  17, -2],
                           [-2, -2, -2]])

# Aplicar el filtro de nítidez el segundo parametro de profundidad de la imagen
# de salida ddepth
sharpened_image = cv2.filter2D(image, 0, sharpen_kernel)

# Mostrar la imagen filtrada
# cv2.imshow('Nitidez', sharpened_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.title('Nitidez')
plt.imshow(sharpened_image)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen filtrada
cv2.imwrite('Actividad en clase 3/Imagenes/imagen_nitida.jpg', sharpened_image)