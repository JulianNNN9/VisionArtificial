import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen binaria con agujeros
image = cv2.imread('/content/room-interior-design.jpg', 0)

# Crear un kernel de 5x5
kernel = np.ones((5, 5), np.uint8)

# Aplicar cierre para rellenar agujeros
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Mostrar resultados
# cv2.imshow('Imagen Original', image)
# cv2.imshow('Relleno de Agujeros con Cierre', closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.title('Imagen Original')
plt.imshow(image)
plt.axis('off')  # Ocultar los ejes
plt.show()

plt.title('Relleno de Agujeros con Cierre')
plt.imshow(closing)
plt.axis('off')  # Ocultar los ejes
plt.show()