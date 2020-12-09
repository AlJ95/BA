import numpy as np
import matplotlib.pyplot as plt
import cv2

image_result = np.load("D:\\Dokumente\\Studium\\Bachelor\\Bachelorarbeit\\Bilder\\tmp.npy")
image = plt.imread("C:\\Users\\janal\\OneDrive\\Bilder\\temp\\tmp.png")

# Dreidimensionale positiv normierte Farbgradient (Rot, Grün, Blau)
gradient = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize=11)[:, :, 0:3]
gradient = np.abs(gradient)
gradient /= np.max(gradient)

# Maximum der drei Farbgradienten
gradientRGB = np.amax(gradient, axis=2)

# Manipulation des Numpy-Arrays, sodass die Ränder den Wert 1 haben und der Rest den Wert 0
filled = np.where(gradientRGB!=0, 1, 0) * 255
filled = filled.astype("uint8")
im_floodfill = filled.copy()

# Höhe und Breite des Numpy-Arrays
h, w = filled.shape[:2]

# Masken müssen eine extra Zeile/Spalte für die Ränder haben
mask = np.zeros((h + 2, w + 2), np.uint8)

# Ausfüllen der Matrix
cv2.floodFill(im_floodfill, mask, (0, 0), 255) # Hintergrund
cv2.floodFill(im_floodfill, mask, (500, 300), 125) # Rechteck
im_floodfill = im_floodfill.astype("float64") / 255

# Erstellen der Grafik
fix, ax = plt.subplots(2, 2)
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
ax[0, 0].imshow(image)
ax[0, 1].imshow(image_result)
ax[1, 0].imshow(gradient)
ax[1, 1].imshow(im_floodfill)
plt.show()
