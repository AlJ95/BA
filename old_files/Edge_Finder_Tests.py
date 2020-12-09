import matplotlib.pyplot as plt
import createImages_cv2 as crImg
import cv2
import numpy as np

grads = ["Laplacian", "Scharr", "Sobel"]
kernel_sizes = [3, 5, 7, 9, 11, 13]

models = [[x, y] for x in grads for y in kernel_sizes if not (x == "Scharr" and y != 3)]


def edge_finder(image, grad, kernel_size):
    if grad == "Laplacian":
        gradient = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize=kernel_size)[:, :, 0:3].clip(0, 1)
    elif grad == "Scharr":
        dx = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=1, dy=0)[:, :, 0:3]
        dy = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=0, dy=1)[:, :, 0:3]
        gradient = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy)).clip(0, 1)
    else:
        dx = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)[:, :, 0:3]
        dy = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)[:, :, 0:3]
        gradient = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy)).clip(0, 1)
    return gradient


results = []
image, _, _ = crImg.produce_image(False, 3, 0, (100, 100, 3))
for model in models:
    results.append(edge_finder(image, model[0], model[1]))

fig, ax = plt.subplots(4, 4)
titles = np.array([str(x[0])+"-"+str(x[1]) for x in models])[:-1].reshape((4, 3))
for i in range(4):
    for j in range(3):
        ax[i, j].set_title(titles[i, j])
for i in range(4):
    for j in range(4):
        ax[i, j].axis('off')

# Laplacian
ax[0, 0].imshow(results[0], cmap="Reds")
ax[0, 1].imshow(results[1], cmap="Reds")
ax[0, 2].imshow(results[2], cmap="Reds")
ax[1, 0].imshow(results[4], cmap="Reds")
ax[1, 1].imshow(results[5], cmap="Reds")
ax[1, 2].imshow(results[6], cmap="Reds")
# Scharr
ax[2, 0].imshow(results[7], cmap="Reds")
# Sobel
ax[2, 1].imshow(results[8], cmap="Reds")
ax[2, 2].imshow(results[9], cmap="Reds")
ax[3, 0].imshow(results[10], cmap="Reds")
ax[3, 1].imshow(results[11], cmap="Reds")
ax[3, 2].imshow(results[12], cmap="Reds")
ax[3, 3].imshow(results[3], cmap="Reds")
ax[3, 3].set_title(str(models[-1][0])+"-"+str(models[-1][1]))

ax[1, 3].imshow(image)
ax[1, 3].set_title('Original')
plt.show()


