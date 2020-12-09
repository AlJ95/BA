import createImages_cv2 as cr
import cv2
import matplotlib.pyplot as plt
import numpy as np


original, gradient, target = cr.produce_image(False, 15, salt_and_pepper_prob=0.00, dim=(100,100,3),
                                              grad="")
original = plt.imread("D:/Dokumente/Studium/Bachelor/Bachelorarbeit/CityDataset/train/1.jpg")
original = original[:, -256:, :]
#original = original[:, :256, :]

#kernel_size = 31
for kernel_size in [2*x + 3 for x in range(3)]:
    image = original
    Laplacian = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize=kernel_size)[:, :, 0:3].clip(0, 1)
    dx = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=1, dy=0)[:, :, 0:3]
    dy = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=0, dy=1)[:, :, 0:3]
    Scharr = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy))
    dx = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)[:, :, 0:3]
    dy = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)[:, :, 0:3]
    Sobel = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy))

    Laplacian = np.amax(Laplacian, axis=2)
    Scharr = np.amax(Scharr, axis=2)
    Sobel = np.amax(Sobel, axis=2)
    if kernel_size==3:
        fig, ax = plt.subplots(3, 4)
        ax[int((kernel_size-3)/2), 0].title.set_text('Original')
        ax[int((kernel_size-3)/2), 1].title.set_text('Laplacian')
        ax[int((kernel_size-3)/2), 2].title.set_text('Scharr')
        ax[int((kernel_size-3)/2), 3].title.set_text('Sobel')
        ax[int((kernel_size - 3) / 2), 2].imshow(Scharr, cmap="inferno")
        ax[int((kernel_size - 3) / 2), 2].title.set_text('Scharr3')
    ax[int((kernel_size - 3) / 2), 0].title.set_text('Original')
    ax[int((kernel_size - 3) / 2), 1].title.set_text('Laplacian' + str(kernel_size))
    ax[int((kernel_size - 3) / 2), 3].title.set_text('Sobel' + str(kernel_size))
    ax[int((kernel_size-3)/2), 0].imshow(image)
    ax[int((kernel_size-3)/2), 1].imshow(Laplacian, cmap="inferno")
    #ax[int((kernel_size-3)/2), 2].imshow(Scharr, cmap="inferno")
    ax[int((kernel_size-3)/2), 3].imshow(Sobel, cmap="inferno")
    ax[int((kernel_size-3)/2), 0].axis('off')
    ax[int((kernel_size-3)/2), 1].axis('off')
    ax[int((kernel_size-3)/2), 2].axis('off')
    ax[int((kernel_size-3)/2), 3].axis('off')
plt.show()