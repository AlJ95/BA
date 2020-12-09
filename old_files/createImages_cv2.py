import warnings
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from skimage.util import random_noise
import cv2

__RGBs__ = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
            [0.5, 1, 0, 1], [1, 0.5, 0, 1], [1, 0, 0.5, 1],
            [0.5, 0, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]]


def create_figs(image, image_pre_result, no_of_objects=1):
    for i in range(no_of_objects):
        width = rand.randint(0, image.shape[0]/2)
        height = rand.randint(0, image.shape[1]/2)
        xpos = rand.randint(0, image.shape[1])
        ypos = rand.randint(0, image.shape[0])
        group_number = i + 1
        rgb = np.array([rand.random(), rand.random(), rand.random(), 1])

        randInt = rand.randint(0, 1)
        if randInt == 0:
            # print(image, (xpos, ypos), (width, height))
            cv2.ellipse(image, (xpos, ypos), (width, height), 0, 0, 360, rgb, -1)
            cv2.ellipse(image_pre_result, (xpos, ypos), (width, height), 0, 0, 360,
                        (group_number + 1) * 25, -1)
        else:
            cv2.rectangle(image, (xpos, ypos), (xpos + width, ypos + height), rgb, -1)
            cv2.rectangle(image_pre_result, (xpos, ypos), (xpos + width, ypos + height),
                          (group_number + 1) * 25, -1)

    return image, image_pre_result


def produce_image(plot, no_of_images, salt_and_pepper_prob=0.05, dim=(1080, 1920, 4), grad="Laplacian",
                  kernel_size=3):
    if kernel_size != 3 and grad == "Scharr":
        warnings.warn('Scharr-Filter is using kernel size of 3, not' + str(kernel_size) + ".")

    image = np.ones(dim)
    image_pre_result = np.full(image.shape[0:2], 1)

    image, image_pre_result = create_figs(image, image_pre_result, no_of_images)

    if plot:
        plt.imshow(image)
        plt.show()

        plt.imshow(image_pre_result)
        plt.show()
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

    gradientRGB = np.amax(gradient, axis=2)

    if plot:
        fig, ax = plt.subplots(3, 3)
        ax[0, 0].imshow(image)
        ax[0, 1].imshow(image_pre_result)
        ax[0, 2].axis('off')
        ax[1, 0].imshow(gradient.astype('int64')[:, :, 0], cmap='inferno')
        ax[1, 1].imshow(gradient.astype('int64')[:, :, 1], cmap='inferno')
        ax[1, 2].imshow(gradient.astype('int64')[:, :, 2], cmap='inferno')
        ax[2, 0].axis('off')
        ax[2, 1].imshow(gradientRGB, cmap='inferno')
        ax[2, 2].axis('off')
        plt.show()

    if salt_and_pepper_prob > rand.random():
        #print('salt and pepper applied')
        image = random_noise(image, mode='s&p', amount=0.05)
        gradient = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize=11)[:, :, 0:3]
        gradient = np.abs(gradient)
        gradient /= np.max(gradient)

    gradientRGB = np.where(abs(gradientRGB) < 0.1, 0, 1)
    return image, gradient, gradientRGB

# @Todo Add Color Gradients
