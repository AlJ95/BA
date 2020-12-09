import os

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL
import matplotlib.pyplot as plt


class ValidationData:
    __CITYSCAPES_MEAN__ = (0.28689554, 0.32513303, 0.28389177)
    __CITYSCAPES_STD__ = (0.18696375, 0.19017339, 0.18720214)
    __path__ = "D:/Dokumente/Studium/Bachelor/Bachelorarbeit/CityDataSet/train"
    __width__ = 256

    def __init__(self):
        self.image_paths = [os.path.join(ValidationData.__path__, x) for x in os.listdir(ValidationData.__path__)]
        self.img_used = 0

    def get_validation_data(self, images=20):
        data = [[plt.imread(self.image_paths[i])[:, :-ValidationData.__width__, :],
                 plt.imread(self.image_paths[i])[:, ValidationData.__width__:, :]]
                for i in np.arange(self.img_used, self.img_used + images)]
        self.img_used += images
        return data

    @staticmethod
    def plot_image(data: list):
        if type(data[0][0]) == torch.Tensor:
            data = [[data[i][0].squeeze(0).numpy().transpose(1, 2, 0),
                     data[i][1].squeeze(0).numpy()]
                    for i in range(data.__len__())]
        n = min(3, data.__len__())
        fig, ax = plt.subplots(n, 2)
        if n > 1:
            for i in range(n):
                ax[i, 0].imshow(data[i][0])
                ax[i, 1].imshow(data[i][1])
                ax[i, 0].axis('off')
                ax[i, 1].axis('off')
        else:
            ax[0].imshow(data[0][0])
            ax[1].imshow(data[0][1])
            ax[0].axis('off')
            ax[1].axis('off')

        plt.show()

    def forward(self, images=1, resize=True, dim_input=(100, 100, 3)):
        x, target = self.get_validation_data(images)[0]
        target = ValidationData.get_gradient(target)
        if resize:
            transform_input = transforms.Compose([
                transforms.Resize(dim_input[0:2]),
                transforms.ToTensor(),
                transforms.Normalize(ValidationData.__CITYSCAPES_MEAN__, ValidationData.__CITYSCAPES_STD__)
            ])
            transform_target = transforms.Compose([
                transforms.Resize(dim_input[0:2]),
                transforms.ToTensor(),
            ])
        else:
            transform_input = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(ValidationData.__CITYSCAPES_MEAN__, ValidationData.__CITYSCAPES_STD__)
            ])
            transform_target = transforms.Compose([
                transforms.ToTensor(),
            ])

        x = PIL.Image.fromarray(np.array(x / np.max(x) * 255, dtype='uint8'))
        target = PIL.Image.fromarray(np.array(target / np.max(x) * 255, dtype='uint8')).convert("L")
        x = transform_input(x)
        target = transform_target(target)

        return x, target

    def get_normalization(self):
        d = self.get_validation_data(1000)
        r, g, b = [], [], []
        d = [x[0] for x in d]
        z = [[np.mean(image.reshape((256 ** 2, 3)), axis=0)] for image in d]
        for x in z:
            r.append(x[0][0])
            g.append(x[0][1])
            b.append(x[0][2])

        return [np.mean(r) / 255, np.mean(g) / 255, np.mean(b) / 255]

    @staticmethod
    def get_gradient(image, grad="Scharr", kernel_size=3):

        if grad == "Laplacian":
            gradient = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize=kernel_size)[:, :, 0:3].clip(0, 1)
        elif grad == "Scharr":
            dx = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=1, dy=0)[:, :, 0:3]
            dy = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=0, dy=1)[:, :, 0:3]
            gradient = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy))
        else:
            dx = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)[:, :, 0:3]
            dy = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)[:, :, 0:3]
            gradient = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy))

        gradientRGB = np.amax(gradient, axis=2)
        return gradientRGB
