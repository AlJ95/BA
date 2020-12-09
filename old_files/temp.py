import cv2
import numpy as np
import matplotlib.pyplot as plt
import createImages_cv2 as cr
import pandas as pd

means = pd.DataFrame([], columns=['i'])
for j in range(15):
    r, g, b = [], [], []
    for i in range(1000):
        image, _, _ = cr.produce_image(False, j + 1, 0, (100, 100, 3))
        v, w, z = np.mean(image.reshape((10000, 3)), axis=0)
        r.append(v)
        g.append(w)
        b.append(z)

    means = means.append(
        pd.DataFrame(
            [[j+1, np.mean(r), np.mean(g), np.mean(b), np.std(r), np.std(g), np.std(b)]], columns=['i', 'rm', 'gm', 'bm','rs', 'gs', 'bs']
        )
    )

means.to_csv()
#
# original, gradient, target = cr.produce_image(False, 15, salt_and_pepper_prob=0.00,
#                                               dim=(100, 100, 3), grad="Laplacian")
#
# # image = original
# # Lap = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize=kernel_size)[:, :, 0:3].clip(0, 1)
# # dx = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=1, dy=0)[:, :, 0:3]
# # dy = cv2.Scharr(image.astype('float64'), cv2.CV_64F, dx=0, dy=1)[:, :, 0:3]
# # Sch = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy)).clip(0, 1)
# # dx = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)[:, :, 0:3]
# # dy = cv2.Sobel(image.astype('float64'), cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)[:, :, 0:3]
# # Sob = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx), np.abs(dy)).clip(0, 1)
# #
# # gradientRGB = np.amax(gradient, axis=2)
#
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(original)
# ax[1].imshow(gradient, interpolation="nearest", cmap="inferno")
# ax[2].imshow(target, interpolation="nearest", cmap="Greys")
# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[0].title.set_text('Input')
# ax[1].title.set_text('Laplacian')
# ax[2].title.set_text('Target')
# #plt.show()
# plt.savefig('D:/Dokumente/Studium/Bachelor/Bachelorarbeit/Bilder/InputTarget.png')