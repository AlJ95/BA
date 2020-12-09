import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

input = torch.zeros((20, 20))
input[6:14, 6:14] = torch.tensor([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]
])
input = input.unsqueeze(0).unsqueeze(0).float()

LPP = nn.LPPool2d(2, 2, 2)
MAX = nn.MaxPool2d(2, 2)
MAXRET = nn.MaxPool2d(2, 2, return_indices=True)
AVG = nn.AvgPool2d(2, 2)

UNPOOL = nn.MaxUnpool2d(2, 2)

fig, ax = plt.subplots(3, 2)
ax[0, 0].imshow(input.squeeze().numpy())
ax[0, 1].imshow(nn.functional.interpolate(LPP(input), [20, 20], mode="bicubic").squeeze(0).squeeze(0).numpy(),
                interpolation="nearest", cmap="Greys")
ax[1, 1].imshow(nn.functional.interpolate(AVG(input), [20, 20], mode="bicubic").squeeze(0).squeeze(0).numpy(),
                interpolation="nearest", cmap="Greys")
ax[1, 0].imshow(nn.functional.interpolate(MAX(input), [20, 20], mode="bicubic").squeeze(0).squeeze(0).numpy(),
                interpolation="nearest", cmap="Greys")
ax[2, 0].imshow(UNPOOL(MAXRET(input)[0], MAXRET(input)[1]).squeeze(0).squeeze(0).numpy(),
                interpolation="nearest", cmap="Greys")
ax[0, 1].axis('off')
ax[0, 0].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
ax[2, 0].axis('off')
ax[2, 1].axis('off')
plt.show()

#

LPP1 = LPP(LPP(input))
LPP2 = nn.functional.interpolate(
    nn.functional.interpolate(LPP1, [10, 10], mode="bicubic"), [20, 20], mode="bicubic") \
    .squeeze(0).squeeze(0).numpy()

AVG1 = AVG(AVG(input))
AVG2 = nn.functional.interpolate(
    nn.functional.interpolate(AVG1, [10, 10], mode="bicubic"), [20, 20], mode="bicubic") \
    .squeeze(0).squeeze(0).numpy()

MAX1 = MAX(MAX(input))
MAX2 = nn.functional.interpolate(
    nn.functional.interpolate(MAX1, [10, 10], mode="bicubic"), [20, 20], mode="bicubic") \
    .squeeze(0).squeeze(0).numpy()

MAXRET1, i1 = MAXRET(input)
MAXRET2, i2 = MAXRET(MAXRET1)
MAXRET3 = UNPOOL(MAXRET2, i2)
MAXRET2 = UNPOOL(MAXRET3, i1).squeeze(0).squeeze(0).numpy()

fig1, ax = plt.subplots(3, 2)
ax[0, 0].imshow(input.squeeze().numpy())
ax[0, 1].imshow(LPP2, interpolation="nearest", cmap="Greys")
ax[1, 1].imshow(AVG2, interpolation="nearest", cmap="Greys")
ax[1, 0].imshow(MAX2, interpolation="nearest", cmap="Greys")
ax[2, 0].imshow(MAXRET2, interpolation="nearest", cmap="Greys")
ax[0, 1].axis('off')
ax[0, 0].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
ax[2, 0].axis('off')
ax[2, 1].axis('off')
plt.show()
