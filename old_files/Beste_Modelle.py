import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import Modell_Variation as model

kernel_size = [1, 3, 5, 7, 9, 11]
model1 = [[nn.MaxPool2d, nn.MaxUnpool2d, nn.MSELoss, x, torch.sigmoid, None, 100, -1] for x in kernel_size]
model1_strings = ["_".join(["MaxPool", "MaxUnpool", "MSEloss", "kernel" + str(x), "sigmoidActF",
                            "NetMax", "loss", "BestApproach"])
                  for x in kernel_size]

models = {x: y for x in model1_strings for y in model1}

t0 = time.time()
outs = [x for x in range(models.__len__())]
for j in range(10):
    for i in range(models.__len__()):
        # for i in [20]:
        netMax, loss_tracker, out = model.train(duration=60 * 60 * 2,
                                                epochs=40,
                                                batch_size=10,
                                                pooling=models[model1_strings[i]][0],
                                                unpooling=models[model1_strings[i]][1],
                                                loss=models[model1_strings[i]][2],
                                                edge_finder="Laplace",
                                                resize=False,
                                                # size=models[model1_strings[i]][4],
                                                size=(100, 100, 3),
                                                kernel_size=models[model1_strings[i]][3],
                                                plot=False,
                                                save=True,
                                                load=False,
                                                title=model1_strings[i] + "_" + str(j),
                                                directory="Models/GoodResults",
                                                activation_function1=model.identity,
                                                activation_function2=model.identity,
                                                activation_function3=models[model1_strings[i]][4],
                                                nn_input="Original"
                                                )
        if np.mean(loss_tracker[-5:]) < models[model1_strings[i]][6]:
            print("Der letzte 5 Bilder-Fehlerwert " + str(models[model1_strings[i]][6]) +
                  " " + model1_strings[i] + " wurde auf " + str(np.mean(loss_tracker[-5:])) + " verringert.")
            models[model1_strings[i]][5] = netMax
            models[model1_strings[i]][6] = np.mean(loss_tracker[-5:])
            models[model1_strings[i]][7] = j
            outs[i] = out

for x in range(outs.__len__()):
    hist = np.histogram(
        np.array(outs[x]), bins=[i / 10 for i in range(11)]
    )
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(outs[x].transpose(1, 0), interpolation="nearest", cmap="inferno")
    ax[0].imshow(np.where(outs[x] > 0.75, 1, 0).transpose(1, 0), interpolation="nearest", cmap="inferno")
    ax[1].bar(hist[1][1:], hist[0], 0.1)
    plt.title(x)
    plt.show()
    plt.close('all')

sums = [np.sum(x) for x in outs]
