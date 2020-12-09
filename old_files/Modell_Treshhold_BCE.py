import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import Modell_Variation as model


def identity(x):
    return x


kernel_size = [1, 3, 5, 7, 9, 11]
model1 = [[nn.MaxPool2d, nn.MaxUnpool2d, nn.BCEWithLogitsLoss, x, model.identity, F.relu, torch.threshold, None, 100, -1]
          for x in kernel_size]
model1_strings = ["_".join(["MaxPool", "MaxUnpool", "BCE", "kernel" + str(x), "a1", "a2", "a3"
                            "NetMax", "loss", "BestApproach"])
                  for x in kernel_size]

models = {x: y for x in model1_strings for y in model1}

t0 = time.time()
outs = [x for x in range(models.__len__())]
for j in range(1):
    for i in range(models.__len__()):
        # for i in [20]:
        netMax, loss_tracker, out = model.train(duration=60 * 60 * 2,
                                                epochs=20,
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
                                                directory="Models/Tests/BCE_Threshold",
                                                activation_function1=models[model1_strings[i]][4],
                                                activation_function2=models[model1_strings[i]][5],
                                                activation_function3=models[model1_strings[i]][6],
                                                nn_input="Original",
                                                start_lr=0.1,
                                                end_lr=10**-15
                                                )
        if np.mean(loss_tracker[-5:]) < models[model1_strings[i]][8]:
            print("Der letzte 5 Bilder-Fehlerwert " + str(models[model1_strings[i]][8]) +
                  " " + model1_strings[i] + " wurde auf " + str(np.mean(loss_tracker[-5:])) + " verringert.")
            models[model1_strings[i]][7] = netMax
            models[model1_strings[i]][8] = np.mean(loss_tracker[-5:])
            models[model1_strings[i]][9] = j
            outs[i] = out

