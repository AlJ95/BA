import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Modell_Variation as model

pooling = [nn.MaxPool2d]  # , nn.MaxPool2d, nn.AvgPool2d, nn.LPPool2d]
p = ["Max"]  # , "Max", "Avg", "LPP"]
unpooling = [nn.MaxUnpool2d]  # , F.interpolate, F.interpolate, F.interpolate]
u = ["Max"]  # , "Int", "Int", "Int"]
edge_finder = ["Scharr", "Sobel", "Laplace"]
resize = [True, False]
size = [(100, 100, 3)]  # , (200, 200, 3), (500, 500, 3), (1920, 1080, 3)

lossf = [nn.BCEWithLogitsLoss, nn.MSELoss]  # "CUSTOM", "DICE"
lo = ["BCE", "MSE"]
kernel_size = [1, 3, 5]
activation_function1 = [F.relu, F.elu, F.softplus, F.softsign]
activation_function2 = [torch.sigmoid, F.relu, F.elu, F.hardtanh, F.softplus, F.softsign, F.tanhshrink]
activation_function3 = [torch.threshold, F.hardtanh]
act1 = ["relu", "elu", "softplus", "softsign"]
act2 = ["sig", "relu", "elu", "hardtanh", "softplus", "softsign", "tanhshrink"]
act3 = ["threshold", "hardtanh"]

models = [[pooling[x], unpooling[x], y, z, w, v1, v2, v3]
          for x in range(pooling.__len__())
          for y in lossf
          for z in kernel_size
          for w in size
          for v1 in activation_function1
          for v2 in activation_function2
          for v3 in activation_function3
          ]
model_strings = [",".join([p[x], u[x], y, str(z), w, v1, v2, v3])
                 for x in range(pooling.__len__())
                 for y in lo
                 for z in kernel_size
                 for w in ["100"]
                 for v1 in act1
                 for v2 in act2
                 for v3 in act3
                 ]

t0 = time.time()
loss_tracker = []
outs = []
models_results = pd.read_csv("Models.csv")
models_results = models_results.sort_values("trainings", 0, False)
for i in range(10):
    # for i in [20]:
    current_model = models[model_strings.index(models_results.iloc[i, 1])]
    netMax, loss, out = model.train(duration=60 * 60 * 2,
                                    epochs=5,
                                    batch_size=20,
                                    pooling=current_model[0],
                                    unpooling=current_model[1],
                                    loss=current_model[2],
                                    edge_finder="Laplace",
                                    resize=False,
                                    # size=models[i][4],
                                    size=size[0],
                                    kernel_size=current_model[3],
                                    plot=False,
                                    save=True,
                                    load=False,
                                    title=models_results.iloc[i, 1],
                                    directory="Models_InputOrig2",
                                    activation_function1=model.identity,
                                    activation_function2=model.identity,
                                    activation_function3=current_model[5],
                                    nn_input="Original"
                                    )
    t_diff = time.time() - t0
    print("Estimated Finish:" + time.ctime((models.__len__() / (i + 1)) * t_diff + t0))

    outs.append(out)


