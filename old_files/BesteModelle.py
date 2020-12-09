import time
import os
import pandas as pd
import Modell_Variation as model
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from custom_loss import weighted_mse_loss

pooling = [nn.MaxPool2d]  # , nn.MaxPool2d, nn.AvgPool2d, nn.LPPool2d]
p = ["Max"]  # , "Max", "Avg", "LPP"]
unpooling = [nn.MaxUnpool2d]  # , F.interpolate, F.interpolate, F.interpolate]
u = ["Max"]  # , "Int", "Int", "Int"]
edge_finder = ["Scharr", "Sobel", "Laplace"]
resize = [True, False]
size = [(100, 100, 3)]  # , (200, 200, 3), (500, 500, 3), (1920, 1080, 3)

lossf = [weighted_mse_loss,nn.BCEWithLogitsLoss, nn.MSELoss]  # "CUSTOM", "DICE"
lo = ["weightedMSE","BCE", "MSE"]
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



path = "C:\\Users\\janal\\OneDrive\\Bilder\\Saved Pictures"
a = os.listdir(path)[1:]

for x in range(a.__len__()):
    a[x] = a[x][:-4]
    a[x] = a[x].split(",")
best = pd.DataFrame(a)

best_models, best_models_string = [], []
for x in range(best.__len__()-3):
    best_models.append(models[model_strings.index(",".join(best.iloc[x+3,:]))])
    best_models_string.append(",".join(best.iloc[x+3,:]))

models_results = pd.DataFrame([], columns=["Model", "loss", "out", "NetMax", "trainings"])
t0 = time.time()
loss_tracker = []
outs = []
for i in range(best_models.__len__()):
    # for i in [20]:
    netMax, loss, out = model.train(duration=60 * 60 * 2,
                                    epochs=50,
                                    batch_size=20,
                                    pooling=best_models[i][0],
                                    unpooling=best_models[i][1],
                                    loss=best_models[i][2],
                                    edge_finder="Laplace",
                                    resize=False,
                                    # size=models[i][4],
                                    size=size[0],
                                    kernel_size=best_models[i][3],
                                    plot=False,
                                    save=True,
                                    load=False,
                                    title=best_models_string[i],
                                    directory="Models/AlleVariationen",
                                    activation_function1=best_models[i][5],
                                    activation_function2=best_models[i][6],
                                    activation_function3=best_models[i][7],
                                    nn_input="Original",
                                    start_lr=0.1,
                                    end_lr=10**-20
                                    )
    models_results = models_results.append(
        pd.DataFrame(
            [[model_strings[i], np.mean(loss[-10]), out, netMax, loss.__len__()]],
            columns=["Model", "loss", "out", "NetMax", "trainings"]
        )
    )
    t_diff = time.time() - t0
    print("Estimated Finish:" + time.ctime((models.__len__() / (i + 1)) * t_diff + t0))

    outs.append(out)

models_results.to_csv("Models_ManyLayer.csv")
