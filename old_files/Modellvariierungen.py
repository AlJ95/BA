import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Modell_Variation as model
from custom_loss import weighted_mse_loss

pooling = [nn.MaxPool2d]  # , nn.MaxPool2d, nn.AvgPool2d, nn.LPPool2d]
p = ["Max"]  # , "Max", "Avg", "LPP"]
unpooling = [nn.MaxUnpool2d]  # , F.interpolate, F.interpolate, F.interpolate]
u = ["Max"]  # , "Int", "Int", "Int"]
edge_finder = ["Scharr", "Sobel", "Laplace"]
resize = [False]
size = [(100, 100, 3)]  # , (200, 200, 3), (500, 500, 3), (1920, 1080, 3)

lossf = [weighted_mse_loss,nn.BCEWithLogitsLoss, nn.MSELoss]  # "CUSTOM", "DICE"
lo = ["weightedMSE","BCE", "MSE"]
kernel_size = [1, 3, 5]
activation_function1 = [F.relu, F.elu, F.softplus, F.softsign]
activation_function2 = [F.relu, F.elu, F.hardtanh, F.softplus, F.softsign, F.tanhshrink]
activation_function3 = [F.hardtanh]
act1 = ["relu", "elu", "softplus", "softsign"]
act2 = ["relu", "elu", "hardtanh", "softplus", "softsign", "tanhshrink"]
act3 = ["hardtanh"]

models = [[pooling[x], unpooling[x], y, z, w, v1, v2, v3]
          for x in range(pooling.__len__())
          for y in lossf
          for z in kernel_size
          for w in size
          for v1 in activation_function1
          for v2 in activation_function2
          for v3 in activation_function3
          ]
model_strings = ["".join([p[x], u[x], y, str(z), w, v1, v2, v3])
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
models_results = []
net_paths = pd.DataFrame([], columns=["name", "loss", "path"])
for j in range(1):
    for i in range(models.__len__()):
        # for i in [20]:
        netMax, loss, out = model.train(duration=60 * 60 * 2,
                                        epochs=5,
                                        batch_size=20,
                                        pooling=models[i][0],
                                        unpooling=models[i][1],
                                        loss=models[i][2],
                                        edge_finder="Laplace",
                                        resize=False,
                                        # size=models[i][4],
                                        size=size[0],
                                        kernel_size=models[i][3],
                                        plot=False,
                                        save=True,
                                        load=False,
                                        title=model_strings[i]+str(j+1),
                                        directory="Models/AlleVariationen3",
                                        activation_function1=models[i][5],
                                        activation_function2=models[i][6],
                                        activation_function3=models[i][7],
                                        nn_input="Original",
                                        start_lr=0.1,
                                        end_lr=0.00000000001
                                        )
        models_results.append([netMax, loss])
        net_paths.append(
            pd.DataFrame(
                [[model_strings[i]+str(j+1), loss, "Models/SavedNets/" + model_strings[i]+str(j+1) + ".pt"]]
            )
        )
        torch.save(netMax.state_dict(), "Models/SavedNets/" + model_strings[i]+str(j+1) + ".pt")
        t_diff = time.time() - t0
        if i % 10 == 0:
            print("Estimated Finish:" + time.ctime((models.__len__() / (i + 1)) * t_diff * 5 + t0))
        print(str(models[i][2]) + ": " + str(np.mean(loss[-10:])))


import os
for x in os.listdir("Models/AlleVariationen2")[2:]:
    path = "Models/SavedNets/" + x
    mean = np.mean(np.load("Models/AlleVariationen2/" + x + "/loss.npy"))
    net_paths = net_paths.append(
        pd.DataFrame(
            [[x, mean, path+ ".pt"]], columns=["name", "loss", "path"]
        )
    )
net_paths.sort_values("loss").to_csv('Models/SavedNets/netPaths.csv')
