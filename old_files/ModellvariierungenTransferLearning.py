import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import ModelVariation_TranserLearning as model
from custom_loss import weighted_mse_loss

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

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
activation_function2 = [F.relu, F.elu, F.hardtanh, F.softplus, F.softsign, F.tanhshrink, torch.sigmoid]
activation_function3 = [F.hardtanh, torch.threshold]
act1 = ["relu", "elu", "softplus", "softsign"]
act2 = ["relu", "elu", "hardtanh", "softplus", "softsign", "tanhshrink", "sig"]
act3 = ["hardtanh", "threshold"]

models = [[pooling[x], unpooling[x], y, z, w, v1, v2, v3]
          for x in range(pooling.__len__())
          for y in lossf
          for z in kernel_size
          for w in size
          for v1 in activation_function1
          for v2 in activation_function2
          for v3 in activation_function3
          for j in range(5)
          ]
model_strings = ["".join([p[x], u[x], y, str(z), w, v1, v2, v3,str(j+1)])
                 for x in range(pooling.__len__())
                 for y in lo
                 for z in kernel_size
                 for w in ["100"]
                 for v1 in act1
                 for v2 in act2
                 for v3 in act3
                 for j in range(5)
                 ]


import os
net_paths = pd.DataFrame([], columns=["name", "loss", "path"])
for x in os.listdir("Models/AlleVariationen2")[2:]:
    path = "Models/SavedNets/" + x
    mean = np.mean(np.load("Models/AlleVariationen2/" + x + "/loss.npy"))
    net_paths = net_paths.append(
        pd.DataFrame(
            [[x, mean, path+ ".pt"]], columns=["name", "loss", "path"]
        )
    )
net_paths.sort_values("loss").to_csv('Models/SavedNets/netPaths.csv')
net_paths = net_paths.sort_values('loss')
net_paths = net_paths.assign(lossfunction=net_paths['name'].str.slice(6, 9))
net_paths = net_paths.assign(ltype="Normal")

bce = net_paths.query('lossfunction == "BCE"')
mse = net_paths.query('lossfunction == "MSE"')
cst = net_paths.query('lossfunction == "wei"')


t0 = time.time()
loss_tracker = []
outs = []
models_results = []
models_by_loss = [bce, mse, cst]
for model_by_loss in models_by_loss:
    for j in range(3):
        parameter = []
        if model_by_loss.iloc[0,3] == "wei":
            i = 8
        else:
            i = 0
        parameter.append(lossf[lo.index(model_by_loss.iloc[j, 0][6:(9 + i)])])
        parameter.append(int(model_by_loss.iloc[j, 0][(9 + i)]))
        parameter.append(models[model_strings.index([x for x in model_strings if x == model_by_loss.iloc[j, 0]][0])][5])
        parameter.append(models[model_strings.index([x for x in model_strings if x == model_by_loss.iloc[j, 0]][0])][6])
        parameter.append(models[model_strings.index([x for x in model_strings if x == model_by_loss.iloc[j, 0]][0])][7])
        #
        # a = plt.imread("Models/AlleVariationen2/figure_from_last_epoch/" +
        #                net_paths.iloc[j, 0] + ".png")
        # plt.imshow(a)
        # plt.show()

        netMax, loss, out = model.train(duration=60 * 60 * 20,
                                        epochs=25,
                                        batch_size=50,
                                        netMax_load=model_by_loss.iloc[j, 2],
                                        pooling=nn.MaxPool2d,
                                        unpooling=nn.MaxUnpool2d,
                                        loss=parameter[0],
                                        edge_finder="Laplace",
                                        resize=False,
                                        # size=models[i][4],
                                        size=size[0],
                                        kernel_size=parameter[1],
                                        plot=False,
                                        save=True,
                                        load=False,
                                        title= model_by_loss.iloc[j, 0],
                                        directory="Models/AlleVariationen2TL",
                                        activation_function1=parameter[2],
                                        activation_function2=parameter[3],
                                        activation_function3=parameter[4],
                                        nn_input="Original",
                                        start_lr=0.00001,
                                        end_lr=10**-15,
                                        loss_tracker=np.load("Models/AlleVariationen2/" + model_by_loss.iloc[j, 0] +
                                                               "/loss.npy")
                                        )
        models_results.append([netMax, loss])
        net_paths.append(
            pd.DataFrame(
                [[model_by_loss.iloc[j, 0] , np.mean(loss[-10:]),
                  "Models/AlleVariationen2TL/SavedNets/" + model_by_loss.iloc[j, 0] + ".pt", parameter[0], "TransferL"]],
                columns=['name', 'loss', 'path', 'lossfunction', 'ltype']
            )
        )
        torch.save(netMax.state_dict(), "Models/AlleVariationen2TL/SavedNets/" + model_strings[i] + ".pt")
        t_diff = time.time() - t0
        if i % 10 == 0:
            print("Estimated Finish:" + time.ctime((models.__len__() / (i + 1)) * t_diff * 3 + t0))
        print(str(models[i][2]) + ": " + str(np.mean(loss[-10:])))
