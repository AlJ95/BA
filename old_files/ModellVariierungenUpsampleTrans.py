import os
import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Model_Upsample_with_TransposeConv as model
from custom_loss import weighted_mse_loss

pooling = [nn.AvgPool2d, nn.LPPool2d]  # , nn.MaxPool2d, nn.AvgPool2d, nn.LPPool2d]
p = ["AVG", "LPP"]  # , "Max", "Avg", "LPP"]
unpooling = [F.interpolate, F.interpolate]  # , F.interpolate, F.interpolate, F.interpolate]
u = ["Int", "Int"]  # , "Int", "Int", "Int"]
edge_finder = ["Scharr", "Sobel", "Laplace"]
resize = [False]
size = [(120, 120, 3)]  # , (200, 200, 3), (500, 500, 3), (1920, 1080, 3)

lossf = [nn.MSELoss]  # "CUSTOM", "DICE"
lo = ["MSE"]
kernel_size = [1, 3, 5]
activation_function1 = [F.elu]
activation_function2 = [F.softsign]
activation_function3 = [F.hardtanh]
act1 = ["elu"]
act2 = ["softsign"]
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
c = 0

#loss_tracker = []
#outs = []
models_results2 = []
net_paths2 = pd.DataFrame([], columns=["name", "loss", "path"])
for j in range(1):
    for i in range(models.__len__()):
        if not os.listdir("Models/TransposedConv/").__contains__(model_strings[i]+"1"):
        # for i in [20]:
            netMax, loss, out = model.train(duration=60 * 60 * 2,
                                            epochs=5,
                                            batch_size=20,
                                            pooling=models[i][0],
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
                                            directory="Models/TransposedConv",
                                            activation_function1=models[i][5],
                                            activation_function2=models[i][6],
                                            activation_function3=models[i][7],
                                            nn_input="Original",
                                            start_lr=0.1,
                                            end_lr=10**-15
                                            )
            models_results2.append([netMax, loss])
            net_paths2.append(
                pd.DataFrame(
                    [[model_strings[i]+str(j+1), loss, "Models/TransposedConv/SavedNets/" + model_strings[i]+str(j+1) + ".pt"]]
                )
            )
            torch.save(netMax.state_dict(), "Models/TransposedConv/SavedNets/" + model_strings[i]+str(j+1) + ".pt")
            t_diff = time.time() - t0
            if i % 3 == 0:
                print("Estimated Finish:" + time.ctime(((models.__len__() - c) / (i + 1)) * t_diff + t0))
            print(str(models[i][2]) + ": " + str(np.mean(loss[-10:])))
        else:
            c += 1
            print(model_strings[i] +  " Skipped")

#
# import os
# for x in os.listdir("Models/TransposedConv")[2:]:
#     if os.path.exists("Models/TransposedConv/" + x + "/loss.npy"):
#         path = "Models/TransposedConv/SavedNets/" + x
#         mean = np.mean(np.load("Models/TransposedConv/" + x + "/loss.npy"))
#         net_paths2 = net_paths2.append(
#             pd.DataFrame(
#                 [[x, mean, path + ".pt"]], columns=["name", "loss", "path"]
#             )
#         )
# net_paths2.sort_values("loss").to_csv('Models/TransposedConv/SavedNets/netPaths2.csv')
# pd.DataFrame(models_results2).to_csv("Models/TransposedConv/test2.csv")