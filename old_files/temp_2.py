import pandas as pd
import numpy as np
import os

losses = [[[], [], []], [[], [], []], [[], [], []]]
for name in os.listdir("Models/AlleVariationen2")[2:-1]:
    for x in range(3):
        for y in range(3):
            if name.__contains__(["xMSE", "xBCE", "xwei"][x]):
                if name.__contains__(["1100", "3100", "5100"][y]):
                    losses[x][y].append(np.mean(np.load("Models/AlleVariationen2/" + name + "/loss.npy")))

for i in range(3):
    for j in range(3):
        print(["xMSE", "xBCE", "xwei"][i] + [" - Kernel 1"," - Kernel 3"," - Kernel 5"][j] + ": "
              + str(np.mean(losses[i][j])))




# tansp = pd.read_csv("Models/TransposedConv/SavedNets/netPaths.csv")
# losses2 = [[],[],[]]
# for i in range(tansp.iloc[:,0].__len__()):
#     for x in range(3):
#         if tansp.iloc[i, 1].__contains__(["xMSE", "xBCE", "xwei"][x]):
#             losses2[x].append(np.mean(np.load("Models/TransposedConv/" + tansp.iloc[i, 1] + "/loss.npy")))
#
#
# for i in range(3):
#     print(["xMSE", "xBCE", "xwei"][i] + ": " + str(np.mean(losses2[i])))