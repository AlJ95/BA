import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
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

loss = [nn.BCEWithLogitsLoss, nn.MSELoss]  # "CUSTOM", "DICE"
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
          for y in loss
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