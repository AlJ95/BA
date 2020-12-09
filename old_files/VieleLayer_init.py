import Modell_VieleLayer as model
# from CityDatasetPrepare import ValidationData as vd
# import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_loss import weighted_mse_loss

netMax, loss, out = model.train(
    duration=60 * 60 * 2,
    epochs=100,
    batch_size=20,
    pooling=nn.MaxPool2d,
    unpooling=nn.MaxUnpool2d,
    loss=weighted_mse_loss,
    edge_finder="Laplace",
    resize=False,
    # size=models[i][4],
    size=(100, 100, 3),
    kernel_size=3,
    plot=False,
    save=True,
    load=False,
    title="VieleLayerMaxMaxWeighted3Softplusthanshrinkhardthan",
    directory="Models/VieleLayer",
    activation_function1=F.softplus,
    activation_function2=F.tanhshrink,
    activation_function3=F.hardtanh,
    nn_input="Original",
    start_lr=0.1,
    end_lr=10*-15
)
