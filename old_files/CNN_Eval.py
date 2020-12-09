import timeit as t
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Progress import load_last
from createImages_cv2 import produce_image


class Net(nn.Module):
    def __init__(self, poolfunc, unpoolfunc):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 1, dilation=1)
        self.conv2 = nn.Conv2d(12, 24, 1)
        self.conv3 = nn.Conv2d(24, 36, 1)

        self.unconv1 = nn.Conv2d(36, 24, 1)
        self.unconv2 = nn.Conv2d(24, 12, 1)
        self.unconv3 = nn.Conv2d(12, 3, 1)

        self.pool = poolfunc((4, 3), return_indices=True)
        self.poolQ = poolfunc((3, 3), return_indices=True)
        self.unpool = unpoolfunc((4, 3))
        self.unpoolQ = unpoolfunc((3, 3))

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(36 * 40 * 40, 10 * 40 ** 2)
        self.fc2 = nn.Linear(10 * 40 ** 2, 36 * 40 * 40)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # torch.Size([1, 3, 1920, 1080])
        output1, indices1 = self.pool(F.relu(self.conv1(x)))
        output2, indices2 = self.pool(F.relu(self.conv2(output1)))
        output3, indices3 = self.poolQ(F.relu(self.conv3(output2)))

        # torch.Size([1, 54, 40, 40])
        x = output3.view(-1, self.num_flat_features(output3))
        # print(x.shape)
        # # #
        # torch.Size([1, 86400])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # # #

        # unpooling section
        output3 = x.view(-1, 36, 40, 40)
        # print(output3.shape)
        # torch.Size([1, 54, 40, 40])
        output2 = self.unpoolQ(output3, indices3)
        # print(output2.shape)
        output2 = self.unconv1(output2)
        # print(output2.shape)
        output1 = self.unpool(output2, indices2)
        # print(output1.shape)
        output1 = self.unconv2(output1)
        # print(output1.shape)
        output = self.unpool(output1, indices1)
        # print(output.shape)
        output = self.unconv3(output)
        # print(output.shape)
        # torch.Size([1, 18, 1920, 1080])
        output = torch.max(F.hardshrink(output, 0.), 1)[0]
        # print(output.shape)
        return x, output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eval_model():
    # Specify a path
    # PATH = "state_dict_model_loss_40_epochs_10_img_per_batch_lr_0_001_loss_0_000974.pt"
    # Load

    netMax = Net(nn.MaxPool2d, nn.MaxUnpool2d)
    loss_tracker, netMax = load_last(netMax)
    # model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(netMax.parameters(), lr=0.0001)

    netMax.eval()

    original, gradient, target = produce_image(False, 30)
    image = torch.from_numpy(gradient[:, :, 0:3].transpose([2, 1, 0])).unsqueeze(0).float()

    result = netMax(image)[1].squeeze(0).detach().numpy()
    plt.imshow(np.where(result > 0.015, 1, 0).transpose(1, 0).astype('int32'), interpolation="nearest")
    plt.show()

    return result
