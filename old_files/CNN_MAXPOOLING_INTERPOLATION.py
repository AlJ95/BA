import timeit as t
import time as tt
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Transformations import forward, forward2
from Progress import save_everything, manage_load
from createImages_cv2 import produce_image

dim_original = (400, 300, 3)
dim_input = (400, 300, 3)
directory = "results_better_approach"


class Net(nn.Module):
    def __init__(self, poolfunc):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 1)
        self.conv2 = nn.Conv2d(12, 24, 1)
        self.conv3 = nn.Conv2d(24, 36, 1)

        self.unconv1 = nn.ConvTranspose2d(36, 24, 1)
        self.unconv2 = nn.ConvTranspose2d(24, 12, 1)
        self.unconv3 = nn.ConvTranspose2d(12, 1, 1)

        self.poolQ2 = poolfunc((2, 2))
        self.poolQ5 = poolfunc((5, 5))

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(36 * 40 * 30, 10 * 40 * 30)
        self.fc2 = nn.Linear(10 * 40 * 30, 36 * 40 * 30)

    def forward(self, x):

        # Max pooling over a (2, 2) window
        # torch.Size([1, 3, 1920, 1080])
        shapes = [x.shape]
        x = self.poolQ5(F.relu(self.conv1(x)))
        shapes.append(x.shape)
        x = self.poolQ2(F.relu(self.conv2(x)))
        # shapes.append(x.shape)
        x = F.relu(self.conv3(x))
        # torch.Size([1, 54, 40, 40])

        x = x.view(-1, self.num_flat_features(x))

        # torch.Size([1, 86400])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # unpooling section
        x = x.view(-1, 36, 40, 30)
        # torch.Size([1, 54, 40, 40])
        x = self.unconv1(x)
        x = nn.functional.interpolate(x, [shapes[-2][2], shapes[-2][3]],mode="bicubic").clamp(min=0, max=255)
        x = self.unconv2(x)
        x = nn.functional.interpolate(x, [shapes[-2][2], shapes[-2][3]],mode="bicubic").clamp(min=0, max=255)
        x = self.unconv3(x)
        # torch.Size([1, 18, 1920, 1080])
        # x = torch.max(torch.sigmoid(x), 1)[0]
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


t1 = tensor(1, requires_grad=True, dtype=torch.float32)
t0 = tensor(0, requires_grad=True, dtype=torch.float32)


def loss_func(output: torch.Tensor, target: torch.Tensor):
    found_edges = torch.where(torch.logical_and(
        output == target,
        output == torch.ones_like(output)),
        t1, t0)
    found_non_edges = torch.where(torch.logical_and(
        output == target,
        output == torch.zeros_like(output)),
        t1, t0)
    found_edges = torch.sum(found_edges) / torch.sum(target)
    found_non_edges = torch.sum(found_non_edges) / torch.sum(torch.where(target == t0, t1, t0))

    if torch.isnan(found_non_edges) or torch.isnan(found_edges):
        return 1
    else:
        return tensor(0.5, requires_grad=True) * (found_edges + found_non_edges)


def train(duration, epochs, batch_size, plot=True, save=True, load=True, title="no title"):
    print(tt.ctime(tt.time() + duration) + ' wird das Training mit Abschluss der Epoche beendet.')

    netMax = Net(nn.MaxPool2d)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(20))
    optimizer = optim.Adam(netMax.parameters(), lr=0.05)

    loss_tracker, netMax = manage_load(netMax, load, False, directory)

    writer = SummaryWriter('runs/temp')

    time = np.array([t.time.time()])
    epoch = 0
    while (epoch < epochs and tt.time() < time[0] + duration) or epoch == 0:
        for i in range(batch_size):
            original, gradient, target = produce_image(False, 15, salt_and_pepper_prob=0.00, dim=dim_original,
                                                       grad="Scharr")
            image = forward(gradient).unsqueeze(0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            out = netMax(image).squeeze(0)
            #out -= out.min(1, keepdim=True)[0]
            #out /= out.max(1, keepdim=True)[0]
            #out[torch.isnan(out)] = 0
            #out = torch.where(out > 0.8, t1, t0)

            # loss = loss_func(out, forward2(target).float().unsqueeze(0))
            loss = criterion(out, forward2(target).float().unsqueeze(0))
            loss.backward()
            optimizer.step()
            # print(out.shape)

            # print statistics
            loss_tracker = np.append(loss_tracker, loss.item())
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))

            if plot and i == 0:
                out_numpy = out.detach().squeeze(0).numpy()
                image = image.squeeze(0).numpy()

                fig, ax = plt.subplots(2, 3)
                ax[0, 0].imshow(image.transpose(2, 1, 0))
                ax[0, 1].imshow(out_numpy.transpose(1, 0), interpolation="nearest", cmap="inferno")
                ax[0, 2].imshow(np.where(out_numpy.transpose(1, 0) > 0.8, 255, 0).astype('int32'),
                                interpolation="nearest")
                ax[1, 0].imshow(target.transpose(1, 0), interpolation="nearest")
                ax[1, 1].imshow(original.transpose(1, 0, 2), interpolation="nearest")
                ax[0, 1].axis('off')
                ax[0, 0].axis('off')
                ax[1, 0].axis('off')
                ax[1, 1].axis('off')
                ax[0, 2].axis('off')
                plt.show()

        time = np.append(time, t.time.time() - np.sum(time))
        print('Estimated finish: ' + t.time.ctime(t.time.time() + (np.mean(time[1:]) * (epochs - epoch))))

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(range(len(loss_tracker)), loss_tracker, color="red")
        ax[0].plot(np.arange(3, len(loss_tracker) - 2), [np.mean(loss_tracker[x - 3:x + 3])
                                                         for x in np.arange(3, len(loss_tracker) - 2)])
        ax[1].plot(range(len(loss_tracker[-batch_size:])), loss_tracker[-batch_size:])
        plt.show()

        epoch += 1

    # Save
    if save:
        save_everything(netMax, loss_tracker, directory, title)
    print(str(epoch) + " epochs done.")

    return netMax, loss_tracker
