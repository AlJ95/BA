import timeit as t
import time as tt
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Transformations import forward, forward2
from Progress import save_everything, manage_load, save_fig
from createImages_cv2 import produce_image

# often used variables
t1 = tensor(1, requires_grad=True, dtype=torch.float32)
t0 = tensor(0, requires_grad=True, dtype=torch.float32)


class Net(nn.Module):
    __first_interp__ = True
    __dim_input__ = (1920, 1080, 3)

    def __init__(self, pool_function, kernel_size, dim_input, activation_function1,
                 activation_function2, activation_function3):
        super(Net, self).__init__()
        padding = int((kernel_size - 1) / 2)
        Net.__dim_input__ = dim_input
        self.pool_func_is_MaxPool = pool_function == nn.MaxPool2d
        self.activation_function1 = activation_function1
        self.activation_function2 = activation_function2
        self.activation_function3 = activation_function3

        self.conv1 = nn.Conv2d(3, 12, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(12, 24, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(24, 36, kernel_size, padding=padding)

        self.unconv1 = nn.ConvTranspose2d(36, 24, 2, padding=0, stride=2)
        self.unconv2 = nn.ConvTranspose2d(24, 12, 2, padding=0, stride=2)
        self.unconv3 = nn.ConvTranspose2d(12, 1, 2, padding=0, stride=2)

        self.poolQ1 = nn.AvgPool2d((2, 2))
        self.poolQ2 = nn.LPPool2d(2, (2, 2))
        self.poolQ3 = nn.MaxPool2d((2, 2))

        # an affine operation: y = Wx + b
        self.flatten_neurons = int(Net.__dim_input__[0] * Net.__dim_input__[1] / 2**6)
        self.fc1 = nn.Linear(36 * self.flatten_neurons, 10 * self.flatten_neurons)
        self.fc2 = nn.Linear(10 * self.flatten_neurons, 36 * self.flatten_neurons)

    def forward(self, x):
        # pooling / convolution section
        x = self.poolQ1(F.relu(self.conv1(x)))
        x = self.poolQ2(F.relu(self.conv2(x)))
        x = self.poolQ3(F.relu(self.conv3(x)))

        # flattened section
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 36, int(Net.__dim_input__[0] / 8), int(Net.__dim_input__[1] / 8))

        # unpooling and unconvolution section
        x = self.activation_function1(self.unconv1(x))
        x = self.activation_function2(self.unconv2(x))
        if self.activation_function3 == torch.threshold:
            x = self.activation_function3(self.unconv3(x), threshold=1, value=0)
        elif self.activation_function3 == nn.Hardtanh:
            x = self.activation_function3(self.unconv3(x), min_val=0.0, max_val=0.0)
        else:
            x = self.activation_function3(self.unconv3(x))

        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def identity(x):
    return x


def train(duration, epochs, batch_size,
          pooling, loss, edge_finder, resize: bool, size, kernel_size,
          plot=True, save=False, load=True, title="no title", directory="models", activation_function1=identity,
          activation_function2=identity, activation_function3=torch.sigmoid, nn_input="Gradient", start_lr=0.01,
          end_lr=10**-5):
    dim_input = size
    out_numpy = None
    learn_rate, lr_data = start_lr, []

    print(tt.ctime(tt.time() + duration) + ' wird das Training mit Abschluss der Epoche beendet.')

    netMax = Net(pooling, kernel_size, dim_input, activation_function1, activation_function2,
                 activation_function3)
    if loss == nn.BCEWithLogitsLoss:
        if size == (100, 100, 3):
            criterion = loss(weight=torch.tensor(10))
        else:
            criterion = loss(weight=torch.tensor(10))
    else:
        criterion = loss()
    optimizer = optim.Adam(netMax.parameters(), lr=learn_rate)

    # loss_tracker, netMax = manage_load(netMax, load, False, directory)
    netMax.load_state_dict(torch.load("D:/Programming/Python/Projekte/VanPixel "
                                      "Spielwiese/Models/TransposedConv/AVGIntMSE3100elusoftsignhardtanh1/"
                                      "load_state_dict.pt.psd.psd.psd"))

    time = np.array([t.time.time()])
    epoch = 0
    batch_diff = []
    while (learn_rate >= end_lr and tt.time() < time[0] + duration) or epoch < epochs:
        for i in range(batch_size):
            original, gradient, target = produce_image(False, 15, salt_and_pepper_prob=0.00, dim=dim_input,
                                                       grad=edge_finder)
            if nn_input == "Gradient":
                image = forward(gradient, resize, dim_input).unsqueeze(0)
            else:
                image = forward(original, resize, dim_input).unsqueeze(0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            out = netMax(image).squeeze(0).clamp(0, 1)
            prep_target = forward2(target, resize, dim_input).float().unsqueeze(0)
            loss = criterion(out, prep_target)
            loss.backward()
            optimizer.step()

            # print statistics
            loss_tracker = np.append(loss_tracker, loss.item())
            out_numpy = out.detach().squeeze(0).numpy()

            if i == batch_size - 1:

                image = image.squeeze(0).numpy()
                if plot or save:
                    fig, ax = plt.subplots(3, 3)
                    ax[0, 0].imshow(image.transpose(2, 1, 0).clip(0, 1), cmap="seismic")
                    ax[0, 1].imshow(out_numpy.transpose(1, 0), interpolation="nearest", cmap="inferno")
                    ax[0, 2].imshow(np.where(out_numpy.transpose(1, 0) > 0.8, 255, 0).astype('int32'),
                                    interpolation="nearest")
                    ax[1, 0].imshow(target.transpose(1, 0), interpolation="nearest")
                    ax[1, 1].imshow(original.transpose(1, 0, 2), interpolation="nearest")
                    ax[2, 0].plot(range(len(loss_tracker)), loss_tracker, color="red")
                    ax[2, 0].plot(np.arange(5, len(loss_tracker) - 4), [np.mean(loss_tracker[x - 5:x + 5])
                                                                        for x in np.arange(5, len(loss_tracker) - 4)])
                    ax[2, 1].plot(range(len(loss_tracker[-batch_size:])), loss_tracker[-batch_size:])
                    ax[2, 2].plot(range(len(lr_data)), lr_data)
                    ax[2, 2].set_yscale('log')
                    ax[0, 0].axis('off')
                    ax[0, 1].axis('off')
                    ax[0, 2].axis('off')
                    ax[1, 0].axis('off')
                    ax[1, 1].axis('off')
                    ax[1, 2].axis('off')

                    plt.title(title)
                    if save:
                        save_fig(fig, ax, directory, title, str(epoch), "figure_from_epoch")

                    save_fig(fig, ax, directory, "figure_from_last_epoch", "", title)
                    if plot:
                        plt.show()
                    plt.close('all')

            lr_data.append(learn_rate)
        time = np.append(time, t.time.time() - np.sum(time))

        if plot or save:
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(range(len(loss_tracker)), loss_tracker, color="red")
            ax[0].plot(np.arange(5, len(loss_tracker) - 4), [np.mean(loss_tracker[x - 5:x + 5])
                                                             for x in np.arange(5, len(loss_tracker) - 4)])
            ax[1].plot(range(len(loss_tracker[-batch_size:])), loss_tracker[-batch_size:])
            if plot:
                plt.show()
            if save:
                save_fig(fig, ax, directory, "losses", "_LOSS", title)

        epoch += 1

        batch_diff.append(np.mean(loss_tracker[-batch_size:]))
        if rek_batch_comparison(batch_diff, learn_rate, 4) and epoch > 4:
            learn_rate /= 10
            optimizer = optim.Adam(netMax.parameters(), lr=learn_rate)
    # Save
    if save:
        save_everything(netMax, loss_tracker, directory, title)
    print(str(epoch) + " epochs done.")
    plt.close('all')

    return netMax, loss_tracker, out_numpy


def rek_batch_comparison(batch_diff, learn_rate, elements=4):
    if elements < 3:
        return True
    else:
        return (np.prod([x / batch_diff[-1] for x in batch_diff[-elements:-1]]) < 1 + learn_rate * elements and
                rek_batch_comparison(batch_diff, learn_rate, elements-1))