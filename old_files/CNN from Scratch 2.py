import timeit as t

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Progress import save_everything, load_last
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

        # torch.Size([1, 86400])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # unpooling section
        output3 = x.view(-1, 36, 40, 40)

        # torch.Size([1, 54, 40, 40])
        output2 = self.unpoolQ(output3, indices3)
        output2 = self.unconv1(output2)
        output1 = self.unpool(output2, indices2)
        output1 = self.unconv2(output1)
        output = self.unpool(output1, indices1)
        output = self.unconv3(output)
        # torch.Size([1, 18, 1920, 1080])
        output = torch.max(torch.sigmoid(output), 1)[0]
        return x, output

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


netMax = Net(nn.MaxPool2d, nn.MaxUnpool2d)
criterion = nn.MSELoss()
optimizer = optim.Adam(netMax.parameters(), lr=0.001)
save, load, load_temp = True, False, True

if load_temp:
    loss_tracker = np.load('D:/Programming/Python/Projekte/VanPixel Spielwiese/loss_temp.npy')
    netMax.load_state_dict(torch.load('D:/Programming/Python/Projekte/VanPixel Spielwiese/temp.pt'))
elif load:
    loss_tracker, netMax = load_last(netMax)
else:
    loss_tracker = np.array([])

writer = SummaryWriter('runs/temp')

epochs = 3
batch = 10
time = np.array([t.time.time()])
for epoch in range(epochs):
    for i in range(batch):
        original, gradient, target = produce_image(False, 15 + 2 * (epoch+1), salt_and_pepper_prob=0.05)
        image = torch.from_numpy(gradient[:, :, 0:3].transpose([2, 1, 0])).unsqueeze(0).float()
        # print(image.shape)
        # print(torch.typename(image))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimizer
        out = netMax(image)[1].squeeze(0)
        loss = criterion(out, torch.from_numpy(target.transpose(1, 0)).float())
        loss.backward()
        optimizer.step()
        # print(out.shape)

        # print statistics
        loss_tracker = np.append(loss_tracker, loss.item())
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, loss.item()))

        plot = True
        if plot:
            out_numpy = out.detach().numpy()
            # print(out_numpy.shape)
            # out_numpy = out_numpy
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(image.squeeze(0).numpy().transpose(2, 1, 0))
            ax[0, 1].imshow(np.where(out_numpy.transpose(1, 0) > 0.75, 1, 0).astype('int32'),
                            interpolation="nearest")
            ax[1, 0].imshow(target, interpolation="nearest")
            ax[1, 1].imshow(original, interpolation="nearest")
            ax[0, 1].axis('off')
            ax[0, 0].axis('off')
            ax[1, 0].axis('off')
            ax[1, 1].axis('off')
            plt.show()

    # Save
    if save:
        save_everything(netMax, loss_tracker)
    time = np.append(time, t.time.time() - np.sum(time))
    print('Estimated finish: ' + t.time.ctime(t.time.time() + (np.mean(time[1:]) * (epochs - epoch))))


    fig, ax = plt.subplots(1, 2)
    ax[0].plot(range(len(loss_tracker)), loss_tracker)
    ax[1].plot(range(len(loss_tracker[-batch:])), loss_tracker[-batch:])
    plt.show()
