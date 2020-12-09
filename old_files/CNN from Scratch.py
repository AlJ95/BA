import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

from createImages_cv2 import produce_image

## 240*180
#tensorflow book (S.181):
# L' = ( L - K + 2P ) / S
class Net(nn.Module):
    def __init__(self, poolfunc, unpoolfunc):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 1, dilation=1)
        self.conv2 = nn.Conv2d(18, 36, 1)

        self.unconv2 = nn.Conv2d(36, 18, 1)
        self.unconv1 = nn.Conv2d(18, 3, 1)

        self.pool = poolfunc((4, 3), return_indices=True)
        self.poolQ = poolfunc((3, 3), return_indices=True)
        self.unpool = unpoolfunc((4, 3))
        self.unpoolQ = unpoolfunc((3, 3))

        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(54 * 58 * 58, 1080 * 8)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # torch.Size([1, 3, 240, 180])
        output1, indices1 = self.pool(F.relu(self.conv1(x)))
        output2, indices2 = self.poolQ(F.relu(self.conv2(output1)))

        # torch.Size([1, 54, 40, 40])
        output3 = output2.view(-1, self.num_flat_features(output2))
        # # #
        # torch.Size([1, 86400])
        # x = F.relu(self.fc1(x))
        # # #

        # unpooling section
        output2 = output3.view(-1, 36, 20, 20)

        # torch.Size([1, 36, 20, 20])

        output1 = self.unpoolQ(output2, indices2)
        output1 = self.unconv2(output1)
        output = self.unpool(output1, indices1)
        output = self.unconv1(output)
        # torch.Size([1, 18, 240, 180])

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


netMax = Net(nn.MaxPool2d, nn.MaxUnpool2d)

#image = produce_image()
image1 = cv2.resize(image, (180, 240), interpolation=cv2.INTER_NEAREST)[:, :, 0:3].transpose((2, 0, 1))
image2 = (image1 + [np.amin(image1, axis=0) for i in range(3)]) / [np.amax(image1, axis=0) for i in range(3)]
image3 = torch.from_numpy(image2).unsqueeze(0).float()


input = torch.randn(1, 3, 100, 100)
out = netMax(image3).squeeze(0)
print(out.shape)
print(out)
out_numpy = out.detach().numpy().transpose(2, 1, 0)
out_numpy -= np.array([np.amin(out_numpy, axis=2) for i in range(3)]).transpose(1, 2, 0)
out_numpy /= np.array([np.amax(out_numpy, axis=2) for i in range(3)]).transpose(1, 2, 0)
plt.imshow(out_numpy)
plt.show()
print(torch.mean(image3))
print(np.mean(out_numpy))
