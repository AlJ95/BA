import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor

t0 = tensor(0.)
t1 = tensor(1.)


def weighted_mse_loss():
    def loss(output: tensor, target: tensor):
        mse = torch.nn.MSELoss()
        n = (target.shape[-1] * target.shape[-2] + t1 + t1)

        weight = (n - target.sum()+1) / (target.sum()+1)
        loss1 = mse(output, target)

        n_pos_in_tar = torch.sum(target) + t1
        n_neg_in_tar = n - n_pos_in_tar + t1

        n_pos_in_out = (torch.sum(output) + t1)
        n_neg_in_out = n - n_pos_in_out

        n_true_pos = torch.sum(torch.where(output == target, output, t0)) + t1
        n_true_neg = torch.sum(torch.where(output == target, t1 - output, t0)) + t1

        loss2 = (n_pos_in_tar - n_true_pos)/n_pos_in_out + (n_neg_in_tar - n_true_neg)/n_neg_in_out * weight
        return loss1*n + loss2
    return loss


# mse = torch.nn.MSELoss()
#
# c = weighted_mse_loss()
# a = tensor([
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0],
# ]).float()
# b = torch.randint(0, 2, (7, 7)).float().unsqueeze(0)
# c(b, a)
# c(torch.zeros((7, 7)), a)
# c(torch.ones((7, 7)), a)
# c(a, a)
#
# z = torch.zeros((7,7)).float()
# losses = []
# losses2 = []
# defbereich = []
# for i in range(7):
#     for j in range(7):
#         if a[i, j] == t1:
#             z[i, j] = t1
#             defbereich.append(str(int((a.sum()-z.sum())))+"f.n.")
#             losses.append(c(z, a).numpy())
#             losses2.append(mse(z, a).numpy()*49)
# for i in range(7):
#     for j in range(7):
#         if a[i, j] == t1:
#             z[i, j] = t0
#             defbereich.append(str(int((a.sum()-z.sum())))+"f.p.")
#             losses.append(c(z, a).numpy())
#             losses2.append(mse(z, a).numpy()*49)
#
# fix, ax = plt.subplots(2)
# ax[0].imshow(a.numpy())
# ax[1].plot(defbereich, losses)
# ax[1].plot(defbereich, losses2)
# plt.show()
# #plt.savefig('Custom_Loss')
#
# dim = (5, 5)
# z = torch.zeros(dim)
# o = torch.ones(dim)
# r = torch.randint(0, 2, dim)
#
# for x in [z, o, r, c]:
#     for y in [z, o, r, c]:
#         if torch.any(x != y):
#             print(':'.join([str(x), str(y), str(weighted_mse_loss(x.float(), y.float(),tensor(2.)))]))
#             print("\n\n")
#
