import os
import numpy as np
import torch
import matplotlib.pyplot as plt

temp_dir_net = 'D:/Programming/Python/Projekte/VanPixel Spielwiese/temp.pt'
temp_dir_loss = 'D:/Programming/Python/Projekte/VanPixel Spielwiese/loss_temp.pt'


def get_path(directory):
    return os.path.join("D:/Programming/Python/Projekte/VanPixel Spielwiese/", directory)


def manage_load(netMax, load, load_temp, directory):
    if load_temp:
        loss_tracker = np.load(temp_dir_loss)
        netMax.load_state_dict(torch.load(temp_dir_net))
    elif load:
        loss_tracker, netMax = load_last(netMax, directory)
    else:
        loss_tracker = np.array([])

    return loss_tracker, netMax


def save_everything(netMax, loss_tracker, directory, title):
    PATH = get_path(directory + "/" + title)
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    np.save(os.path.join(PATH, "loss"), loss_tracker)
    torch.save(netMax.state_dict(), os.path.join(PATH, "load_state_dict.pt.psd.psd.psd"))
    #print("Saved in " + PATH)
    pass


def load_last(netMax, directory):
    i = 1
    while os.path.exists(get_path(directory)+str(i)):
        i += 1
    PATH = get_path(directory)+str(i-1)
    netMax.load_state_dict(torch.load(os.path.join(PATH, "load_state_dict.pt.psd.psd.psd")))
    loss_tracker = np.load('loss.npy')
    print("Loaded from " + PATH)
    return loss_tracker, netMax


def save_fig(fig: plt.Figure, ax: plt.Axes, directory, subdirectory, epoch, title):
    if epoch != "_LOSS":
        PATH = get_path(directory + "/" + subdirectory)
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        fig.savefig(os.path.join(PATH, title + str(epoch) + ".png"))
    else:
        PATH = get_path(directory)
        fig.savefig(os.path.join(PATH, subdirectory, title + ".png"))


