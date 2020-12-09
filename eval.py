import torch
import Models as model
import custom_loss

specs = 4

if specs == 1:
    # Abb 6.1
    Net = model.UnpoolingNet
    dim_input = (100, 100, 3)
    pooling = torch.nn.MaxPool2d
    loss = custom_loss.weighted_mse_loss
    kernel_size = 3
    act_func1 = torch.nn.functional.relu
    act_func2 = torch.nn.functional.tanhshrink
    act_func3 = torch.nn.functional.hardtanh
    net_path = "UnpoolingConv/MaxMaxweightedMSE3100relutanhshrinkhardtanh3"

elif specs == 2:
    #Abb 6.2
    Net = model.TransposedNet
    net_path = "TransposedConv/MaxMaxBCE1100elueluhardtanh1"
    dim_input = (120, 120, 3)
    pooling = torch.nn.AvgPool2d
    loss = torch.nn.BCEWithLogitsLoss
    kernel_size = 1
    act_func1 = torch.nn.functional.elu
    act_func2 = torch.nn.functional.elu
    act_func3 = torch.nn.functional.hardtanh

elif specs == 3:
    Net = model.UnpoolingNet
    net_path = "UnpoolingConv/MaxMaxBCE1100eluhardtanhhardtanh2"
    dim_input = (100, 100, 3)
    pooling = torch.nn.MaxPool2d
    loss = torch.nn.BCEWithLogitsLoss
    kernel_size = 1
    act_func1 = torch.nn.functional.elu
    act_func2 = torch.nn.functional.hardtanh
    act_func3 = torch.nn.functional.hardtanh
else:
    # Transposed Convolutional Layer
    Net = model.TransposedNet
    net_path = "TransposedConv/AVGIntMSE3100elusoftsignhardtanh1"
    dim_input = (120, 120, 3)
    pooling = torch.nn.AvgPool2d
    loss = torch.nn.MSELoss
    kernel_size = 3
    act_func1 = torch.nn.functional.elu
    act_func2 = torch.nn.functional.softsign
    act_func3 = torch.nn.functional.hardtanh

netMax, loss_tracker, out, image, target, original, lr_data = model.train(Net=Net,
                                                                          epochs=1,
                                                                          batch_size=1,
                                                                          pooling=pooling,
                                                                          loss=loss,
                                                                          edge_finder="Scharr",
                                                                          sp_prob=0.0,
                                                                          resize=False,
                                                                          dim_input=dim_input,
                                                                          kernel_size=kernel_size,
                                                                          act_func1=act_func1,
                                                                          act_func2=act_func2,
                                                                          act_func3=act_func3,
                                                                          net_path=net_path
                                                                          )

# Plot results
if True:
    import matplotlib.pyplot as plt
    import numpy as np

    out_numpy = out.detach().squeeze(0).numpy()
    image = image.squeeze(0).numpy()

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
    ax[2, 1].plot(range(len(loss_tracker[-10:])), loss_tracker[-10:])
    ax[2, 2].plot(range(len(lr_data)), lr_data)
    ax[2, 2].set_yscale('log')
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[1, 2].axis('off')
    plt.show()
