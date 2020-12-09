import numpy as np
import os
import pandas as pd

path = "Models/AlleVariationen2TL"

losses = pd.DataFrame([], columns=["name", "loss"])
for name in os.listdir(path)[2:-1]:
    losses = losses.append(
        pd.DataFrame([
            [name, np.mean(np.load(os.path.join(path,name,"loss.npy")))]
        ], columns=["name", "loss"])
    )

losses.sort_values("loss").to_csv("TransferLearningLosses.csv")