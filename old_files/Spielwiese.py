import numpy as np

output = np.random.random((200,200))
output2 = 1 - output
target = np.where(np.random.random((200,200))>0.5,1,0)

loss = - np.mean(
    np.sum(
        [
            np.where(target == x[0], 1, 0) * np.log(x[1])
            for x in [[0, output], [1, output2]]]
    )
)