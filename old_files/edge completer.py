import matplotlib.pyplot as plt
import numpy as np
from copy import copy
#from CNN_Eval import eval_model
import time

result = eval_model()

start = lambda x: max(0, x)
end = lambda x, dim: min(x, result.shape[dim] - 1)

# result = np.zeros((50, 50))
# result[10, 10] = 1
# result[20, 20] = 1
#i0, j0 = 10, 10
counter = 0
completed_edges = np.where(copy(result)>0.15, 1, 0)
tt = time.time()
timer = True
for i0 in range(completed_edges.shape[0]):
    for j0 in range(completed_edges.shape[1]):
        if completed_edges[i0, j0] == 1 and i0 not in [0, completed_edges.shape[0] - 1] and \
                j0 not in [0, completed_edges.shape[1] - 1] and timer:

            matrix_expander = 1
            while np.sum(completed_edges[start(i0 - matrix_expander):end(i0 + matrix_expander + 1, 0),
                         start(j0 - matrix_expander):end(j0 + matrix_expander + 1, 1)]) < 2:
                matrix_expander += 1
            if matrix_expander > 2:
                counter += 1
                if time.time() - tt > 60:
                    timer = False
                if counter % 100 == 0:
                    print(i0, j0, counter)
                # Indices of Interest
                IoI = np.argwhere(
                    completed_edges[start(i0 - matrix_expander):end(i0 + matrix_expander + 1, 0),
                                    start(j0 - matrix_expander):end(j0 + matrix_expander + 1, 1)] == 1)
                dims = completed_edges[start(i0 - matrix_expander):end(i0 + matrix_expander + 1, 0),
                                       start(j0 - matrix_expander):end(j0 + matrix_expander + 1, 1)].shape

                IoI = IoI[np.bitwise_or(IoI[:, 0] == 0,
                                        np.bitwise_or(IoI[:, 0] == dims[0] - 1,
                                                      np.bitwise_or(IoI[:, 1] == 0, IoI[:, 1] == dims[1] - 1)))]

                if IoI.shape == (2, ):
                    i1, j1 = IoI
                else:
                    i1, j1 = IoI[0]

                counter2 = 0
                ihat, jhat = i0, j0
                di, dj = np.sign(i1 - i0), np.sign(j1 - j0)
                print(max(abs((dims[0]+1)/2-i0), abs((dims[1]+1)/2-j0)))
                print("###########################")
                while ihat != i1 and jhat != j1 and counter < max(abs((dims[0]+1)/2-i1), abs((dims[1]+1)/2-j1)):
                    ihat, jhat = ihat + di, jhat + dj
                    completed_edges[ihat, jhat:jhat + 2 * np.sign(j1 - j0)] = 1
                    di, dj = np.sign(i1 - i0), np.sign(j1 - j0)
                    counter2 += 1

                # while i0 + di != i1:
                #     di += np.sign(i1 - i0)
                #     completed_edges[i0 + di, j0 + dj] = 1
                #
                # while j0 + dj != j1:
                #     dj += np.sign(j1 - j0)
                #     completed_edges[i0 + di, j0 + dj] = 1

fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.where(result > 0.015, 1, 0).transpose(1, 0).astype('int32'), interpolation="nearest")
ax[1].imshow(completed_edges.transpose(1, 0).astype('int32'), interpolation="nearest")
plt.show()
