import createImages_cv2 as cr
import numpy as np

probs = []
for i in range(1000):
    if i % 50 == 1:
        print(i)
    _, _, target = cr.produce_image(False, 15, 0, (100, 100, 3))
    a, b = target.shape
    probs.append(np.sum(target) / (a*b))

#np.mean(probs)
#Out[3]: 0.004784597800925926
