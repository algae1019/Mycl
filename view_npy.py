import numpy as np
import matplotlib.pyplot as plt

data = np.load('outputs/predictions/prediction_2.npy')

plt.matshow(data)
plt.show()