import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('outputs/predictions/prediction_2.npy')

plt.matshow(data)
plt.show()