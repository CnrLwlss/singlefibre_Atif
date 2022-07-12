import matplotlib; matplotlib.use("module://mplcairo.qt")
from matplotlib import pyplot as plt
from mplcairo import operator_t
import numpy as np

x1 = np.random.randn(1000)
y1 = np.random.randn(1000)
x2 = np.random.randn(1000) * 5
y2 = np.random.randn(1000)
fig, ax = plt.subplots()
# The figure and axes background must be made transparent.
fig.patch.set(alpha=0)
ax.patch.set(alpha=0)
pc1 = ax.scatter(x1, y1, c='b', edgecolors='none')
pc2 = ax.scatter(x2, y2, c='r', edgecolors='none')
operator_t.ADD.patch_artist(pc2)  # Use additive blending.
plt.show()
