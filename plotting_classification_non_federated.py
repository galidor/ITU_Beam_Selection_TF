import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rc('font', family='serif')

proposed_accuracy = np.load("proposed.npz")["classification"]
baseline_accuracy = np.load("baseline.npz")["classification"]

plt.plot(np.arange(1, 31), proposed_accuracy[:30], marker='o')
plt.plot(np.arange(1, 31), baseline_accuracy[:30], marker='^')

plt.grid()
plt.xlabel("K")
plt.ylabel("Top-K accuracy")
plt.legend(["2D conv network", "Baseline network"])

# plt.show()
plt.savefig("accuracy.pdf")
