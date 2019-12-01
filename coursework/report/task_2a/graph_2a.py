import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# ax.scatter(np.array([1, 0.023, 0.0015]), np.array([1,1,1]))
ax.plot(np.array([0,1]), np.array([0,1]), linestyle="--", c="red", zorder=-1)
ax.scatter(1,1, label='Stage 0')
ax.scatter(0.023, 1, label='Stage 1')
ax.scatter(0.0015, 1, label='Stage 2')
plt.text(0.47,0.5, "TPR = FPR", fontsize=16, rotation=38, rotation_mode='anchor')
plt.legend()
plt.xlabel("False Positive Rate ")
plt.ylabel("True Positive Rate")
plt.title("True Positive Rate against False Positive Rate")
plt.show()
