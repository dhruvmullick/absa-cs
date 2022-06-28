import numpy as np
from matplotlib import pyplot as plt

### Top-K sampling.

### Greedy. ALSC Results.
# baseline = [69.08, 71.68, 69.94, 71.39, 70.81, 70.81, 73.41, 75.72, 76.59, 73.70]
# commongen_0_1 = [73.99, 74.57, 74.57, 71.39, 74.57, 73.70, 75.14, 73.41, 74.86, 74.86]
# commongen_0_2 = [75.43, 73.12, 73.99, 70.81, 72.25, 70.81, 74.86, 69.65, 76.30, 69.94]
# commongen_0_5 = [70.23, 74.28, 69.94, 71.97, 71.97, 69.65, 73.70, 73.99, 74.86, 76.01]
# commongen_1_0 = [73.12, 74.86, 74.57, 72.54, 72.25, 65.61, 73.41, 71.68, 73.99, 74.28]


def add_to_plot(x, y):
    mean_val = np.mean(y)
    std_val = np.std(y)
    plt.plot(x * 10, y, 'x')
    plt.plot(x, mean_val, 'o')
    plt.errorbar(x, mean_val, yerr=std_val, xerr=0, ls='none', capsize=10)


add_to_plot([0], baseline)

add_to_plot([0.1], commongen_0_1)
add_to_plot(p[0.2], commongen_0_2)
# add_to_plot([0.5], commongen_0_5)
# add_to_plot([1.0], commongen_1_0)
plt.xlabel("Commongen Fraction")

# plt.xticks([0, 0.1, 0.2, 0.5, 1.0])
plt.show()
