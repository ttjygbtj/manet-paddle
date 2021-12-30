import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import cnames, to_rgba

for k, i in list(cnames.items()):
    im = np.full((400, 400, 4), to_rgba(i))
    plt.imsave(f'{k}.png', im)
    print(2)