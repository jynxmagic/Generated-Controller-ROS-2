import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

# Define Okabe–Ito / CUD palette
okabe_ito = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7"   # reddish purple
]

# Set it as the default color cycle
plt.rcParams['axes.prop_cycle'] = cycler(color=okabe_ito)

# Example plot
x = np.linspace(0, 10, 200)
plt.figure(figsize=(8, 5))

for i in range(len(okabe_ito)):
    plt.plot(x, np.sin(x + i), label=f"Line {i+1}")

plt.legend(ncol=2)
plt.title("Okabe–Ito (CUD) Color Palette")
plt.show()
