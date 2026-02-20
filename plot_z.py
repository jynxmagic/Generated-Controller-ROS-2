import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('traj.csv')

z = data['z']
gz = data['gz']

plt.plot(z - gz, label='z')
plt.axhline(0, color='red', linestyle='--', label='target z')
plt.legend()
plt.xlabel('Time step')
plt.show()
