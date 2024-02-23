import numpy as np
import csv
from matplotlib import pyplot as plt

X = []
Y = []
with open('energia_total.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X.append(int(ROWS[0]))
        Y.append(float(ROWS[1]))
plt.plot(X,Y)
plt.title('Line graph using CSV')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# lattice = np.loadtxt("energia_total.txt", dtype=np.int32)
# plt.imshow(lattice)
# plt.title('Final Lattice Configuration')
# plt.colorbar()
# plt.show()

