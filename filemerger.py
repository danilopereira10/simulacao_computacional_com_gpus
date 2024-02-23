import numpy as np
import csv
from matplotlib import pyplot as plt

X = []
Y = []
for i in range(31):
    file1 = open('valores0.txt')
    file2 = open('valores'+str(i+1)+".txt")
    content1 = file1.read()
    content2 = file2.read()
with open('valores0.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        Y.append(float(ROWS[1]))

X2 = []
Y2 = []
with open('valores31.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X2.append(float(ROWS[0]))
        Y2.append(float(ROWS[1]))
plt.plot(X,Y, color='r', label='1')
plt.plot(X2,Y2, color='g', label='2')


plt.title('Line graph using CSV')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# lattice = np.loadtxt("energia_total.txt", dtype=np.int32)
# plt.imshow(lattice)
# plt.title('Final Lattice Configuration')
# plt.colorbar()
# plt.show()

