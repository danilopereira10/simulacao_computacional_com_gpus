import numpy as np
import csv
from matplotlib import pyplot as plt

X = []
Y = []

with open('0.285.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        Y.append(float(ROWS[1]))
idx = np.argsort(X)
X = np.array(X)[idx]
Y = np.array(Y)[idx]
X2 = []
Y2 = []
with open('0.25.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X2.append(float(ROWS[0]))
        Y2.append(float(ROWS[1]))
idx = np.argsort(X2)
X2 = np.array(X2)[idx]
Y2 = np.array(Y2)[idx]
X3 = []
X4 = []
Y3 = [] 
Y4 = []
X5 = []
Y5=[]
with open('0.2.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X3.append(float(ROWS[0]))
        Y3.append(float(ROWS[1]))

with open('0.15.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X4.append(float(ROWS[0]))
        Y4.append(float(ROWS[1]))

with open('0.0.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X5.append(float(ROWS[0]))
        Y5.append(float(ROWS[1]))

idx = np.argsort(X3)
X3 = np.array(X3)[idx]
Y3 = np.array(Y3)[idx]
idx = np.argsort(X4)
X4 = np.array(X4)[idx]
Y4 = np.array(Y4)[idx]
idx = np.argsort(X5)
X5 = np.array(X5)[idx]
Y5 = np.array(Y5)[idx]
plt.plot(X,Y, color='r', label='alpha=0.376')
plt.plot(X2,Y2, color='g', label='2')
plt.plot(X3,Y3, color='y', label='3')
plt.plot(X4,Y4,color='b', label='4')
plt.plot(X5,Y5,color='c', label='5')


plt.title('Line graph using CSV')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# lattice = np.loadtxt("energia_total.txt", dtype=np.int32)
# plt.imshow(lattice)
# plt.title('Final Lattice Configuration')
# plt.colorbar()
# plt.show()

