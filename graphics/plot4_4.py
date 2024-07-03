import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

X = []
Y = []

with open('5_0.376.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        Y.append(float(ROWS[1]))
idx = np.argsort(X)
X = np.array(X)[idx]
Y = np.array(Y)[idx]
X2 = []
Y2 = []
with open('10_0.376.txt', 'r') as datafile:
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
with open('20_0.376.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X3.append(float(ROWS[0]))
        Y3.append(float(ROWS[1]))



idx = np.argsort(X3)
X3 = np.array(X3)[idx]
Y3 = np.array(Y3)[idx]


fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(0.2))
#ax.xaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

ax.yaxis.set_major_locator(MultipleLocator(0.2))
#ax.xaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
plt.xlim(1.0,2.0)
plt.ylim(0.2, 0.8)

plt.plot(X,Y, color='r', label='alpha=0.376')
plt.plot(X2,Y2, color='g', label='2')
plt.plot(X3,Y3, color='y', label='3')



plt.title('')
plt.xlabel('$k_{B}T/J_{0}$')
plt.ylabel('C/R')
plt.show()



# lattice = np.loadtxt("energia_total.txt", dtype=np.int32)
# plt.imshow(lattice)
# plt.title('Final Lattice Configuration')
# plt.colorbar()
# plt.show()

