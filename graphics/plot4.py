import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#from scipy import interpolate
from scipy.interpolate import make_interp_spline

X = []
Y = []

with open('0.376.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        Y.append(float(ROWS[1]))
idx = np.argsort(X)
X = np.array(X)[idx]
Y = np.array(Y)[idx]
X2 = []
Y2 = []
with open('0.637.txt', 'r') as datafile:
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

with open('0.8.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X3.append(float(ROWS[0]))
        Y3.append(float(ROWS[1]))

with open('1.0.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for ROWS in plotting:
        X4.append(float(ROWS[0]))
        Y4.append(float(ROWS[1]))

idx = np.argsort(X3)
X3 = np.array(X3)[idx]
Y3 = np.array(Y3)[idx]
idx = np.argsort(X4)
X4 = np.array(X4)[idx]
Y4 = np.array(Y4)[idx]
xticks = []
xl = []
i = 5
while (i < 31):
    xticks.append(i/10)
    if ((i%5) == 0):
        xl.append(i/10)
    else:
        xl.append("j")
    i += 1
yticks = []
i = 5
while (i < 16):
    yticks.append(i/10)
    i += 1


#plt.xticks(xticks)
yticks = [0, 0.5, 1.0, 1.5]
fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(0.5))
#ax.xaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

ax.yaxis.set_major_locator(MultipleLocator(0.5))
#ax.xaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

#tck = interpolate.splrep(X, Y)
X_Y_Spline = make_interp_spline(X, Y)
 
# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(X.min(), X.max(), 500)
Y_ = X_Y_Spline(X_)

X_Y_Spline2 = make_interp_spline(X2, Y2)
X2_ = np.linspace(X2.min(), X2.max(), 500)
Y2_ = X_Y_Spline2(X2_)
X_Y_Spline3 = make_interp_spline(X3, Y3)
X3_ = np.linspace(X3.min(), X3.max(), 500)
Y3_ = X_Y_Spline3(X3_)
X_Y_Spline4 = make_interp_spline(X4, Y4)
X4_ = np.linspace(X4.min(), X4.max(), 500)
Y4_ = X_Y_Spline4(X4_)

plt.xlim(0.5,3.0)
plt.ylim(0, 1.5)
#plt.yticks(yticks)

#plt.plot(X_,Y_, color='r', label='alpha=0.376')
#plt.plot(X2_,Y2_, color='g', label='2')
#plt.plot(X3_,Y3_, color='y', label='3')
#plt.plot(X4_,Y4_,color='b', label='4')
# plt.plot(X,Y, 'or', color='r', label='alpha=0.376')
# plt.plot(X2,Y2, 'og', color='g', label='2')
# plt.plot(X3,Y3, 'oy', color='y', label='3')
# plt.plot(X4,Y4, 'ob', color='b', label='4')
plt.plot(X,Y, color='r', label='alpha=0.376')
plt.plot(X2,Y2, color='g', label='2')
plt.plot(X3,Y3, color='y', label='3')
plt.plot(X4,Y4, color='b', label='4')

plt.title('')
plt.xlabel('$k_{B}T/J_{0}$')
plt.ylabel('C/R')
plt.show()


# lattice = np.loadtxt("energia_total.txt", dtype=np.int32)
# plt.imshow(lattice)
# plt.title('Final Lattice Configuration')
# plt.colorbar()
# plt.show()

