import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#from scipy import interpolate
from scipy.interpolate import make_interp_spline

X = []
Y = []
dict = {}



def write_file(filename):
    dict = {}
    s = filename.replace(".txt", "_fim.txt")
    f = open(s, "r")
    with open(filename, 'w') as datafile:
        plotting = csv.reader(f, delimiter=',')
        for ROWS in plotting:
            x = (float(ROWS[0]))
            y = (float(ROWS[1]))
            if x not in dict:
                dict[x] = Y
                datafile.write("{:.6f}, {:.6f} \n".format(x, y))
    f.close()

l2 = ["0.376", "0.637", "0.8", "1.0", "0.0", "0.2", "0.15", "0.25", "0.285", "5_0.285", "5_0.376", "5_0.637", "10_0.285",
      "10_0.376", "10_0.637", "20_0.285", "20_0.376", "20_0.637"]
for l in l2:
    write_file(l+".txt")

