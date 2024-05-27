from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = 0.376
t =  0.6


t_end = t + 1
step = 10
l = ["./simulator", str(alpha), str(t), str(t_end), str(step), str(alpha)+".txt", "10"]
subprocess.Popen(l, stdout=subprocess.PIPE)

