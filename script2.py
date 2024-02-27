from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = [0.285, 0.25, 0.2, 0.15, 0.0]
t1 =  [0.6, 0.8, 1.0, 1.0, 1.5]
t2 = [2.4, 2.4, 2.8, 3.0, 3.0]

j = 0
for i in range (len(alpha)):
    t = t1[i]
    
    while(t <= t2[i]):
        l = ["./simulator", str(alpha[i]), str(t), str(alpha[i])+".txt"]
        subprocess.Popen(l, stdout=subprocess.PIPE)
        t += 0.02
        j +=1
