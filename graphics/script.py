from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = [0.376, 0.637, 0.8, 1.0]
t1 =  [0.6, 1.2, 1.4, 1.5]
t2 = [2.5, 2.7, 3.0,3.0]

j = 0
niters = "1000000"
for i in range (len(alpha)):
    t = t1[i]
    t_end = t + 0.02
    step = 0.1
    while(t <= t2[i]):
        l = ["./simulator", str(alpha[i]), str(t), str(t_end), str(step), str(alpha[i])+".txt", "10", niters]
        subprocess.Popen(l, stdout=subprocess.PIPE)
        t += 0.02
        t_end += 0.02
        j +=1
