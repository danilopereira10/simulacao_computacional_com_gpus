from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = [0.376, 0.637, 0.8, 1.0]
t1 =  [0.6, 1.2, 1.4, 1.5]
t2 = [2.5, 2.7, 3.0,3.0]

j = 0
nwarmup = "100"
niters = "1000"
for i in range (len(alpha)):
    t = t1[i]
    while(t <= t2[i]):
        t_end = t + 0.02
        l = ["./ising_basic", str(alpha[i]), str(t), str(t_end), str(1), str(alpha[i])+"_cu.txt", "10", niters]
        subprocess.check_call(l, stdout=subprocess.PIPE)
        t += 0.02
        j +=1
