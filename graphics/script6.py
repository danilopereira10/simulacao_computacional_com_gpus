from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = [1.0]
t1 =  [2.5]
t2 = [2.5]

j = 0
nwarmup = "100"
niters = "1000"
for i in range (len(alpha)):
    t = t1[i]
    while(t <= t2[i]):
        l = ["./ising_basic", str(alpha[i]), str(t), str(alpha[i])+"_cu.txt", "176", "10", nwarmup, niters]
        subprocess.check_call(l, stdout=subprocess.PIPE)
        t += 0.02
        j +=1
