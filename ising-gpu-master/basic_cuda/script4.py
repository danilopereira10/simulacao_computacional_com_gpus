from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = [0.376, 0.637, 0.8, 1.0]
t1 =  [0.6, 1.2, 1.4, 1.5]
t2 = [2.5, 2.7, 3.0,3.0]

j = 0
niters = str("100000")
for i in range (len(alpha)):
    t = t1[i]
    while(t <= t2[i]):
        l = ["./ising_basic", str(alpha[i]), str(t), "0", "0", str(alpha[i])+".txt", "10", niters]
        subprocess.check_call(l, stdout=subprocess.PIPE)
        t += 0.002
        j +=1

alpha = [0.285, 0.25, 0.2, 0.15, 0.0]
t1 =  [0.6, 0.8, 1.0, 1.0, 1.5]
t2 = [2.4, 2.4, 2.8, 3.0, 3.0]

j = 0
for i in range (len(alpha)):
    t = t1[i]

    
    while(t <= t2[i]):
        l = ["./ising_basic", str(alpha[i]), str(t), "0", "0", str(alpha[i])+".txt", "10", niters]
        subprocess.check_call(l, stdout=subprocess.PIPE)
        t += 0.002
        j +=1

alpha = [0.285, 0.376, 0.637]
n = [5, 10, 20]
t1 =  [0.9, 1.0, 1.5]
t2 = [2.1, 2.0 , 2.5]

j = 0

for i in range (len(alpha)):
    for j in range(len(n)):
        n2 = n[j]
        al = alpha[i]
        t = t1[i]
        while (t <= t2[i]):
            l = ["./ising_basic", str(alpha[i]), str(t), "0", "0", str(n2)+"_"+str(alpha[i])+".txt", str(n2), niters]
            subprocess.check_call(l, stdout=subprocess.PIPE)
            t += 0.002
            j += 1
