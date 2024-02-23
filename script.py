from multiprocessing.pool import ThreadPool
import subprocess

def f(x):
    a,b,c = x
#alpha = [0.285, 0.376, 0.637, 0.8, 1.0]
alpha = [0.285, 1.0]


j = 0
for i in range (len(alpha)):
    t = 0.0
    
    while(t < 3.1):
        l = ["./simulator", str(alpha[i]), str(t), "valores"+str(j)+".txt"]
        subprocess.Popen(l, stdout=subprocess.PIPE)
        t += 0.1
        j +=1
