import matplotlib.pyplot as plt
import random

def prob(colour_total,total):
    p = (colour_total/total)*100
    return p

num_trails = []
# this loop does 100000 trails of 100 attempts
for trails in range(0,100000):
    black_num = 7
    red_num = 3
    total = black_num + red_num
    successB = 0
    for n in range(0,100):
        s = random.randint(0,101)
        p_red = round(prob(red_num, total))
        if s > p_red:
            successB += 1
    num_trails.append(successB)
       
plt.hist(num_trails)         
        
        
    
