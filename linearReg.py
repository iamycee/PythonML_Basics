from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')
#xs= np.array([1, 2, 3, 4, 5, 6], dtype= np.float64)
#ys= np.array([5, 4, 6, 5, 6, 7], dtype= np.float64)

#A function to create a fake dataset

def create_dataset(hm, variance, step=2, correlation=False):
    val=1
    ys=[]
    for i in range(hm):
        y= val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs= [i for i in range(len(ys))]
    return np.array(xs, dtype= np.float64), np.array(ys, dtype= np.float64)
        
def best_fit(xs, ys):
    m= (mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
    b= mean(ys)- m*mean(xs)
    return m, b
def sqError(ys_orig,  ys_line):
    return sum((ys_line-ys_orig)**2)  #lineDist - pointDist

def coeff_of_determination(ys_orig, ys_line):
    y_meanLine= [mean(ys_orig) for y in ys_orig]
    sq_err_regLine= sqError(ys_orig, ys_line)
    sq_err_yMean= sqError(ys_orig, y_meanLine)
    return 1- (sq_err_regLine/sq_err_yMean)


xs, ys= create_dataset(40, 40, 2, correlation= 'pos')

m, b= best_fit(xs, ys)

#for x in xs:
    #regLine.append((m*x)+b)
regLine= [m*x+b for x in xs]

rSqr= coeff_of_determination(ys, regLine)
print(rSqr)

plt.scatter(xs, ys, color='red')
plt.plot(xs, regLine, color='blue')
plt.show()


