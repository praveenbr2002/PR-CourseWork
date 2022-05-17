import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from sympy import *

def disc_eqtn(omg1,pw):
    u1 = np.mean(omg1,axis=0)  
    cov_mat1 = np.stack(omg1,axis = 1)
    cov1 = np.cov(cov_mat1)
    inv1 = np.linalg.inv(cov1)   
    det1 = np.linalg.det(cov1)
    x = Symbol('x')
    y = Symbol('y')
    tra = np.transpose(np.array(([x,y])) - u1)
    p1 = np.dot(tra,inv1)
    p2 = np.dot(p1,np.array(([x,y]) - u1))
    equation = -0.5 * p2 - np.log(2*np.pi) - 0.5*np.log(abs(det1)) + np.log(pw)
    return equation
      
def ylist(xlist,temp):
    return [lambdify(Symbol('x'),temp)(val) for val in xlist]    

omg1 = np.array([[1,-1],[2,-5],[3,-6],[4,-10],[5,-12],[6,-15]])
omg2 = np.array([[-1,1],[-2,5],[-3,6],[-4,10],[-5,12],[-6,15]])
equn_1 = disc_eqtn(omg1,0.3)
equn_2 = disc_eqtn(omg2,0.7)

print("g1(x) = ",equn_1)
print("g2(x) = ",equn_2)

#x = Symbol('x')
#y = Symbol('y')

f = solve(equn_1 - equn_2,Symbol('y'),dict=True)
xlist = np.linspace(-20,20,num=1000)
temp = f[0][Symbol('y')]
ylist = ylist(xlist,temp)
plt.plot(xlist,np.transpose(ylist),label = "Decision_Boundary")

Xω1 = [p[0] for p in omg1]
Xω2 = [p[0] for p in omg2]
Yω1 = [p[1] for p in omg1]
Yω2 = [p[1] for p in omg2]
plt.scatter(Xω1,Yω1,label = "Class-ω1",color="blue")
plt.scatter(Xω2,Yω2,label = "Class-ω2",color ="red")
plt.legend(loc="upper right")
plt.show()

