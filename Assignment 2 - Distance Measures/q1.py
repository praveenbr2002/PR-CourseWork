import numpy as np
import matplotlib.pyplot as plt
from sympy import *

w1 = np.array([[1,6],[3,4],[3,8],[5,6]])       # inputs
w2 = np.array([[3,0],[1,-2],[3,-4],[5,-2]])

Pw1 = 0.5
Pw2 = 0.5

x = Symbol('x')
y = Symbol('y')

def Discriminant_function(w1,pw):        # function to find the discriminant equation for the given class
    u1 = np.mean(w1,axis=0) 
    cov_mat1 = np.stack(w1,axis = 1) 
    covariance_mat = np.cov(cov_mat1)
    inverse = np.linalg.inv(covariance_mat) 
    determinant = np.linalg.det(covariance_mat)
    transpose = np.transpose(np.array(([x,y]) - u1))
    temp1 = np.dot(transpose,inverse)
    product = np.dot(temp1,np.array(([x,y]) - u1))
    eqn = -0.5 * product - np.log(2*np.pi) - 0.5*np.log(abs(determinant)) + np.log(pw)
    return eqn

g_1 = Discriminant_function(w1,Pw1)
g_2 = Discriminant_function(w2,Pw2)

print("g1(x) = ",g_1)
print("g2(x) = ",g_2)

x1 = [each[0] for each in w1]
y1 = [each[1] for each in w1]

x2 = [each[0] for each in w2]
y2 = [each[1] for each in w2]

gx = solve(g_1 - g_2,y,dict=True)                                     #Plotting of Decision Boundary
xlist = np.linspace(-20,10,num=500)
ylist = [lambdify(x,gx[0][y])(each) for each in xlist]
plt.plot(xlist,np.transpose(ylist),label = "Decision_Boundary")

plt.scatter(x1,y1,label = "Class_w1",color="violet",marker="+")           #Plotting of points
plt.scatter(x2,y2,label = "Class_w2",color ="green",marker="+")
plt.legend()
plt.show()