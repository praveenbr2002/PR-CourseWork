from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sympy import *


def disc_equn(c1,pw):
    u1 = np.mean(c1,axis=0)
    cov_mat1 = np.stack(c1,axis = 1)
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
    
    
def ylistf(xlist,temp):
    return [lambdify(x,temp)(val) for val in xlist]
    
    
file = open("Iris.csv","r")
records = []
x = Symbol('x')
y = Symbol('y')
csfile = csv.reader(file)
t1 = np.array([0])

for i in csfile:
    records.append(i)
new_record = []
for j in range(1,41):
    test = [float(records[j][3]),float(records[j][4])]
    new_record.append(test)
a1 = np.array(new_record)
new_record.clear()
for j in range(51,91):
    test = [float(records[j][3]),float(records[j][4])]
    new_record.append(test)
a2 = np.array(new_record)
new_record.clear()
for j in range(101,141):
    test = [float(records[j][3]),float(records[j][4])]
    new_record.append(test)
a3 = np.array(new_record)

test_x = []
test_y = []


for j in range(41,51):
    test_x.append(float(records[j][3]))
    test_y.append(float(records[j][4]))
for j in range(91,101):
    test_x.append(float(records[j][3]))
    test_y.append(float(records[j][4]))
for j in range(141,151):
    test_x.append(float(records[j][3]))
    test_y.append(float(records[j][4]))


equn_1 = disc_equn(a1,0.33)
equn_2 = disc_equn(a2,0.33)
equn_3 = disc_equn(a3,0.33)
print("g1(x) = ",equn_1)
print("g2(x) = ",equn_2)
print("g3(x) = ",equn_3)
x = Symbol('x')
y = Symbol('y')


# plot g12
f = solve(equn_1 - equn_2,y,dict=True)
xlist = np.linspace(-0.51, 2.42,num = 1000)
temp = f[0][y]
ylist = ylistf(xlist,temp)
plt.plot(xlist,np.transpose(ylist),color="green",label ="g12")
temp = f[1][y]
ylist = ylistf(xlist,temp)
plt.plot(xlist,np.transpose(ylist),color ="green")


f = solve(equn_1 - equn_3,y,dict=True)
xlist = np.linspace(-0.5, 2.67, 1000)
temp = f[0][y]
ylist = ylistf(xlist,temp)
plt.plot(xlist,np.transpose(ylist),color="blue",label="g13")
temp = f[1][y]
ylist = ylistf(xlist,temp)
plt.plot(xlist,np.transpose(ylist),color="blue")


# plot g23
f = solve(equn_2 - equn_3,y,dict=True)
xlist = np.linspace(-5.15, 5.18, 1000)
temp = f[0][y]
ylist = ylistf(xlist,temp)
plt.plot(xlist,np.transpose(ylist),color = "red",label = "g23")
temp = f[1][y]
ylist = ylistf(xlist,temp)
plt.plot(xlist,np.transpose(ylist),color = "red")

x1 = [p[0] for p in a1]
y1 = [p[1] for p in a1]
x2 = [p[0] for p in a2]
y2 = [p[1] for p in a2]
x3 = [p[0] for p in a3]
y3 = [p[1] for p in a3]

plt.scatter(x1,y1,color="red",label="Setosa",marker="*")
plt.scatter(x2,y2,color="green",label="Versicolor",marker="^")
plt.scatter(x3,y3,color="violet",label="Virginicia",marker="+")
plt.scatter(test_x,test_y,color = "black",marker=".",label="test-cases")
plt.legend()

plt.show()

