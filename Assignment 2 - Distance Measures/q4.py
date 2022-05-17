import numpy as np
import matplotlib.pyplot as plt
import pandas as p
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
    z = Symbol('z')
    w = Symbol('w')
    tra = np.transpose(np.array(([x,y,z,w])) - u1)
    p1 = np.dot(tra,inv1)
    p2 = np.dot(p1,np.array(([x,y,z,w]) - u1))
    equation = -0.5 * p2 - np.log(2*np.pi) - 0.5*np.log(abs(det1)) + np.log(pw)
    return equation
    
    
def Classifier(test,eq1,eq2,eq3):
    p =[]
    p.append(eq1.subs([(x,test[0]),(y,test[1]),(z,test[2]),(w,test[3])]))
    p.append(eq2.subs([(x,test[0]),(y,test[1]),(z,test[2]),(w,test[3])]))
    p.append(eq3.subs([(x,test[0]),(y,test[1]),(z,test[2]),(w,test[3])]))
    return p.index(max(p))
    
    
file = open("Iris.csv","r")
records = []
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
w = Symbol('w')
csfile = csv.reader(file)
t1 = np.array([0])
for i in csfile:
    records.append(i)
new_record = []
for j in range(1,41):
    test = [float(records[j][1]),float(records[j][2]),float(records[j][3]),float(records[j][4])]
    new_record.append(test)
a1 = np.array(new_record)
new_record.clear()
for j in range(51,91):
    test = [float(records[j][1]),float(records[j][2]),float(records[j][3]),float(records[j][4])]
    new_record.append(test)
a2 = np.array(new_record)
new_record.clear()
for j in range(101,141):
    test = [float(records[j][1]),float(records[j][2]),float(records[j][3]),float(records[j][4])]
    new_record.append(test)
a3 = np.array(new_record)
test_record = []
eq1 = disc_equn(a1,0.33)
eq2 = disc_equn(a2,0.33)
eq3 = disc_equn(a3,0.33)
for j in range(41,51):
    test = [float(records[j][1]),float(records[j][2]),float(records[j][3]),float(records[j][4])]
    test_record.append(test)
test1 = np.array(test_record)
test_record.clear()
for j in range(91,101):
    test = [float(records[j][1]),float(records[j][2]),float(records[j][3]),float(records[j][4])]
    test_record.append(test)
test2 = np.array(test_record)
test_record.clear()
for j in range(141,151):
    test = [float(records[j][1]),float(records[j][2]),float(records[j][3]),float(records[j][4])]
    test_record.append(test)
test3 = np.array(test_record)


print("g1(x) = ",eq1,"\n")
print("g2(x) = ",eq2,"\n")
print("g3(x) = ",eq3,"\n")
m1 = 0
m2 = 0
m3 = 0


for j in range(0,10):
    if(Classifier(test1[j],eq1,eq2,eq3)!=0):
        m1 = m1 + 1
    if(Classifier(test2[j],eq1,eq2,eq3)!=1):
        m2 = m2 + 1
    if(Classifier(test3[j],eq1,eq2,eq3)!=2):
        m3 = m3 + 1
        
        
accuracy1 = (1 - m1/10)*100
accuracy2 = (1 - m2/10)*100
accuracy3 = (1 - m3/10)*100


print("The Accuracy for Class 1: ",accuracy1,"%")
print("The Accuracy for Class 2: ",accuracy2,"%")
print("The Accuracy for Class 3: ",accuracy3,"%")
print("The Accuracy of Bayes Classifier: ",(accuracy1+accuracy2+accuracy3)/3,"%")





