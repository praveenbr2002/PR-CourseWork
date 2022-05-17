import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import csv
from sympy import *
def disc_equn(c1,pw,test):
    u1 = np.mean(c1,axis=0)
    cov_mat1 = np.stack(c1,axis = 1)
    cov1 = np.cov(cov_mat1)
    inv1 = np.linalg.inv(cov1)
    tra = np.transpose(np.array(test - u1))
    p1 = np.dot(tra,inv1)
    p2 = np.dot(p1,np.array(test - u1))
    equation = (-0.5*p2) - (np.log(2*np.pi)) + np.log(pw)
    return equation
    
    
file = open("face feature vectors.csv","r")
records = []
csfile = csv.reader(file)
t1 = np.array([0])
for i in csfile:
    records.append(i)
new_record = []
for j in range(1,6):
    temp = []
    for i in range(0,128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
test_male = np.array(new_record)
new_record.clear()
for j in range(6,401):
    temp = []
    for i in range(0,128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
male = np.array(new_record)
new_record.clear()


for j in range(401,406):
    temp = []
    for i in range(0,128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
test_female = np.array(new_record)
new_record.clear()
for j in range(406,801):
    temp = []
    for i in range(0,128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
female = np.array(new_record)
new_record.clear()
m1 = 0
m2 = 0
print("Testing Male vectors...")
for i in range(0,5):
    score1 = disc_equn(male,0.5,test_male[i])
    score2 = disc_equn(female,0.5,test_male[i])
    if(score2>score1):
        print("The vector ",i + 1," Belongs to Female")
        m1 = m1 + 1
    else:
       print("The vector ",i + 1," Belongs to Male")
print("Testing Female Vectors")
for i in range(0,5):
    score1 = disc_equn(male,0.5,test_female[i])
    score2 = disc_equn(female,0.5,test_female[i])
    if(score1>score2):
        print("The vector ",i + 401," Belongs to Male")
        m2 = m2 + 1
    else:
        print("The vector ",i + 401," Belongs to Female")
        
        
accuracy1 = (1 - m1/10)*100
accuracy2 = (1 - m2/10)*100
print("The Accuracy for Class 1: ",accuracy1,"%")
print("The Accuracy for Class 2: ",accuracy2,"%")
print("The Accuracy of Bayes Classifier: ",(accuracy1+accuracy2)/2,"%")