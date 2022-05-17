import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import csv
from sympy import *

# discrminant equation calculator
def discrminant_equation(c1, pw, test):
    u1 = np.mean(c1, axis=0)
    cov_mat1 = np.stack(c1, axis=1)
    cov1 = np.cov(cov_mat1)
    inv1 = np.linalg.inv(cov1)
    tra = np.transpose(np.array(test - u1))
    p1 = np.dot(tra, inv1)
    p2 = np.dot(p1, np.array(test - u1))
    equation = (-0.5*p2) - (math.log(2*np.pi)) + math.log(pw)
    return equation


def oneD_Matrix_Multiplication(m1, s):
    result = np.zeros((s, s))
    for i in range(0, s):
        for j in range(0, s):
            result[i][j] = m1[i] * m1[j]
    return result

# Scatter within matrix
def Scatter_Within_Matrix(male,test_male):
    f_test = np.concatenate((male,test_male))
    u1 = np.mean(f_test,axis=0)
    # print(male_mean)
    temp1 = np.subtract(male,u1)
    temp2 = np.transpose(temp1)
    f = np.dot(temp1,temp2)
    return f


# Code starts here
file = open("data/face_vector.csv", "r")
records = []
csfile = csv.reader(file)
t1 = np.array([0])
for i in csfile:
    records.append(i)


# male data Extraction
new_record = []
for j in range(1, 11):
    temp = []
    for i in range(0, 128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
test_male = np.array(new_record)
new_record.clear()
for j in range(11, 401):
    temp = []
    for i in range(0, 128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
male = np.array(new_record)
new_record.clear()


# female data extraction
for j in range(401, 411):
    temp = []
    for i in range(0, 128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
test_female = np.array(new_record)
new_record.clear()
for j in range(411, 801):
    temp = []
    for i in range(0, 128):
        temp.append(float(records[j][i + 2]))
    new_record.append(temp)
female = np.array(new_record)
new_record.clear()

# LDA starts
# within class scatter matrix
Train_data = np.concatenate((male, female))
total_male = np.concatenate((test_male, male))
total_female = np.concatenate((test_female, female))


cov_mat1 = np.stack(total_male, axis=1)
s1 = np.cov(cov_mat1)
cov_mat2 = np.stack(total_female, axis=1)
s2 = np.cov(cov_mat2)

Sw = s1 + s2

# Scatter between class matrix


u1 = np.mean(total_male, axis=0)
u2 = np.mean(total_female, axis=0)
sub = np.array(u1 - u2)
Sb = oneD_Matrix_Multiplication(sub, 128)
# print(Sb)


# step 3 sw inverse . sb

swinverse = np.linalg.inv(Sw)
pro = np.dot(swinverse, Sb)

# step 4 finding eigen vectors & eigne values

eig_values,eig_vectors = np.linalg.eig(pro)

temp_sort = np.sort(eig_values)
sort_eigen = np.flipud(temp_sort)


for i in range(0,66):
    temp_index = np.where(eig_values == sort_eigen[i])
    # print(temp_index)
    colval = Train_data[:, temp_index[0][0]]
    new_record.append(colval)
# print(new_record)
temp_FT = np.array(new_record)
final_Train_temp1 = np.transpose(temp_FT)
final_Train_temp2 = np.copy(final_Train_temp1)

final_Train_male = final_Train_temp1[:390]
final_Train_female = final_Train_temp2[390:]
# print(shape(final_Train_male))
# print(shape(final_Train_female))
new_record.clear()

temp_record = []

for i in range(0,66):
    temp_index = np.where(eig_values == sort_eigen[i])
    # print(temp_index)
    colval1 = test_male[:, temp_index[0][0]]
    colval2 = test_female[:, temp_index[0][0]]
    new_record.append(colval1)
    temp_record.append(colval2)

temp = np.array(new_record)
male_data = np.transpose(temp)
temp = np.array(temp_record)
female_data = np.transpose(temp)

    

# Testing face in Principal componnet analysis

m1 = 0
m2 = 0
print("Testing Male vectors")
for i in range(0,10):
    score1 = discrminant_equation(final_Train_male,0.5,male_data[i])
    score2 = discrminant_equation(final_Train_female,0.5,male_data[i])
    if(score2>score1):
        print("The vector ",i + 1," Belongs to Female")
        m1 = m1 + 1
    else:
       print("The vector ",i + 1," Belongs to Male")
print("Testing Female Vectors")
for i in range(0,10):
    score1 = discrminant_equation(final_Train_male,0.5,female_data[i])
    score2 = discrminant_equation(final_Train_female,0.5,female_data[i])
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
CF = []
CF.append([(10-m1), m1])
CF.append([m2, (10-m2)])
print("Confusion Matrix:-")
print(str(CF[0][0])+" "+str(CF[0][1])+"\n"+str(CF[1][0])+" "+str(CF[1][1]))
