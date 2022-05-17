# @Date    : 2022-02-14 04:52:12
# @Author  : Praveen B R (COE19B007)
# @Subject : PR
# @Question: Assignment 1 - Q4
#Classify flower 1, 51, and 101 from the Iris Dataset (.csv file) attached along with the assignment document
#into one of the three classes as given in dataset specification:
#Dataset Specifications:
#Total number of samples = 150
#Number of classes = 3 (Iris setosa, Iris virginica, and Iris versicolor)
#The number of samples in each class = 50
#Directions to classify:
#1. Use features PetalLengthCm and PetalWidthCm only for classification.
#2. Consider flowers 1,51 and 101 as test cases.
#3. Plot the distribution of rest 147 sample points along with their classes( differentiate classes with different colour). Consider PetalWidthCm along Y-axis and PetalLengthCm along X-axis.
#4. Capture the properties of the distribution and use suitable distance metrics to classify the flowers 1,51 and 101 into one of the classes.
#5. Print their class and plot the points on the previous plot with a marker differentiating the three points.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Dm(df,p): #to find Mahalanobis distance
    mean = np.transpose(df.mean().to_numpy())
    Z = np.add(df,-np.transpose(mean)).to_numpy()
    Ci = np.linalg.inv(np.divide(np.matmul(np.transpose(Z),Z),len(df)-1))  #Inverse of co variance matrix
    return(np.sqrt(np.matmul(np.matmul(np.transpose(np.add(p,-mean)),Ci),np.add(p,-mean)))) #mahalanobis distance

def classify_Dm(df,tc):
    dm = {"Iris-setosa":Dm(df[:49],tc), "Iris-versicolor":Dm(df[49:98],tc), "Iris-virginica":Dm(df[98:147],tc)}
    print("Petal Width :",tc["PetalWidthCm"],"cm\t\t","Petal Length :",tc["PetalLengthCm"],"cm")
    print("Mahalanobis Distance wrt Iris-setosa:",dm["Iris-setosa"])
    print("Mahalanobis Distance wrt Iris-versicolor:",dm["Iris-versicolor"])
    print("Mahalanobis Distance wrt Iris-virginica:",dm["Iris-virginica"])
    print('Classification by Mahalanobis Distance :',min(dm, key=dm.get),'\n\n')


def main():
    dataSet = pd.read_csv("data/Iris.csv",index_col="Id")
    dataSet.drop(["SepalLengthCm","SepalWidthCm"], axis=1, inplace=True)        #droping unnecessary columns from dataframe
    test_cases = [ dataSet.loc[1], dataSet.loc[51], dataSet.loc[101]]
    
    tdf = pd.concat(test_cases,axis=1).transpose().drop(["Species"],axis=1)     #Test case dataframe
    #print(tdf)    
    #print(dataSet.columns)
    #print(dataSet.head())
    dataSet.drop([1,51,101],inplace = True)                                     #remove test cases from sample data

    fig,ax1 = plt.subplots(figsize=(10,10))

    #plot
    ax1.set(title="Plot", xlabel="PetalLengthCm",ylabel="PetalWidthCm")
    
    #Plotting Samples cases
    ax1.scatter(dataSet.loc[dataSet["Species"]=="Iris-setosa"]["PetalLengthCm"],dataSet.loc[dataSet["Species"]=="Iris-setosa"]["PetalWidthCm"],c="red", s=6, label="Iris-setosa")
    ax1.scatter(dataSet.loc[dataSet["Species"]=="Iris-versicolor"]["PetalLengthCm"],dataSet.loc[dataSet["Species"]=="Iris-versicolor"]["PetalWidthCm"],c="green", s=6, label="Iris-versicolor")
    ax1.scatter(dataSet.loc[dataSet["Species"]=="Iris-virginica"]["PetalLengthCm"],dataSet.loc[dataSet["Species"]=="Iris-virginica"]["PetalWidthCm"],c="blue", s=6, label="Iris-virginica")
    
     #Classifying test cases
    classify_Dm(dataSet.drop(["Species"],axis=1),tdf.iloc[0])
    classify_Dm(dataSet.drop(["Species"],axis=1),tdf.iloc[1])
    classify_Dm(dataSet.drop(["Species"],axis=1),tdf.iloc[2])
    
    #Plotting test cases
    ax1.scatter(tdf["PetalLengthCm"],tdf["PetalWidthCm"],c="black",marker='x',label="Test Case")
    ax1.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()
