# @Date    : 2022-02-14 04:39:31
# @Author  : Praveen B R (COE19B007)
# @Subject : PR
# @Question: Assignment 1 - Q3
#Compare two text files doc1.txt and doc2.txt using cosine distance.
import numpy as np
import math
from collections import Counter

def dotProduct(D1, D2): 
    dot_product = 0.0
    for key in D1:
        if key in D2:
            dot_product += (D1[key] * D2[key])
    return dot_product


def cos_angle(D1, D2): 
    numerator = dotProduct(D1, D2)
    denominator = math.sqrt(dotProduct(D1, D1)*dotProduct(D2, D2))
    return (numerator / denominator)

def main():
    list1=[]
    with open('data/doc1.txt','r') as f1:  
        for line in f1:
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in line:
                if ele in punc:
                    line = line.replace(ele, "")
            stop_words  = ['a','an','the','this','is','was','in','on','with','of','are','and','for','must','so','these','be']
            for word in line.split():
                if word not in stop_words:
                    list1.append(word) 

    list2=[]
    with open('data/doc2.txt','r') as f2:  
        for line in f2:
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in line:
                if ele in punc:
                    line = line.replace(ele, "")
            stop_words  = ['a','an','the','this','is','was','in','on','with','of','are','and','for','must','so','these','be']
            for word in line.split():
                if word not in stop_words:
                    list2.append(word) 
    for i in range(len(list1)):
        list1[i] = list1[i].lower()
    for j in range(len(list2)):
        list2[j] = list2[j].lower()

    sorted_l1 = Counter(list1)
    sorted_l2 = Counter(list2)
    cos_theta = cos_angle(sorted_l1, sorted_l2)
    print("Cosine distance = ",1 - cos_theta)



if __name__ == "__main__":
    main()
