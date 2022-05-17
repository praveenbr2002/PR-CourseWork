# @Date    : 2022-02-14 04:52:39
# @Author  : Praveen B R (COE19B007)
# @Subject : PR
# @Question: Assignment 1 - Q5
#Consider the following images. Obtain the histograms for each of the images. 
#Using a suitable distance measure, find the distance between the query image and reference images.

import cv2 
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    query = cv2.imread('data/queryimage.png',0)
    ref1 = cv2.imread('data/refimage1.png',0)
    ref2 = cv2.imread('data/refimage2.png',0)
    
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(15,5))

    ax1.set(title="Query image")
    ax2.set(title="Ref 1 image")
    ax3.set(title="Ref 2 image")
    ax1.hist(query.ravel(),256,[0,256])
    ax2.hist(ref1.ravel(),256,[0,256])
    ax3.hist(ref2.ravel(),256,[0,256])
    plt.show()
    # show the plotting graph of an image
    #print(query.shape)
    #print(ref1.shape)
    #print(ref2.shape)
    query_list = construct_feature(query)
    ref1_list  = construct_feature(ref1)
    ref2_list  = construct_feature(ref2)

    A = construct_similarity_matrix() 
    print("QFD of reference image 1 wrt query image :",quadratic_form_distance(ref1_list,query_list,A))
    print("QFD of reference image 2 wrt query image :",quadratic_form_distance(ref2_list,query_list,A))

def construct_similarity_matrix():
    c_max = 255
    A = np.zeros((256,256))
    
    for i in range(256):
        for j in range(256):
            if(i == j):
                A[i,j] = 1
            else:
                A[i,j] = 1-(abs(i-j)/c_max)
    return A

def quadratic_form_distance(x,y,A):
    x = np.array(x)
    y = np.array(y)
    
    p  = np.subtract(x,y)
    pt = np.transpose(p)
    
    a = np.dot(pt,A)
    d2 = np.dot(a,p)
    return math.sqrt(d2)

def construct_feature(img):
    hist = dict()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] not in hist.keys()):
                hist[img[i,j]] = 1 
            else:
                hist[img[i,j]] += 1
    lst = []
    for i in range(256):
        if(i not in hist.keys()):
            lst.append(0)
        else:
            lst.append(hist[i])
    return lst


if __name__ == "__main__":
    main()
