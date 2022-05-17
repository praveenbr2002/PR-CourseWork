# @Date    : 2022-02-14 04:39:31
# @Author  : Praveen B R (COE19B007)
# @Subject : PR
# @Question: Assignment 1 - Q1

#Calculate the distance between the two normalized histograms H1 and H2 using each of the following methods:
#(a) KL Distance
#(b) Bhattacharyya Distance
#H1 = [ 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04]
#H2 = [ 0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02]

import numpy as np

def findKL(p,q):
    return np.sum([np.multiply(p[i],np.log2(p[i]/q[i])) for i in range(len(p))])

def findBattacharya(p,q):
    return (-np.log(np.sum([np.multiply(np.sqrt(p[i]),np.sqrt(q[i])) for i in range(len(p))])))

def main():
    #h1 = np.array(list(map(float,input().split()))) #enter space separated number for normalized hist 1
    #h2 = np.array(list(map(float,input().split()))) #enter space separated number for normalized hist 2
    h1 = np.array([ 0.24, 0.20, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04])
    h2 = np.array([ 0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02])
    print('H1 : ',h1,'\n','H2 : ',h2,'\n','KL distance : ',findKL(h1,h2),sep='')
    print('Battacharya distance :',findBattacharya(h1,h2))

if __name__ == "__main__":
    main()

