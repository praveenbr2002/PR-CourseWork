# @Date    : 2022-02-14 04:39:31
# @Author  : Praveen B R (COE19B007)
# @Subject : PR
# @Question: Assignment 1 - Q2

#Given (hq − ht)T = (0.5 0.5 -0.5 -0.25 -0.25)
#A = [
#    [1, 0.135, 0.195, 0.137, 0.157],
#    [0.135, 1, 0.2, 0.309, 0.143],
#    [0.195, 0.2, 1, 0.157, 0.122],
#    [0.137, 0.309, 0.157, 1, 0.195],
#    [0.157, 0.143, 0.122, 0.195, 1]
#]
#Find QFD
import numpy as np

def findQFD(p,A): #p--> Transpose of hq-ht      A--> Similarity matrix
    return(np.sqrt(np.dot(np.dot(p,A),p)))
    
def main():
    p = np.array([ 0.5, 0.5, -0.5, -0.25, -0.25]) # Transpose of hq-ht  1x5
    A = np.array([
                [1, 0.135, 0.195, 0.137, 0.157],  # Similarity matrix   5x5    
                [0.135, 1, 0.2, 0.309, 0.143],
                [0.195, 0.2, 1, 0.157, 0.122],
                [0.137, 0.309, 0.157, 1, 0.195],
                [0.157, 0.143, 0.122, 0.195, 1]
            ])
    print("\nhq-ht Transpose :\n",p)
    print("\nSimilarity matrix :\n",A)
    print('\nQFD :',findQFD(p,A))

if __name__ == "__main__":
    main()