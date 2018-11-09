##hw3 function
import sys
import numpy as np
import pandas as pd 
import keras
def data_process_init(data,label=True):
    X = data['feature']
    
    if label:
        Y = data['label']
        return X,Y
    else:
        return X

def feat_init (data,row=48,col=48):
    print(1)
    X = []
    for i in range (len(data)):
        X.append(data[i].split())
    newX = np.zeros((len(data),row,col))
    for i in range (len(data)):
        index = 0
        for j in range (row):
            for k in range (col):
                newX[i][j][k]=X[i][index]
                index+=1
    return newX
