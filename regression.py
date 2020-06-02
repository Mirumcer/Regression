import numpy as np
import pylab as pl
import math
from sklearn.metrics import confusion_matrix, accuracy_score

#Regression program for k-means and fuzzy c-means
#by spencer Duncan

def readdata(filename):
    temparray = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('  ')
        temparray.append(line)
    data = np.asarray(temparray, float)
    return data

def 

def main():
    filename = "cluster_dataset.txt"
    data = readdata(filename)
    print(data)
    w = np.random.rand()

    pl.scatter(data[:5,0], data[:5,1])
    pl.show()

main()