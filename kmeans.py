import numpy as np
import pylab as pl
import math
from sklearn.metrics import confusion_matrix, accuracy_score

#Regression program for k-means and fuzzy c-means
#by spencer Duncan

clusters = 10


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



def e(w, data, centroids):
    
    return

def m(w, data, centroids):
    for i in range(10):
        pred = w[:] == i
        cluster = data[pred]
        centroids[i] = np.mean(cluster, axis=0)
        print(centroids)
    pl.scatter(centroids[:,0], centroids[:,1],)
    pl.show()
    return

    

def main():
    filename = "cluster_dataset.txt"
    data = readdata(filename)
    print(data)
    datasize = data.shape[0]
    #used for fuzzy c means
    #w = np.random.rand(datasize, clusters)
    w = np.random.randint(clusters,size=datasize)
    centroids = np.zeros((clusters,2))
    m(w, data, centroids)
    print(w)
    error = 1
    while(error > .2):
        
        
        error -= .2


main()