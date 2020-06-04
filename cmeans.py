import numpy as np
import pylab as pl
import math
from sklearn.metrics import confusion_matrix, accuracy_score
#Regression program for fuzzy c-means
#by spencer Duncan

clusters = 4
rounds = 3
fuzzifier = 3

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

#update class membership weights
def m(w, data, centroids):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            numerator = np.linalg.norm(data[i,:]- centroids[j,:])
            denominator = 0
            wSum = 0
            for k in range(clusters):
                denominator = np.linalg.norm(data[i,:]- centroids[k,:])
                wSum += math.pow(numerator/denominator, (2/(fuzzifier-1)))
            w[i,j] = wSum
    print("weights:", w)
    return

#calc new centroids
def e(w, data, centroids):
    global clusters
    clusterRange = clusters
    for i in range(clusterRange):
        classweights = np.power(w[:,i], fuzzifier)
        centroids[i] = np.sum(np.multiply(classweights[:,np.newaxis], data), axis=0)
        centroids[i] = centroids[i]/np.sum(classweights)
    print("new centroids: ",centroids)
    return

def updateLabels(labels, w):
    newlabels = np.argmax(w, axis=1)
    print("updated labes:",newlabels)
    return newlabels


def display(labels, data, centroids):
    colors = labels/clusters
    pl.scatter(data[:,0], data[:,1],c=colors)
    pl.scatter(centroids[:,0], centroids[:,1], c='r')
    pl.show()

def calcError(labels, data, centroids):
    sum = 0
    for i in range(clusters):

        pred = labels[:] == i
        cluster = data[pred]

        diff = np.linalg.norm(cluster-centroids[i], axis=1)
        diff = np.square(diff)
        diff = np.sum(diff)
        sum += diff
    return sum

def regress(data, absCord):
    datasize = data.shape[0]
    centroids = np.random.uniform(low=-absCord, high=absCord, size=(clusters,2))
    w = np.random.rand(datasize, clusters)
    labels = np.zeros((datasize,1))
    pError = 0
    error = 0
    deltError = 100
    while(deltError > 1):
        m(w, data, centroids)
        e(w, data, centroids)
        labels = updateLabels(labels, w)
        error = calcError(labels, data, centroids)
        deltError = abs(pError-error)
        pError = error
        display(labels, data, centroids)
    display(labels, data, centroids)
    return labels, centroids, error

def main():
    filename = "cluster_dataset.txt"
    data = readdata(filename)
    absCord = max(abs(np.min(data[:,0])),abs(np.max(data[:,0])),abs(np.min(data[:,1])),abs(np.max(data[:,1])))
    bestLabels = 0
    bestCentroids = 0
    bestError = 50000
    for i in range(rounds):
        labels, centroids, error = regress(data,absCord)
        if error < bestError:
            bestLabels = labels
            bestCentroids = centroids
            bestError = error
    print("final")
    display(bestLabels, data, bestCentroids)
    
main()