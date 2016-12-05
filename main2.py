# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:41:24 2016

@author: Niko
"""
import csv
import numpy as np
import scipy as sc
import matplotlib
import random
import math

rows = []
keys = []
def init():
    with open('winequality-white.csv') as csvfile:
        reader = csv.DictReader(csvfile, ["fixed acidity","volatile acidity",
                                          "citric acid","residual sugar",
                                          "chlorides","free sulfur dioxide",
                                          "total sulfur dioxide","density",
                                          "pH","sulphates","alcohol","quality"
                                          ], delimiter=";")
        rows.extend(reader)
    for dct in rows:
        for k, v in dct.items():
            dct[k] = float(v)
    keys.extend(list(rows[0].keys()))
    
    params = {'backend': 'ps',
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'axes.linewidth': 0.5,
              'boxplot.boxprops.linewidth': 0.5,
              'boxplot.medianprops.linewidth': 0.5,
              'boxplot.meanprops.linewidth': 0.5,
              'boxplot.flierprops.linewidth': 0.5,
              'boxplot.whiskerprops.linewidth': 0.5,
              'boxplot.capprops.linewidth': 0.5,
              'lines.linewidth': 0.5,
              'text.usetex': True,
              'font.family': 'serif'
    }
    matplotlib.rcParams.update(params)

def toMatrix(rows):
    return [[each[var] for var in each] for each in rows]
    
init()
mtx = toMatrix(rows)
qIndex = keys.index('quality')

def dist(v1,v2):
    # Do not use the quality value for distance measurement
    return np.linalg.norm(np.array(v1[:qIndex]+v1[qIndex+1:])-np.array(v2[:qIndex]+v2[qIndex+1:]))
    
def getEstimate(inpt, neighbors):
    ### An ordinary average over the selected neighbours
    return round(np.average([n[qIndex] for n in neighbors]))

def getKNeighbors(k, inpt, trainSet):
    return sorted(trainSet, key=lambda row: dist(inpt, row))[:k]
    
def getDatasets(ratio):
    trainSet, testSet = [], []
    
    for i, row in enumerate(mtx):
        if random.uniform(0.0,1.0) < ratio:
            testSet.append(row)
        else:
            trainSet.append(row)
            
    return trainSet, testSet
        
def testKNN(k, ratio):
    trainSet, testSet = getDatasets(ratio)
    correct = 0
    total = len(testSet)
    
    for testRow in testSet:
        estimate = getEstimate(testRow, getKNeighbors(k, testRow, trainSet))
        if math.isclose(testRow[qIndex], estimate):
            correct += 1
    return correct/total
    
def nFoldCV_KNN(n, k):
    l = len(mtx)
    shf = random.sample(mtx, l)
    results = []
    for i in range(n):
        low = int((i/n)*l)
        high = int(((i+1)/n)*l)
        testSet = shf[low:high]
        trainSet = shf[:low] + shf[high:]
        totalError = 0
        for test in testSet:
            estimate = getEstimate(test, getKNeighbors(k, test, trainSet))
            totalError += abs(test[qIndex] - estimate)
        results.append(totalError*n/l)
        
    return results