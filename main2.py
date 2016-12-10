# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:41:24 2016

@author: Niko
"""
import csv
import numpy as np
import matplotlib as plt
import random
import math
import subprocess
import os

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
    plt.rcParams.update(params)

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
    return np.average([n[qIndex] for n in neighbors])

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
        estimate = round(getEstimate(testRow, getKNeighbors(k, testRow, trainSet)))
        if math.isclose(testRow[qIndex], estimate):
            correct += 1
    return correct/total

qMin = int(min([row[qIndex] for row in mtx]))
qMax = int(max([row[qIndex] for row in mtx]))
def nFoldCV_KNN(n, k):
    l = len(mtx)
    shf = random.sample(mtx, l)
    qRange = qMax - qMin
    results = [[0 for col in range(qRange+1)] for row in range(qRange+1)]
    results2 = {}
    for i in range(10):
        results2[str(i+1)] = []
    for i in range(n):
        low = int((i/n)*l)
        high = int(((i+1)/n)*l)
        testSet = shf[low:high]
        trainSet = shf[:low] + shf[high:]
        for test in testSet:
            estimate = getEstimate(test, getKNeighbors(k, test, trainSet))
            real = int(test[qIndex])
            results[real-qMin][int(round(estimate))-qMin] += 1
            results2[str(real)].append(round(estimate,2))
        
    return results, results2
    
def leaveOneOut_KNN(k):
    return nFoldCV_KNN(len(mtx), k)
    
def dataToHeatmap(data):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()
    
def dataToScatter(data, k_used):
    keys, values = [],[]
    for key in data:
        for value in data[key]:
            keys.append(int(key)+np.random.normal(0,0.1))
            values.append(value+np.random.normal(0,0.1))
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111)
    for i in range(2,9):
        ax.add_patch(plt.patches.Rectangle((i+0.5,i+0.5), 1, 1, alpha=0.3, facecolor="#80FF00", edgecolor="none"))
    ax.scatter(keys, values, color='none', edgecolors='black', linewidth=0.2, s=0.5)
    fig.savefig('cv_scatters/' + str(k_used) + '.png', bbox_inches='tight', dpi=300)
    
#==============================================================================
# Project LaTeXifying
#==============================================================================
    
def looKnnToLaTeX(f, k):
    results = leaveOneOut_KNN(k)
    
    f.write("\\begin{tabular}{r | *{" + str(qMax-qMin+1) + "}{c}}\n")
    f.write("& \\multicolumn{" + str(qMax-qMin+1) + "}{c}{Predicted class} \\\\\n")
    f.write("True class & " + "&".join([str(i) for i in range(qMin, qMax+1)]) + " \\\\\n")
    f.write("\\hline\n")
    for i in range(qMax-qMin+1):
        f.write(str(qMin+i) + " & " + "&".join([str(r) if j != i else "\\textbf{"+str(r)+"}" for j,r in enumerate(results[i])]) + " \\\\\n")
    f.write("\\end{tabular}\n")
genPdf = True
def LaTeXifyProjct():
    filename = 'temp2'
    with open(filename+'.tex', 'w') as f:
        with open("preamble2") as pre:
            f.writelines(pre.readlines())

        f.write("\\section{Leave-one-out cross validations of k-NN}\n\n")
        for i in range(5,51,5):
            f.write("\\begin{table}\n")
            f.write("\\centering\n")
            looKnnToLaTeX(f, i)
            f.write("\\caption{" + str(i) + "-NN cross validation}\n")
            f.write("\\end{table}\n")
        f.write("\\end{document}")
        f.flush()
        f.close()

    if genPdf:
        cmd = ['pdflatex', '-interaction', 'nonstopmode', '-output-directory', 'build/', filename+'.tex']
        proc = subprocess.Popen(cmd)
        proc.communicate()

        os.startfile('build\\'+filename+'.pdf')