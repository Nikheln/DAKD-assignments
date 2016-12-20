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

def loadMatrix():
    mtx = [[row[key] for key in keys] for row in rows]
    stds = [np.std([row[i] for row in mtx]) for i in range(len(keys))]
    means = [np.mean([row[i] for row in mtx]) for i in range(len(keys))]
    mtx = [[(mtx[i][j]-means[j])/stds[j] for j in range(len(keys))] for i in range(len(rows))]
    
    return mtx, stds, means
    
init()
mtx, stds, means = loadMatrix()
loadMatrix()
qIndex = keys.index('quality')

def unZQuality(quality):
    return quality*stds[qIndex]+means[qIndex]

def getVectorWithoutQuality(index):
    return mtx[index][:qIndex]+mtx[index][qIndex+1:]
    
def dist(index1,index2):
    # Do not use the quality value for distance measurement
    return np.linalg.norm(np.array(getVectorWithoutQuality(index1))-np.array(getVectorWithoutQuality(index2)))
    
def getEstimate(inpt, neighbors):
    ### An ordinary average over the selected neighbours
    return np.average([n[qIndex] for n in neighbors])

distMatrix = None
def buildDistMatrix():
    global distMatrix
    distMatrix = [[dist(i,j) if j > i else 0 for j in range(len(rows))] for i in range(len(rows))]
    for i in range(len(mtx)):
        for j in range(i):
            distMatrix[i][j] = distMatrix[j][i]
    
def getKNeighborIndices(k, inptIndex, trainIndexSet):
    if distMatrix == None:
        return sorted(trainIndexSet, key=lambda row: dist(inptIndex, row))[:k]
    return sorted(trainIndexSet, key = lambda row: distMatrix[inptIndex][row])[:k]

qMin = 3
qMax = 9
qRange = qMax - qMin
def nFoldCV_KNN(n, k):
    l = len(rows)
    shf = random.sample(range(l), l)
    results = [[0 for col in range(qRange+1)] for row in range(qRange+1)]
    results2 = {}
    for i in range(qMin,qMax+1):
        results2[i] = []
    for i in range(n):
        low = int((i/n)*l)
        high = int(((i+1)/n)*l)
        testIndexSet = shf[low:high]
        trainIndexSet = shf[:low] + shf[high:]
        for testIndex in testIndexSet:
            kni = getKNeighborIndices(k, testIndex, trainIndexSet)
            kns = [mtx[i] for i in kni]
            estimate = unZQuality(getEstimate(mtx[testIndex], kns))
            real = unZQuality(mtx[testIndex][qIndex])
            results[int(round(real-qMin))][int(round(estimate-qMin))] += 1
            results2[int(round(real))].append(round(estimate,2))
    # results:matrix with real-approximation counts
    # results2:object with lists for each real quality with approximated values
    return results, results2
    
def leaveOneOut_KNN(k):
    return nFoldCV_KNN(len(rows), k)
    
def getMSEs_KNN(lower,upper):
    errors = {}
    for i in range(lower, upper+1):
        res = leaveOneOut_KNN(i)[1]
        sm = 0
        for key in res.keys():
            for j in res[key]:
                diff = float(key)-float(j)
                if diff < -0.5:
                    sm = sm + (diff+0.5)**2
                elif diff >= 0.5:
                    sm = sm + (diff-0.5)**2
        errors[i] = sm/len(rows)
    return errors
    
def dataToHeatmap(data):
    plt.pyplot.imshow(data, cmap=plt.pyplot.cm.jet, interpolation='nearest')
    plt.show()
    
def dataToScatter(data, k_used):
    keys, values = [],[]
    for key in data:
        for value in data[key]:
            keys.append(int(key)+np.random.normal(0,0.05))
            values.append(value+np.random.normal(0,0.05))
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111)
    for i in range(2,9):
        ax.add_patch(plt.patches.Rectangle((i+0.5,i+0.5), 1, 1, alpha=0.3, facecolor="#80FF00", edgecolor="none"))
    ax.scatter(keys, values, color='none', edgecolors='black', linewidth=0.2, s=0.5)
    ax.set_title('Cross-validation results with $k='+str(k_used)+'$')
    ax.set_xlabel('Real quality')
    ax.set_ylabel('Approximated quality using '+str(k_used)+'-NN')
    fig.savefig('cv_scatters/' + str(k_used) + '.png', bbox_inches='tight', dpi=300)
    
def cvResultsToPlot(lower, upper):
    errors = getMSEs_KNN(lower, upper)
    plt.pyplot.title('Cross-validation error with values of $k$ in range '+str(lower)+'--'+str(upper))
    plt.pyplot.xlabel('$k$ used')
    plt.pyplot.ylabel('Mean-square error')
    plt.pyplot.plot([key for key in errors],[errors[key] for key in errors])
    minkey = min(errors,key=errors.get)
    plt.pyplot.xticks(list(plt.pyplot.xticks()[0])+[minkey])
    plt.pyplot.savefig('mse_k'+str(lower)+'-'+str(upper)+'.png',dpi=300,bbox_inches='tight')
    
def cvResultsToHeatmap(k):
    data = leaveOneOut_KNN(k)[0]
    data2 = [list(x) for x in zip(*[[val/sum(row) for val in row] for row in data])]
    heatmap=plt.pyplot.pcolor([p+0.5 for p in range(2,10)], [p+0.5 for p in range(2,10)], data2, cmap=plt.pyplot.cm.Blues, vmin=0, vmax=1.0)
    for i in range(len(data)):
        for j in range(len(data[i])):
            plt.pyplot.text(i+3, j+3, data[i][j], horizontalalignment='center', verticalalignment='center', color='black')
    cbar = plt.pyplot.colorbar(heatmap)
    cbar.ax.set_ylabel('\% of all predictions in current real quality')
    heatmap.axes.set_xlim(2.5,9.5)
    heatmap.axes.set_ylim(2.5,9.5)
    heatmap.axes.set_xlabel('Real quality')
    heatmap.axes.set_ylabel('Predicted quality')
    plt.pyplot.title('Leave-one-out cross-validation results of 12-NN')
    plt.pyplot.savefig(str(k)+'-nn-heatmap.png', dpi=300, bbox_inches='tight')
    
#==============================================================================
# Regression
#==============================================================================
def rls(_X, _y, regparam):
    X = np.matrix(_X)
    y = np.matrix(_y).T
    d = X.shape[1]
    l = np.eye(d)
    A = np.dot(X.T, X) + regparam * l
    b = np.dot(X.T, y)
    w = np.linalg.solve(A, b)
    return w

def getRegressionEstimate(vec, w):
    return np.dot(vec, w)[0,0]
    
def nFoldCV_reg(n, regparam):
    l = len(mtx)
    shf = random.sample(mtx, l)
    
    results = [[0 for col in range(qRange+1)] for row in range(qRange+1)]
    results2 = {}
    for i in range(qMin,qMax+1):
        results2[i] = []
    for i in range(n):
        low = int((i/n)*l)
        high = int(((i+1)/n)*l)
        testSet = shf[low:high]
        trainSet = shf[:low] + shf[high:]
        w = rls([row[:qIndex]+row[qIndex+1:] for row in trainSet], [row[qIndex] for row in trainSet], regparam)
        
        for test in testSet:
            estimate = unZQuality(getRegressionEstimate(test[:qIndex]+test[qIndex+1:], w))
            real = unZQuality(test[qIndex])
            results[int(round(real-qMin))][int(round(estimate-qMin))] += 1
            results2[int(round(real))].append(round(estimate,2))
        
    return results, results2

def leaveOneOut_reg(regparam):
    return nFoldCV_reg(len(mtx), regparam)

def getConfusionRates(rng):
    confRates = {}
    for i in rng:
        res = nFoldCV_reg(50,i)[0]
        confRates[i] = 1-sum([res[j][j] for j in range(len(res))])/len(rows)
    return confRates
def getMSEs_reg(rng):
    errors = {}
    for i in rng:
        res = nFoldCV_reg(50, i)[1]
        sm = 0
        for key in res.keys():
            for j in res[key]:
                diff = float(key)-float(j)
                if diff < -0.5:
                    sm = sm + (diff+0.5)**2
                elif diff >= 0.5:
                    sm = sm + (diff-0.5)**2
        errors[i] = sm/len(rows)
    return errors
    
def dataToScatter_reg(data, lmbd_used):
    keys, values = [],[]
    for key in data:
        for value in data[key]:
            keys.append(int(key)+np.random.normal(0,0.05))
            values.append(value+np.random.normal(0,0.05))
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111)
    for i in range(2,9):
        ax.add_patch(plt.patches.Rectangle((i+0.5,i+0.5), 1, 1, alpha=0.3, facecolor="#80FF00", edgecolor="none"))
    ax.scatter(keys, values, color='none', edgecolors='black', linewidth=0.2, s=0.5)
    ax.set_title('Cross-validation results with $\\lambda='+str(lmbd_used)+'$')
    ax.set_xlabel('Real quality')
    ax.set_ylabel('Approximated quality using Ridge regression')
    fig.savefig('cv_scatters/reg_' + str(lmbd_used) + '.png', bbox_inches='tight', dpi=300)
    
def cvResultsToPlot_reg(rng):
    errors = getMSEs_reg(rng)
    plt.pyplot.title('Cross-validation error with values of $\\lambda$ in range '+str(min(rng))+'--'+str(max(rng)))
    plt.pyplot.xlabel('$\\lambda$ used')
    plt.pyplot.ylabel('Mean-square error')
    plt.pyplot.plot(rng,[errors[i] for i in rng])
    minkey = min(errors,key=errors.get)
    plt.pyplot.xticks(list(plt.pyplot.xticks()[0])+[minkey])
    plt.pyplot.savefig('mse_lmbd.png',dpi=300,bbox_inches='tight')
    
def cvResultsToHeatmap_reg(lmbd):
    data = leaveOneOut_reg(lmbd)[0]
    data2 = [list(x) for x in zip(*[[val/sum(row) for val in row] for row in data])]
    heatmap=plt.pyplot.pcolor([p+0.5 for p in range(2,10)], [p+0.5 for p in range(2,10)], data2, cmap=plt.pyplot.cm.Blues, vmin=0, vmax=1.0)
    for i in range(len(data)):
        for j in range(len(data[i])):
            plt.pyplot.text(i+3, j+3, data[i][j], horizontalalignment='center', verticalalignment='center', color='black')
    cbar = plt.pyplot.colorbar(heatmap)
    cbar.ax.set_ylabel('\% of all predictions in current real quality')
    heatmap.axes.set_xlim(2.5,9.5)
    heatmap.axes.set_ylim(2.5,9.5)
    heatmap.axes.set_xlabel('Real quality')
    heatmap.axes.set_ylabel('Predicted quality')
    plt.pyplot.title('Leave-one-out cross-validation results of Ridge regression with $\\lambda='+str(lmbd)+'$')
    plt.pyplot.savefig(str(lmbd)+'-reg-heatmap.png', dpi=300, bbox_inches='tight')
    
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