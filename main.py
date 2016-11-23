# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:56:53 2016

@author: nipehe
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats
import subprocess
import os
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

rows = []
keys = []
replot = False
genPdf = True

def init():
    with open('winequality-white.csv') as csvfile:
        reader = csv.DictReader(csvfile, ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], delimiter=";")
        rows.extend(reader)
    for dict in rows:
        for k, v in dict.items():
            dict[k] = float(v)
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

init()

#plt.plot([d['quality'] for d in rows], [d['sulphates'] for d in rows], 'ro')
#plt.show()

def drawScatterMatrix(**kwargs):
    plotwidth = 15
    fig, axes = plt.subplots(nrows=len(keys), ncols=len(keys), figsize=(plotwidth, plotwidth))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    filtering = kwargs.get('filterOutliers', False)
    means = {}
    stds = {}
    for i, key in enumerate(keys):
        data = [r[key] for r in rows]
        means[key] = np.mean(data)
        stds[key] = np.std(data)
    filterDistance = kwargs.get('filterLimit', 3)
    def included(value, key):
        return not filtering or abs(value - means[key]) < filterDistance * stds[key]
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            dataX = [d[keys[x]] for d in rows if included(d[keys[x]], keys[x]) and included(d[keys[y]], keys[y])]
            dataY = [d[keys[y]] for d in rows if included(d[keys[x]], keys[x]) and included(d[keys[y]], keys[y])]
            
            axes[x,y].scatter(dataX, dataY, marker='o', s=1)
            
    
    # Label the diagonal subplots...
    for i, label in enumerate(keys):
        axes[i,i].annotate(label.replace(" ", "\n"), (0.5, 0.5), size=15, xycoords='axes fraction',
                ha='center', va='center')
    
    if filtering:
        plt.savefig('scattermatrix_filtered.png', bbox_inches='tight', dpi=300)
    else:
        plt.savefig('scattermatrix.png', bbox_inches='tight', dpi=300)

def drawParallelPlot(key):
    dims = len(keys)
    fig, axes = plt.subplots(1, dims, sharey=False)
    plt.subplots_adjust(wspace=0)
    
    # Move the key used for coloring to the last position
    keylist = [k for k in keys if k != key]
    keylist.append(key)
    
    # Create the list of colors used for lines
    coloring = [r[key] for r in rows]
    coloring = [(i-min(coloring))/(max(coloring)-min(coloring)) for i in coloring]
    
    for i, axis in enumerate(axes.flat):
        axis.xaxis.set_visible(False)
        axis.yaxis.set_ticks_position('none')
        plt.setp(axis.get_yticklabels(), visible=False)
        axis.set_title(keylist[i].replace(" ", "\n"), x=0)
        if i+1 < dims:
            rside = [r[keylist[i]] for r in rows]
            rside = [(i-min(rside))/(max(rside)-min(rside)) for i in rside]
            lside = [r[keylist[i+1]] for r in rows]
            lside = [(i-min(lside))/(max(lside)-min(lside)) for i in lside]
            for i in range(len(rside)):
                axis.plot([1,0],[lside[i], rside[i]], color=plt.cm.jet(coloring[i]))
    plt.savefig('parallel_coords.png', bbox_inches='tight', dpi=300)
    plt.clf()
    
    
def drawBoxPlots():
    ncols = 4
    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, k in enumerate(keys):
        sp = axes[int(i/ncols), i%ncols]
        sp.boxplot([r[k] for r in rows], flierprops={'marker':'o', 'markerfacecolor':'none', 'markersize':'2', 'linewidth':'0.5'})
        sp.set_title(k)
        sp.xaxis.set_ticklabels([''])
    plt.savefig('boxplots.pdf', bbox_inches='tight')
    
def drawAttributeBoxplot(key):
    data = [r[key] for r in rows]
    plt.clf()
    plt.boxplot(data, vert=False,flierprops={'marker':'o', 'markerfacecolor':'none', 'markersize':'2', 'linewidth':'0.5'})
    plt.savefig('boxplots/' + key.replace(" ", "_") + ".pdf", bbox_inches='tight')

##### Functionalities for drawing histograms #####
class BinAmountFormula(object):
    def __init__(self, name, code):
        self.name = name
        self.code = code
    
    def calculateBinAmount(data):
        raise NotImplementedError("Not implemented!")

def hToK(data, h):
    # Convert a given bin width to bin amount
    return math.ceil((max(data) - min(data))/h)
        
sturges = BinAmountFormula("Sturges\' rule", "sturges")
def sturgesFormula(data):
    return math.ceil(math.log2(len(data))+1)
sturges.calculateBinAmount = sturgesFormula

scott = BinAmountFormula("Scott\'s rule", "scott")
def scottFormula(data):
    return hToK(data, (3.5*np.std(data))/len(data)**(1/3))
scott.calculateBinAmount = scottFormula

sqrtformula = BinAmountFormula("Square-root choice", "sqrt")
def sqrtFormula(data):
    return int(math.sqrt(len(data)))
sqrtformula.calculateBinAmount = sqrtFormula

fd = BinAmountFormula("Freedman-Diaconis\' choice", "freedman-diaconis")
def fdFormula(data):
    return hToK(data, (2*np.subtract(*np.percentile(data, [75, 25]))/math.pow(len(data), 1/3)))
fd.calculateBinAmount = fdFormula

binAmountFunctions = [sturges, scott, sqrtformula, fd]

def drawSingleHistograms(function, ncols, nrows):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, k in enumerate(keys):
        data = [r[k] for r in rows]
        binAmount = function.calculateBinAmount(data)
        sp = axes[int(i/ncols), i%ncols]
        sp.hist(data, bins=binAmount, linewidth=0.5)
        sp.set_title(k)
    plt.savefig('histograms/histograms_' + function.code + '.pdf', bbox_inches="tight")
    
def drawAttributeHistograms(key, **kwargs):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=len(binAmountFunctions))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    data = [r[key] for r in rows]
    if kwargs.get('filterOutliers', False):
        data = [x for x in data if abs(x - np.mean(data)) < kwargs.get('filterLimit', 3) * np.std(data)]
    
    for index, function in enumerate(binAmountFunctions):
        sp = axes[index]
        sp.hist(data, function.calculateBinAmount(data))
        sp.set_title(function.name)
    
    if kwargs.get('filterOutliers', False):
        plt.savefig('histograms/' + key.replace(" ", "_") + '_filtered.pdf')
    else:
        plt.savefig('histograms/' + key.replace(" ", "_") + '.pdf')

##### Functionalities for correlation coefficient calculations #####
class CCFormula(object):
    def __init__(self, name):
        self.name = name
    
    def calculateCC():
        raise NotImplementedError("Not implemented!")

pearson = CCFormula('Pearson\'s correlation coefficient')
def pearsonCC():
    return {k1:{k2:stats.pearsonr([r[k1] for r in rows], [r[k2] for r in rows])[0] for k2 in keys} for k1 in keys}
pearson.calculateCC = pearsonCC

spearman = CCFormula('Spearman\'s rho')
def spearmanCC():
    return {k1:{k2:stats.spearmanr([r[k1] for r in rows], [r[k2] for r in rows])[0] for k2 in keys} for k1 in keys}
spearman.calculateCC = spearmanCC

kendall = CCFormula('Kendall\'s tau')
def kendallCC():
    return {k1:{k2:stats.kendalltau([r[k1] for r in rows], [r[k2] for r in rows])[0] for k2 in keys} for k1 in keys}
kendall.calculateCC = kendallCC
    
correlationCoefficientFunctions = [pearson, spearman, kendall]

##### Principal component analysis #####
w=len(rows)
h=len(keys)

def getMeanVector():
    X = np.matrix([[r[k] for r in rows] for k in keys])
    return np.array([np.mean(X[i,:]) for i in range(h)]).reshape(h,1)

def getStandardizedData():
    X = np.matrix([[r[k] for r in rows] for k in keys])
    mean_vector = getMeanVector()
    std_vector = np.array([np.std(X[i,:]) for i in range(h)]).reshape(h,1)
    
    # Z-score standardization
    for i in range(h):
        for j in range(w):
            X[i,j] = (X[i,j]-mean_vector[i])/std_vector[i]
    
    return X
    
def principalComponentAnalysis(dimensions):
    
    X = getStandardizedData()
    mean_vector = getMeanVector()
    
    cov_matrix = np.zeros((h, h))
    for i in range(w):
        x = X[:,i].reshape(h,1)
        cov_matrix += (x - mean_vector).dot((x - mean_vector).T)
    cov_matrix[:,:] /= (w-1)
    
    eig_val_sc, eig_vec_sc = np.linalg.eig(cov_matrix)
    
    # Pair up eigenvalues and the corresponding eigenvectors
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(h)]
    
    # Sort to decreasing order of eigenvalue
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    transformation_matrix = np.matrix([ep[1] for ep in eig_pairs[0:dimensions]]).T
    
    return transformation_matrix

def PCAplot():
    tm = principalComponentAnalysis(2)
    data = getStandardizedData()
    
    projected_data = tm.T.dot(data)
    
    filtered_data = [[projected_data[x,y] for y in range(len(rows))
        if abs(projected_data[0,y]) < 6
        and abs(projected_data[1,y] < 6)] for x in range(2)]
    
    plt.scatter(filtered_data[0], filtered_data[1])

def correlationCoefficientsToLaTeX(f):
    f.write("\\section{Correlation coefficients using different functions}\n\n")
    
    for function in correlationCoefficientFunctions:
    
        ccs = function.calculateCC()
        f.write("\\subsection{Correlation coefficients using " + function.name + "}\n")
        f.write("\\begin{adjustbox}{max width=\\textwidth}\n")
        f.write("\\begin{tabular}{l || *{12}{P{1.2cm}}}\n")
        
        f.write("& " + " & ".join(keys))
        f.write("\\\\\n\\hline ")
        for k1 in keys:
            f.write("\n" + k1 + " & ")
            strs = []
            for k2 in keys:
                value = ccs[k1][k2]
                if (value < -0.5):
                    strs.append("\\bftab -" + ("%.4f" % value))
                elif (value < 0):
                    strs.append("-" + ("%.4f" % value))
                elif (value < 0.5):
                    strs.append("%.4f" % value)
                else:
                    strs.append("\\bftab " + ("%.4f" % value))
                    
            f.write(" & ".join(strs))
            f.write(" \\\\\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{adjustbox}\n")
    

def histogramsToLaTeX(key, f):
    if replot:
        drawAttributeHistograms(key)
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{histograms/" + key.replace(" ", "_") + ".pdf}\n")
    f.write("\\caption{Histograms of attribute \\emph{" + key + "} using different binning methods}")
    f.write("\\end{figure}\n\n")
    
def outlierFilteredHistogramsToLaTeX(key, f):
    if replot:
        drawAttributeHistograms(key, filterOutliers=True)
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{histograms/" + key.replace(" ", "_") + "_filtered.pdf}\n")
    f.write("\\caption{Histograms of attribute \\emph{" + key + "} with outliers further than 3 standard deviations from the mean filtered}\\n")
    f.write("\\end{figure}\n\n")
    
def boxplotToLaTeX(key, f):
    if replot:
        drawAttributeBoxplot(key)
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{boxplots/" + key.replace(" ", "_") + ".pdf}\n")
    f.write("\\caption{Boxplot of attribute \\emph{" + key + "}}")
    f.write("\\end{figure}\n\n")

def scatterMatrixToLaTeX(f):
    if replot:
        drawScatterMatrix()
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{scattermatrix.png}\n")
    f.write("\\caption{Scatter matrix of the whole feature set}\\n")
    f.write("\\end{figure}\n\n")

def outlierFilteredScatterMatrixToLaTeX(f):
    if replot:
        drawScatterMatrix(filterOutliers=True, filterLimit=3)
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{scattermatrix_filtered.png}\n")
    f.write("\\caption{Scatter matrix of the whole feature set with outliers further than 3 standard deviations from the mean filtered}\\n")
    f.write("\\end{figure}\n\n")

def parallelPlotToLaTeX(f):
    if replot:
        drawParallelPlot()
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{parallel_coords.png}\n")
    f.write("\\caption{Parallel coordinates representation of the data set}\\n")
    f.write("\\end{figure}\n\n")
    
        
def LaTeXifyProjct():
    filename = 'temp'
    with open(filename+'.tex', 'w') as f:
        with open("preamble") as pre:
            f.writelines(pre.readlines())
        
        f.write("\\section{Plots of single attributes}\n\n")
        for key in keys:
            f.write("\\subsection{" + key + "}\n")
            histogramsToLaTeX(key, f)
            outlierFilteredHistogramsToLaTeX(key, f)
            boxplotToLaTeX(key, f)
            f.write("\\newpage")
        
        
        f.write("\\section{Plots for the whole feature set}\n\n")
        scatterMatrixToLaTeX(f)
        outlierFilteredScatterMatrixToLaTeX(f)
        parallelPlotToLaTeX(f)
        
        correlationCoefficientsToLaTeX(f)
        
        f.write("\\end{document}")
        f.flush()
        f.close()
    
    if genPdf:
        cmd = ['pdflatex', '-interaction', 'nonstopmode', '-output-directory', 'build/', filename+'.tex']
        proc = subprocess.Popen(cmd)
        proc.communicate()
        
        os.startfile('build\\'+filename+'.pdf')