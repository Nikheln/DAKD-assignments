# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:56:53 2016

@author: Niko
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sc
import itertools
import math
import scipy.constants as mconsts
import scipy.stats as stats

rows = []
keys = []

def init():
    with open('winequality-white.csv') as csvfile:
        reader = csv.DictReader(csvfile, ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], delimiter=";")
        rows.extend(reader)
    for dict in rows:
        for k, v in dict.items():
            dict[k] = float(v)
    keys.extend(list(rows[0].keys()))
    
    figwidth = 6.9
    params = {'backend': 'ps',
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'axes.linewidth': 0.5,
              'lines.linewidth': 0.3,
              'text.usetex': True,
              'figure.figsize': [figwidth, math.sqrt(2)*figwidth],
              'font.family': 'serif'
    }
    matplotlib.rcParams.update(params)

init()

#plt.plot([d['quality'] for d in rows], [d['sulphates'] for d in rows], 'ro')
#plt.show()

def drawScatterMatrix():
    plotwidth = 30
    #keys.remove('quality')
    fig, axes = plt.subplots(nrows=len(keys), ncols=len(keys), figsize=(plotwidth, plotwidth))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
        
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].scatter([d[keys[x]] for d in rows], [d[keys[y]] for d in rows], marker='o', c=[d['quality'] for d in rows])
            
    
    # Label the diagonal subplots...
    for i, label in enumerate(keys):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')
    
    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(len(keys)), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

def drawBoxPlots():
    ncols = 3
    fig, axes = plt.subplots(nrows=4, ncols=ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, k in enumerate(keys):
        sp = axes[int(i/ncols), i%ncols]
        sp.boxplot([r[k] for r in rows], flierprops={'marker':'o', 'markerfacecolor':'none', 'markersize':'2', 'linewidth':'0.5'})
        sp.set_title(k)
    plt.savefig('boxplots.pdf')

def calculateBinAmount(data, function):
    """
    Calculate the amount of bins that should be used for a histogram of the given data.
    
    @param data: The data to analyze as a list of numbers
    
    @param function: The function to be used. Possible values: 'sturges', 'scott', 'sqrt', 'freedman-diaconis'
    
    @return: The amount of bins to be used as a number
    """
    def hToK(data, h):
        # Convert a given bin width to bin amount
        return math.ceil((max(data) - min(data))/h)
    return {
            'sturges':
                math.ceil(math.log2(len(data))+1),
            'scott':
                hToK(data, (3.5*np.std(data))/math.pow(len(data), 1/3)),
            'sqrt':
                int(math.sqrt(len(data))),
            'freedman-diaconis':
                hToK(data, (2*np.subtract(*np.percentile(data, [75, 25]))/math.pow(len(data), 1/3)))
            }.get(function, 10)

def drawSingleHistograms(function):
    ncols = 3
    fig, axes = plt.subplots(nrows=4, ncols=ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, k in enumerate(keys):
        data = [r[k] for r in rows]
        sp = axes[int(i/ncols), i%ncols]
        print(np.percentile(data, [75, 25]), math.pow(len(data), 1/3))
        print(calculateBinAmount(data, function), len(data))
        sp.hist(data, calculateBinAmount(data, function), linewidth=0.5)
        sp.set_title(k)
    plt.savefig('histograms.pdf')
    
def drawHistograms():
    for k in keys:
        data = [r[k] for r in rows]
        plt.hist(data, min(20, max(max(data)-min(data),len(set(data)))))
        plt.title(k)
        plt.savefig('histograms/' + k + '.pdf')
        plt.clf()

        
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
    
availableFunctions = [pearson, spearman, kendall]
    
def correlationCoefficientsToLaTeX():
    with open('temp.tex', 'w') as f:
        with open("preamble") as pre:
            f.writelines(pre.readlines())
        
        for function in availableFunctions:
        
            ccs = function.calculateCC()
            f.write("\\section{Correlation coefficients using " + function.name + "}\n")
            f.write("\\begin{tabular}{l || *{12}{P{1.2cm}}}\n")
            
            f.write("& " + " & ".join(keys))
            f.write("\\\\\n\\hline ")
            for k1 in keys:
                f.write("\n" + k1 + " & ")
                strs = []
                for k2 in keys:
                    value = ccs[k1][k2]
                    if (abs(value) > 0.5):
                        strs.append("\\bftab " + ("%.4f" % value))
                    else:
                        strs.append("%.4f" % value)
                    
                f.write(" & ".join(strs))
                f.write(" \\\\\n")
            f.write("\\end{tabular}\n\n")
        
        
        f.write("\\end{document}")
        f.flush()
        f.close()