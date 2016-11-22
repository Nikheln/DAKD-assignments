# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:56:53 2016

@author: nipehe
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
import scipy.stats as stats
import subprocess
import os

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
    
    figwidth = 7.3
    figheight = 3
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
              #'figure.figsize': [figwidth, figheight],
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
    
def correlationCoefficientsToLaTeX(f):
    f.write("\\section{Correlation coefficients using different functions}\n\n")
    
    for function in correlationCoefficientFunctions:
    
        ccs = function.calculateCC()
        f.write("\\subsection{Correlation coefficients using " + function.name + "}\n")
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
    

def boxplotToLaTeX(key, f):
    if replot:
        drawAttributeBoxplot(key)
    
    f.write("\\begin{figure}[H]\n")
    f.write("\\includegraphics[width=\\textwidth]{boxplots/" + key.replace(" ", "_") + ".pdf}\n")
    f.write("\\caption{Boxplot of attribute \\emph{" + key + "}}")
    f.write("\\end{figure}\n\n")

    
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
            
        correlationCoefficientsToLaTeX(f)
        
        f.write("\\end{document}")
        f.flush()
        f.close()
    
    if genPdf:
        cmd = ['pdflatex', '-interaction', 'nonstopmode', '-output-directory', 'build/', filename+'.tex']
        proc = subprocess.Popen(cmd)
        proc.communicate()
        
        os.startfile('build\\'+filename+'.pdf')