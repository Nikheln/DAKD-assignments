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
import itertools
import math
import scipy.constants as mconsts

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

def drawSingleHistograms():
    ncols = 3
    fig, axes = plt.subplots(nrows=4, ncols=ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, k in enumerate(keys):
        data = [r[k] for r in rows]
        sp = axes[int(i/ncols), i%ncols]
        sp.hist(data, min(20, max(max(data)-min(data),len(set(data)))))
        sp.set_title(k)
    plt.savefig('histograms.pdf')
    
def drawHistograms():
    for k in keys:
        data = [r[k] for r in rows]
        plt.hist(data, min(20, max(max(data)-min(data),len(set(data)))))
        plt.title(k)
        plt.savefig('histograms/' + k + '.pdf')
        plt.clf()