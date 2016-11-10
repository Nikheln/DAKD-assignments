# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:56:53 2016

@author: Niko
"""
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools

rows = []

with open('winequality-white.csv') as csvfile:
    reader = csv.DictReader(csvfile, ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], delimiter=";")
    rows.extend(reader)

#plt.plot([d['quality'] for d in rows], [d['sulphates'] for d in rows], 'ro')
#plt.show()

plotwidth = 30
keys = list(rows[0].keys())
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