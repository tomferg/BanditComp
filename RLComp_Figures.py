# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:42:40 2022

@author: Tom Ferguson, PhD, University of Alberta
"""
# %% #######  Set Up and Packages ####### 
# Packages
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ptitprince as pt


# Load Data - Manuscript
# Mean Performance
meanPerf_NS = pd.read_csv('./Modelling_NS/Arrays_NS/NS_Mean_Model.csv')
meanPerf_S = pd.read_csv('./Modelling_S/Arrays_S/S_Mean_Model.csv')

meanPerf_NS_Comb= pd.read_csv('./Combined_Fitting/Arrays/Comb_Mean_BestFit_NS.csv')
meanPerf_S_Comb= pd.read_csv('./Combined_Fitting/Arrays/Comb_Mean_BestFit_S.csv')

totalPerf_NS = pd.read_csv('./Modelling_NS/Arrays_NS/Mean_BestFit_NS.csv')
totalPerf_S = pd.read_csv('./Modelling_S/Arrays_S/Mean_BestFit_S.csv')

# Trial By Trial
trialoptMean_NS = np.load('./Modelling_NS/Arrays_NS/NS_optMean.npy')
trialoptErr_NS = np.load('./Modelling_NS/Arrays_NS/NS_opt_Err.npy')
trialoptMean_S  = np.load('./Modelling_S/Arrays_S/S_optMean.npy')
trialoptErr_S = np.load('./Modelling_S/Arrays_S/S_opt_Err.npy')

trialoptMean_NS_Comb =  np.load('./Combined_Fitting/Arrays/NS_optMean_Comb.npy')
trialoptErr_NS_Comb =  np.load('./Combined_Fitting/Arrays/NS_opt_Err_Comb.npy')
trialoptMean_S_Comb =  np.load('./Combined_Fitting/Arrays/S_optMean_Comb.npy')
trialoptErr_S_Comb =  np.load('./Combined_Fitting/Arrays/S_opt_Err_Comb.npy')

# R^2
NS_R2 = pd.read_csv('./Modelling_NS/Arrays_NS/NS_BIC.csv')
S_R2 = pd.read_csv('./Modelling_S/Arrays_S/S_BIC.csv')
Comb_R2 = pd.read_csv('./Combined_Fitting/Arrays/NS_BIC_Comb.csv')

# Exploration Rate
# EERate_NS = pd.read_csv('./Modelling_NS/Arrays_NS/NS_EERate.csv')
# EERate_S = pd.read_csv('./Modelling_NS/Arrays_S/S_EERate.csv')
Overlap_NS = np.load('./Modelling_NS/Arrays_NS/NS_overlapTable.npy')
Overlap_S = np.load('./Modelling_S/Arrays_S/S_overlapTable.npy')

# Load Data - Supplimental Materials
testBed_NS = np.load('./Modelling_NS/Arrays_NS/testBed_NS.npy')
testBed_S = np.load('./Modelling_S/Arrays_S/testBed_S.npy')

paraRecovery_NS = np.genfromtxt('./Modelling_NS/Arrays_NS/paramRecovery_NS.csv', delimiter=",", 
                                skip_header=1)
paraRecovery_S = np.genfromtxt('./Modelling_S/Arrays_S/paramRecovery_S.csv', delimiter=",", 
                                skip_header=1)

# Values
numModels = 8

# %% Palette Set up
#palette = sns.color_palette("bright")
palette = sns.color_palette("Set2")

# Code to  drop first color
palette2 = palette
my_list = list(palette2)
my_list.pop(0)
palette2 = tuple(my_list)

my_list2 = list(palette2)
my_list2.pop(0)
palette3 = tuple(my_list2)

labels = ['Human', 'Bias', 'eGreedy', 'Softmax', 'WSLS', 'UCB-SW', 'Gradient', "KFTS",  "SM-UCB"]


palette4 = palette
my_list = list(palette4)
my_list.insert(0, [51/255, 50/255, 205/255])
palette4 = tuple(my_list)

# %% #######  Main Manuscript ####### 
# %% ## F2 - BIC - Non-Stationary ##

sns.set_style("white")
dx='variable'; dy='value'; ort="v"; sigma = .4;
plt.rcParams["figure.figsize"] = (4.5, 3)
sns.set(font_scale = .75)
sns.set_style("white")
pt.half_violinplot(x="variable", y="value", data = NS_R2, 
                                  palette = palette2, bw = sigma, cut = 0.,
                      scale = "area", width = 1, inner = None, orient = ort)
ax=sns.stripplot(x="variable", y="value", data = NS_R2, palette = palette2, edgecolor = "black",
                 size = 5, jitter = 1, zorder = 1, orient = ort, linewidth=1, alpha = .5)
ax.axhline(0,linewidth=2,linestyle='--', color="black")
ax = sns.pointplot(x="variable", y="value", data=NS_R2, color="black",
                    linewidth=.01, errorbar=(95),join=False, 
                    capsize=.2, scale = .5)
ax.set_ylim([-.25, 1])
#ax.set_xticklabels(lab_Names)
ax.set(xlabel=' ', ylabel='$R^2$')
# plt.savefig('./Figures/BIC_NS.png',dpi=800)


# %% ## F3 - BIC - Stationary ##

sns.set_style("white")
dx='variable'; dy='value'; ort="v"; sigma = .4;
plt.rcParams["figure.figsize"] = (5, 3.33)
sns.set(font_scale = .75)
sns.set_style("white")
pt.half_violinplot(x="variable", y="value", data = S_R2, 
                                  palette = palette2, bw = sigma, cut = 0.,
                      scale = "area", width = 1, inner = None, orient = ort)
ax=sns.stripplot(x="variable", y="value", data = S_R2, palette = palette2, edgecolor = "black",
                 size = 5, jitter = 1, zorder = 1, orient = ort, linewidth=1, alpha = .5)
ax.axhline(0,linewidth=2,linestyle='--', color="black")
ax = sns.pointplot(x="variable", y="value", data=S_R2, color="black",
                    linewidth=.01, errorbar=(95),join=False, 
                    capsize=.2, scale = .5)
ax.set_ylim([-1, 1])
#ax.set_xticklabels(lab_Names)
ax.set(xlabel=' ', ylabel='$R^2$')
# plt.savefig('./Figures/BIC_S.png',dpi=800)


# %% ## F4 - Performance - Mean - Non-Stationary ##
# Plot Color Set Up
sns.set_style("white")
dx='variable'; dy='value'; ort="v"; sigma = .4;
plt.rcParams["figure.figsize"] = (5, 3)
sns.set(font_scale = .75)
sns.set_style("white")
pt.half_violinplot(x="variable", y="value", data = meanPerf_NS, 
                                  palette = palette4, bw = sigma, cut = 0.,
                      scale = "area", width = 1, inner = None, orient = ort)
ax=sns.stripplot(x="variable", y="value", data = meanPerf_NS, palette = palette4, edgecolor = "black",
                 size = 5, jitter = 1, zorder = 1, orient = ort, linewidth=1, alpha = .5)
ax = sns.pointplot(x="variable", y="value", data=meanPerf_NS, color="black",
                   errorbar=("ci",95),join=False,capsize=.25,zorder=0,
                   scale =.5, )
ax.set_ylim([20, 80])
#ax.set_xticklabels(lab_Names)
ax.set(xlabel=' ', ylabel='Points Acquired')
# plt.savefig('./Figures/PerfMean_NS.png',dpi=800)


# %% ## F5 - Performance - Trial - Non-Stationary ##
# Figure A - Win-Stay
x = np.linspace(0,400,400)

# Win Stay Figure
fig, (axs) = plt.subplots(2,4, figsize=(10, 4),sharey=True)
counter = 0    
for i in range(2):
    for j in range(4):
        sns.set(font_scale = .75)
        sns.lineplot(ax = axs[i, j],data=trialoptMean_NS[0, :], color='black',linestyle='--')    
        axs[i, j].fill_between(x, trialoptMean_NS[0, 0:]-trialoptErr_NS[0, :], 
                                    trialoptMean_NS[0, 0:]+trialoptErr_NS[0, :],
                                    alpha = .5, color='black')
        mod = sns.lineplot(ax = axs[i, j], data=trialoptMean_NS[counter+1, :], 
                      estimator='mean', color=palette[counter])
        axs[i, j].fill_between(x, trialoptMean_NS[counter+1, :]-trialoptErr_NS[counter+1, :], 
                                    trialoptMean_NS[counter+1, :]+trialoptErr_NS[counter+1, :],
                                    alpha = .5, color=palette[counter])
        sns.set_style("white")
        axs[i, j].set_ylim([0, 100])
        #axs[plotCount].set_xlim([0, 19])
        if i == 1:
            axs[i, j].set(xlabel='Trial')            
        if counter == 0 or counter == 4:
            axs[i, j].set(ylabel='Optimal Arm (%)')
        axs[i, j].set_xticks([0, 99, 199, 299, 399], [1, 100, 200, 300, 400])
        # Update counter
        counter += 1    
        mod.annotate(labels[counter], xy=(28, 95), xycoords='axes points',
                size=6, ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none'))
       # if 
        
        #plt.setp(mod.get_legend().get_texts(), fontsize='5')
fig.tight_layout()
# plt.savefig('./Figures/PerfTrial_NS.png',dpi=800)


# %% ## F6 - Performance - Mean - Stationary ##

sns.set_style("white")
dx='variable'; dy='value'; ort="v"; sigma = .4;
plt.rcParams["figure.figsize"] = (5, 3)
sns.set(font_scale = .75)
sns.set_style("white")
pt.half_violinplot(x="variable", y="value", data = meanPerf_S, 
                                  palette = palette4, bw = sigma, cut = 0.,
                      scale = "area", width = 1, inner = None, orient = ort)
ax=sns.stripplot(x="variable", y="value", data = meanPerf_S, palette = palette4, edgecolor = "black",
                 size = 5, jitter = 1, zorder = 1, orient = ort, linewidth=1, alpha = .5)
ax = sns.pointplot(x="variable", y="value", data=meanPerf_S, color="black",
                    errorbar=("ci",20000),join=False,capsize=.25,zorder=0,
                   scale = .5)
ax.set_ylim([15, 75])
#ax.set_xticklabels(lab_Names)
ax.set(xlabel=' ', ylabel='Win Percentage')
# plt.savefig('./Figures/PerfMean_S.png',dpi=800)

# %% ## F7 - Performance - Trial - Stationary ##
lab_Names = ["Human", 'Bias', 'eGreedy', 'Softmax', 'WSLS', 'UCB1', 'Gradient', "KFTS", "SM-UCB"]

x = np.linspace(0, 20, 20)

# Win Stay Figure
fig, (axs) = plt.subplots(2,4, figsize=(10, 4),sharey=True)
counter = 0    
for i in range(2):
    for j in range(4):
        sns.set(font_scale = .75)
        sns.lineplot(ax = axs[i, j],data=trialoptMean_S[0, :], color='black',linestyle='--')    
        axs[i, j].fill_between(x, trialoptMean_S[0, 0:]-trialoptErr_S[0, :], 
                                    trialoptMean_S[0, 0:]+trialoptErr_S[0, :],
                                    alpha = .5, color='black')
        mod = sns.lineplot(ax = axs[i, j], data=trialoptMean_S[counter+1, :], 
                      estimator='mean', color=palette[counter])
        axs[i, j].fill_between(x, trialoptMean_S[counter+1, :]-trialoptErr_S[counter+1, :], 
                                    trialoptMean_S[counter+1, :]+trialoptErr_S[counter+1, :],
                                    alpha = .5, color=palette[counter])
        sns.set_style("white")
        axs[i, j].set_ylim([0, 100])
        #axs[plotCount].set_xlim([0, 19])
        if i == 1:
            axs[i, j].set(xlabel='Trial')            
        if counter == 0 or counter == 4:
            axs[i, j].set(ylabel='Optimal Arm (%)')
        axs[i, j].set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
        # Update counter
        counter += 1    
        mod.annotate(lab_Names[counter], xy=(28, 15), xycoords='axes points',
                size=6, ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none'))
        #plt.setp(mod.get_legend().get_texts(), fontsize='5')
fig.tight_layout()
# plt.savefig('./Figures/PerfTrial_S.png',dpi=800)


# %% ## F10 - Exploration Overlap - Non-Stationary ##
labs = ['eGreedy', 'Softmax', 'WSLS', 'UCB-SW', 'Grad', "KFTS", "SM-UCB"]

fig1, ax1= plt.subplots()
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 1})
data = Overlap_NS * 100
data = data.astype(int)
plt.imshow(data, cmap ='plasma', interpolation=None)
for (j,i),label in np.ndenumerate(data):
    ax1.text(i,j,label,ha='center',va='center', fontsize = 14)
plt.yticks([0, 1, 2, 3, 4, 5, 6],labels=labs, fontsize = 14)
plt.xticks([0, 1, 2, 3, 4, 5, 6],labels=labs, fontsize = 14)
ax1.grid(False)
# plt.savefig('./Figures/EEOver_NS.png',dpi=800)


# %% ## F11 - Exploration Overlap - Stationary ##
labs = ['eGreedy', 'Softmax', 'WSLS', 'UCB1', 'Grad', "KFTS", "SM-UCB"]
fig1, ax1= plt.subplots()
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 1})
data = Overlap_S * 100
data = data.astype(int)
plt.imshow(data, cmap ='plasma', interpolation=None)
for (j,i),label in np.ndenumerate(data):
    ax1.text(i,j,label,ha='center',va='center', fontsize = 15)
plt.yticks([0, 1, 2, 3, 4, 5, 6],labels=labs, fontsize = 14)
plt.xticks([0, 1, 2, 3, 4, 5, 6],labels=labs, fontsize = 14)
ax1.grid(False)
# plt.savefig('./Figures/EEOver_S.png',dpi=800)


# %% New Analysis Figures
# %% R2

sns.set_style("white")
dx='variable'; dy='value'; ort="v"; sigma = .4;
plt.rcParams["figure.figsize"] = (4.5, 3)
sns.set(font_scale = .75)
sns.set_style("white")
pt.half_violinplot(x="variable", y="value", data = Comb_R2, 
                                  palette = palette2, bw = sigma, cut = 0.,
                      scale = "area", width = 1, inner = None, orient = ort)
ax=sns.stripplot(x="variable", y="value", data = Comb_R2, palette = palette2, edgecolor = "black",
                 size = 5, jitter = 1, zorder = 1, orient = ort, linewidth=1, alpha = .5)
ax.axhline(0,linewidth=2,linestyle='--', color="black")
ax = sns.pointplot(x="variable", y="value", data=Comb_R2, color="black",
                    linewidth=.01, errorbar=(95),join=False, 
                    capsize=.2, scale = .5)
ax.set_ylim([-.5, 1])
#ax.set_xticklabels(lab_Names)
ax.set(xlabel=' ', ylabel='$R^2$')
# plt.savefig('./Figures/BIC_Combined.tiff',dpi=800)

# %% Learning Curves - Non-Stationary
x = np.linspace(0,400,400)

# Win Stay Figure
fig, (axs) = plt.subplots(2,4, figsize=(10, 4),sharey=True)
counter = 0    
for i in range(2):
    for j in range(4):
        sns.set(font_scale =1)
        sns.lineplot(ax = axs[i, j],data=trialoptMean_NS_Comb[0, :], color='black',linestyle='--')    
        axs[i, j].fill_between(x, trialoptMean_NS_Comb[0, 0:]-trialoptErr_NS_Comb[0, :], 
                                    trialoptMean_NS_Comb[0, 0:]+trialoptErr_NS_Comb[0, :],
                                    alpha = .5, color='black')
        mod = sns.lineplot(ax = axs[i, j], data=trialoptMean_NS_Comb[counter+1, :], 
                      estimator='mean', color=palette[counter])
        axs[i, j].fill_between(x, trialoptMean_NS_Comb[counter+1, :]-trialoptErr_NS_Comb[counter+1, :], 
                                    trialoptMean_NS_Comb[counter+1, :]+trialoptErr_NS_Comb[counter+1, :],
                                    alpha = .5, color=palette[counter])
        sns.set_style("white")
        axs[i, j].set_ylim([0, 100])
        #axs[plotCount].set_xlim([0, 19])
        if i == 1:
            axs[i, j].set(xlabel='Trial')            
        if counter == 0 or counter == 4:
            axs[i, j].set(ylabel='Optimal Arm (%)')
        axs[i, j].set_xticks([0, 99, 199, 299, 399], [1, 100, 200, 300, 400])
        # Update counter
        counter += 1    
        mod.annotate(labels[counter], xy=(28, 85), xycoords='axes points',
                size=10, ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none'))
        #plt.setp(mod.get_legend().get_texts(), fontsize='5')
fig.tight_layout()
# plt.savefig('./Figures/PerfTrial_NS_Combined.png',dpi=800)

# %% Learning Curves - Stationary
lab_Names = ["Human", 'Bias', 'eGreedy', 'Softmax', 'WSLS', 'UCB1', 'Gradient', "KFTS", "SM-UCB"]

x = np.linspace(0, 20, 20)

# Win Stay Figure
fig, (axs) = plt.subplots(2,4, figsize=(10, 4),sharey=True)
counter = 0    
for i in range(2):
    for j in range(4):
        sns.set(font_scale = 1)
        sns.lineplot(ax = axs[i, j],data=trialoptMean_S_Comb[0, :], color='black',linestyle='--')    
        axs[i, j].fill_between(x, trialoptMean_S_Comb[0, 0:]-trialoptErr_S_Comb[0, :], 
                                    trialoptMean_S_Comb[0, 0:]+trialoptErr_S_Comb[0, :],
                                    alpha = .5, color='black')
        mod = sns.lineplot(ax = axs[i, j], data=trialoptMean_S_Comb[counter+1, :], 
                      estimator='mean', color=palette[counter])
        axs[i, j].fill_between(x, trialoptMean_S_Comb[counter+1, :]-trialoptErr_S_Comb[counter+1, :], 
                                    trialoptMean_S_Comb[counter+1, :]+trialoptErr_S_Comb[counter+1, :],
                                    alpha = .5, color=palette[counter])
        sns.set_style("white")
        axs[i, j].set_ylim([0, 100])
        #axs[plotCount].set_xlim([0, 19])
        if i == 1:
            axs[i, j].set(xlabel='Trial')            
        if counter == 0 or counter == 4:
            axs[i, j].set(ylabel='Optimal Arm (%)')
        axs[i, j].set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
        # Update counter
        counter += 1    
        mod.annotate(labels[counter], xy=(28, 85), xycoords='axes points',
                size=10, ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none'))
        #plt.setp(mod.get_legend().get_texts(), fontsize='5')
fig.tight_layout()
# plt.savefig('./Figures/PerfTrial_S_Combined.png',dpi=800)

# %% Performance Figure
t = sns.color_palette("bright")

# Code to  drop first color
t2 = t
my_list = list(t2)
my_list.pop(0)
#my_list.pop(0)
t2 = tuple(my_list)

fig, axs = plt.subplots(2,1, figsize=(3, 5),sharey=False)
for pCt in range(2):
    if pCt == 0:
        perfDat = meanPerf_NS_Comb
    else:
        perfDat = meanPerf_S_Comb
    sns.set_style("white")
    dx='variable'; dy='value'; ort="v"; sigma = .4;
    #plt.rcParams["figure.figsize"] = (4.5, 3)
    sns.set(font_scale = .75)
    sns.set_style("white")
    pt.half_violinplot(x="variable", y="value", data = perfDat, 
                                      palette = t2, bw = sigma, cut = 0.,
                          scale = "area", width = 1, inner = None, orient = ort, ax =axs[pCt])
    ax=sns.stripplot(x="variable", y="value", data = perfDat, palette = t2, edgecolor = "black",
                     size = 5, jitter = 1, zorder = 1, orient = ort, linewidth=1, alpha = .7, ax =axs[pCt])
    ax = sns.pointplot(x="variable", y="value", data=perfDat, color="black",
                        errorbar=("ci",20000),join=False,capsize=.25,zorder=0,
                       scale = .5, ax =axs[pCt])
    ax.set_ylim([20, 80])
    #ax.set_xticklabels(lab_Names)
    if pCt == 0:
        axs[0].set(xlabel=' ', ylabel='Points Acquired')
    else:
        axs[1].set(xlabel=' ', ylabel='Win Percentage')
# plt.savefig('./Figures/BestFit_Perf_Comb.png',dpi=800)



# %% ####### Supplimental Materials ####### 
# %% ## SM1 - Test Bed - Non-Stationary ##
# Add Labels
labels_TB_NS = ['Bias', 'eGreedy', 'Softmax', 'WSLS', 'UCB-SW', 'Gradient', "KFTS", "Softmax Explore", 'human']

# Set up BS for labels
hTest = np.array([0, 2, 4])
vTest = np.array([4, 5])

# Start counter
counter = 0

# Plot Comp
fig, (axs) = plt.subplots(4,2, figsize=(5, 8),sharey=True)
counter = 0

#for plotCount in range(numModels):
for i in range(4):
    for j in range(2):
        #plt.subplot(4,2,plotCount+1)
        sns.set(font_scale = .75)
        sns.lineplot(ax = axs[i, j], data=testBed_NS[8, :], color='black',
                     linestyle='--') 
        mod = sns.lineplot(ax = axs[i, j], data=testBed_NS[counter, :], 
                      estimator='mean', color=palette[counter])
        sns.set_style("white")
        axs[i, j].set_ylim([0, 75])
        #axs[plotCount].set_xlim([0, 19])
        if counter == 6 or counter == 7:
            axs[i, j].set(xlabel='Trial')
        if counter == 0 or counter == 2 or counter == 4 or counter == 6:
            axs[i, j].set(ylabel='Optimal Arm (%)')
        axs[i, j].set_xticks([0, 99, 199, 299, 399], [1, 100, 200, 300, 400])
      
        mod.annotate(labels_TB_NS[counter], xy=(28, 10), xycoords='axes points',
                size=6, ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none'))
        # Update counter
        counter += 1  
    #plt.setp(mod.get_legend().get_texts(), fontsize='5')
fig.tight_layout()
# plt.savefig('./Figures/TestBed_NS.png',dpi=800)


# %% ## SM2 - Test Bed - Stationary ##
# Labels
labels_TB_S = ['Bias', 'eGreedy', 'Softmax', 'WSLS', 'UCB1', 'Gradient',"KFTS", "Softmax Explore",  'Human']

# Set up BS for labels
hTest = np.array([0, 2, 4])
vTest = np.array([4, 5])

# Start counter
counter = 0
# Plot Comp
fig, (axs) = plt.subplots(4,2, figsize=(5, 8),sharey=True)
counter = 0
for i in range(4):
    for j in range(2):
        sns.set(font_scale = .75)
        sns.lineplot(ax = axs[i, j], data=testBed_S[8, :], color='black',
                     linestyle='--') 
        mod = sns.lineplot(ax = axs[i, j], data=testBed_S[counter, :], 
                      estimator='mean', color=palette[counter])
        sns.set_style("white")
        axs[i, j].set_ylim([0, 100])
        #axs[plotCount].set_xlim([0, 19])
        if counter == 6 or counter == 7:
            axs[i, j].set(xlabel='Trial')
        if counter == 0 or counter == 2 or counter == 4 or counter == 6:
            axs[i, j].set(ylabel='Optimal Arm (%)') 
        axs[i, j].set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
        mod.annotate(labels_TB_S[counter], xy=(28, 34), xycoords='axes points',
                size=6, ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none'))
        # Update counter
        counter += 1  
    #plt.setp(mod.get_legend().get_texts(), fontsize='5')
fig.tight_layout()
# plt.savefig('./Figures/TestBed_S.png',dpi=800)


# %% SM2 - PR Non-Stationary ##
#Convert Data into DF
yLabelArray = ['Fit - Bias','Fit - Learning Rate','Fit - Epsilon','Fit - Temperature',
                'Fit - Learning Rate','Fit - WinStay','Fit - LoseShift','Fit - Window',
                'Fit - Innovation Variance','Fit - Error Variance','Fit - Temperature','Fit - Explore Value']
xLabelArray = ['Sim - Bias','Sim - Learning Rate','Sim - Epsilon',
                'Sim - Temperature','Sim - Learning Rate','Sim - WinStay',
                'Sim - LoseShift','Sim - Window', 
                'Sim - Innovation Variance','Sim - Error Variance',
                'Sim - Temperature','Sim - Explore Value']
titleArray = ['Random','eGreedy','eGreedy','Softmax','Softmax','WSLS','WSLS',
              'UCB-SW','KFTS','KFTS','Softmax - Explore','Softmax - Explore']
colors = []

t = sns.color_palette("Set2")

colors = [t[0],t[1],t[1],t[2],t[2],t[3],t[3],t[4],t[6],t[6],t[7],t[7]]

fig, axs = plt.subplots(7,2)
plt.subplots_adjust(hspace=0.5)
plt.rcParams["figure.figsize"] = [6, 12]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.family':'Times New Roman'})
count = 1
for plotCount in range(12):
    sns.set(font_scale = .75)
    if plotCount == 1 or plotCount == 8:
        count +=1
    plt.subplot(7,2,count)
    sns.regplot(paraRecovery_NS[:,plotCount+1],paraRecovery_NS[:,plotCount+15],
                    color=colors[plotCount],scatter_kws={'s':50,'alpha':.9,'edgecolor':'black'})
    sns.set_style("white")
    count+=1
    plt.xlabel(xLabelArray[plotCount])
    plt.ylabel(yLabelArray[plotCount])
    plt.title(titleArray[plotCount])    
# plt.savefig('./Figures/ParameterRecovery_NS.png',dpi=300)
#plt.show()

# %% SM3 - PR Stationary ##

#Convert Data into DF
yLabelArray = ['Fit - Bias','Fit - Epsilon','Fit - Temperature','Fit - WinStay',
               'Fit - Info Bonus','Fit - Step-Size', 'Fit - Innovation Variance',
               'Fit - Temperature','Fit - Explore Value']
xLabelArray = ['Sim - Bias','Sim - Epsilon','Sim - Temperature','Sim - WinStay',
               'Sim - Info Bonus','Sim - Step-Size','Sim - Innovation Variance',
               'Sim - Temperature','Sim - Explore Value']
titleArray = ['Bias','eGreedy','Softmax','WSLS','UCB-1','Gradient',"KFTS", "Softmax - Explore", "Softmax - Explore"]
colors = []

t = sns.color_palette("Set2")

colors = [t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[7]]

fig, axs = plt.subplots(2,3)
plt.subplots_adjust(hspace=0.5)
plt.rcParams["figure.figsize"] = [6, 8]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.family':'Times New Roman'})
#fig.suptitle('Parameter Recovery')
# for plotCount in range(6):
#     axs[plotCount].scatter(parameterSetSim[:,plotCount],modeledParam[:,plotCount])    
# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')
count = 1
for plotCount in range(9):
    sns.set(font_scale = .75)
    sns.set_style("white")
    plt.subplot(5,2,count)
    sns.regplot(paraRecovery_S[:,plotCount+1],paraRecovery_S[:,plotCount+10],
                    color=colors[plotCount],scatter_kws={'s':50,'alpha':.9,'edgecolor':'black'}) 
    count+=1
    if plotCount == 5:
        plt.ylim([0, 10])
    plt.xlabel(xLabelArray[plotCount])
    plt.ylabel(yLabelArray[plotCount])
    plt.title(titleArray[plotCount])    
# plt.savefig('./Figures/ParameterRecovery_S.png',dpi=300)
#plt.show()