# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:55:06 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

# %% Load Packages
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# from scipy.optimize import minimize
# from scipy.io import loadmat

#Load Data and mat files
#Set up parameters
numPart = 30
startCount_S = 0
startCount_NS = 0
matcount = 0
startArray_Size = 10
numModels = 8

# Non Stationary Parameters
numArms_NS = 4
numTrials_NS = 400
# Stationary Parameters
numArms_S = 2
numTrials_S = 20
numBlocks = 5

# Task Parameters
taskParam = [numArms_NS, numTrials_NS, numArms_S,  numTrials_S, numBlocks]

#Set up arrays
#Loading Data
all_files_S =  [None] * numPart

#Loading Data
all_files_NS =  [None] * numPart
all_files_Mat = [None] * numPart

# adding folders to path - Environment variables are likely not ideal....
sys.path.insert(0, './Likelihood_CF/')

#Assign Path
NS_Path = 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/NS_Data/'
#Assign Path
S_Path = 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/S_Data/'

# Non Stationary data
for file in os.listdir(NS_Path):
    if file.endswith(".txt"):
        
        all_files_NS[startCount_NS] = os.path.join(NS_Path, file)
        startCount_NS += 1
    
    if file.endswith(".mat"):
        all_files_Mat[startCount_NS] = os.path.join(NS_Path, file)
#Sort Files
all_file_NS = sorted(all_files_NS)

#Stationary data
for file in os.listdir(S_Path):
    if file.endswith(".txt"):
        
        all_files_S[startCount_S] = os.path.join(S_Path, file)
        
        startCount_S += 1
#Sort Files
all_file_S = sorted(all_files_S)

# %% Load Functions
#Define Split Function for non-stationary data
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def loadDataFile(participantNum):
    
    #load data file
    #File names
    fileName_NS = all_files_NS[participantNum]
    #Import data and assing to array
    data_NS = np.loadtxt(fileName_NS, delimiter="\t")
    data_NS[:,3] = data_NS[:,3] - 1
    data_NS[:,4] = data_NS[:,4] / 100
    #Modify NaN's to -1 (for functions)
    data_NS[np.isnan(data_NS)] = -1
        
    # load data file stationary
    # File names
    fileName_S = all_files_S[participantNum]
    # Import data and assing to array
    data_S = np.loadtxt(fileName_S, delimiter="\t")
    # extract reward values
    rewardCrap_long = data_S[:,14]
    # extract choices
    choices_long = data_S[:,10]
    # Modify choices for functions
    choices_long[choices_long == 1] = 0
    choices_long[choices_long == 2] = 1


    # Reorganize Choice and Reward Arrays
    rewardCrap = np.array(list(split(rewardCrap_long, 5)))
    choices =  np.array(list(split(choices_long, 5)))

    #Set up Reward and Choice Arrays
    rewPeople = []
    # Insert Reward Values
    rewPeople.insert(0, data_NS[:, 4])
    rewPeople.insert(1, rewardCrap)

    # Set up choice Arrays
    choPeople = []
    # Insert Choices Values
    choPeople.insert(0, data_NS[:, 3])
    choPeople.insert(1, choices)
    
    return rewPeople, choPeople

def fitFunct(startParam, funcFit, fitType, partRew, partCho, taskParam, boundVal):
    
    if len(boundVal) == 1:
        
        #Find Function to optimize
        res_Output = minimize(funcFit, startParam,
                        args=(partRew,  partCho, taskParam),
                        method=fitType, bounds=(boundVal))
    
    else:
        
        #Find Function to optimize
        res_Output = minimize(funcFit, startParam,
                        args=(partRew,  partCho, taskParam),
                        method=fitType, bounds= boundVal)
    

    return res_Output.x, res_Output.fun

# Load Likelihood Functions
from Bias_Lik_Comb import Bias_Lik_Comb
from eGreedy_Lik_Comb import eGreedy_Lik_Comb
from softmax_Lik_Comb import softmax_Lik_Comb
from WSLS_Lik_Comb import WSLS_Lik_Comb
from UCB_Lik_Comb import UCB_Lik_Comb
from Gradient_Lik_Comb import Gradient_Lik_Comb
from KFTS_Lik_Comb import KFTS_Lik_Comb
from UCBSM_Lik_Comb import UCBSM_Lik_Comb

# Assing to Arrays
LL_Part = np.zeros(shape=[numPart, numModels])

modeledPara_Man = np.zeros(shape=[numPart, 14])

    
# %% Actuall Fitting?
from scipy.optimize import minimize

start_Size = 5

#Parameter Generation - start as list?
#Array is set up: 
modeledParam = np.zeros(shape=[numPart, 15])
LLArray = np.zeros(shape=[numPart, 8]) 

startLoc = np.zeros(shape=[start_Size, 15])

# Type of algo's to try
fitType1 = 'nelder-mead'
fitType2 = 'trust-constr'

for pCt in tqdm(range(numPart)):       #range(10, numPart)): #range(1): #

    
    rewards, choices = loadDataFile(pCt)
            
    #Min arrays
    min_LL = np.zeros(shape=[start_Size, 8])
    min_Para = np.zeros(shape=[start_Size, 15]) 
    
    # Starting Loc randomization
    startLoc[:, 0] = np.random.beta(1.1, 1.1, size = start_Size) # N/A - Random - bias#
    startLoc[:, 1] = np.random.beta(1.1, 1.1, size = start_Size) # egreedy - epsilon
    startLoc[:, 2] = np.random.beta(1.1, 1.1, size = start_Size) # egreedy - learning rate
    startLoc[:, 3] = np.random.gamma(1.2, 5, size = start_Size)  # Softmax - Temperature
    startLoc[:, 4] = np.random.beta(1.1, 1.1, size = start_Size) # softmax - learning rate 
    startLoc[:, 5] = np.random.beta(1.1, 1.1, size = start_Size) # Win-Stay - WSLS
    startLoc[:, 6] = np.random.beta(1.1, 1.1, size = start_Size) # Lose-Shift - WSLS
    startLoc[:, 7] = np.random.randint(1, 200 , size = start_Size) # UCB - Window
    startLoc[:, 8] = np.random.uniform(0.01, 5, size = start_Size) # UCB - Explore
    startLoc[:, 9] = np.random.beta(1.1, 1.1, size = start_Size) # Gradient - LR
    startLoc[:, 10] = np.random.beta(1.1, 1.1, size = start_Size) # Gradient - LR 2
    startLoc[:, 11] = np.random.uniform(0.1, 1000, size = start_Size) # KFTS - SigX
    startLoc[:, 12] =  np.random.uniform(0.1, 1000, size = start_Size)  # KFTS - SigEps
    startLoc[:, 13] = np.random.gamma(1.2, 5, size = start_Size) # UCBSM - temp
    startLoc[:, 14] = np.random.uniform(0.01, 2, size = start_Size) # UCBSM - expPara

    #Modify choices and rewards
    for sLc in range(start_Size):
        
        # Random
        x1 = startLoc[sLc, 0]
        boundVal1 = ((.0001, .95,),)
        [min_Para[sLc, 0], min_LL[sLc, 0]] = fitFunct(x1, Bias_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal1)
        
        # eGreedy
        x2 = [startLoc[sLc, 1], startLoc[sLc, 2]]
        boundVal2 = ((.0001, .99), (.001, .99))
        [min_Para[sLc, 1:3], min_LL[sLc, 1]] = fitFunct(x2, eGreedy_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal2)
        
        # softmax
        x3 = [startLoc[sLc, 3], startLoc[sLc, 4]]
        boundVal3 = ((.0001, 40), (.001, .99))
        [min_Para[sLc, 3:5], min_LL[sLc, 2]] = fitFunct(x3, softmax_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal3)
        
        # Winstay-Lose Shift
        x4 = [startLoc[sLc, 5], startLoc[sLc, 6]]
        boundVal4 = ((.0001, .99), (.001, .99))
        [min_Para[sLc, 5:7], min_LL[sLc, 3]] = fitFunct(x4, WSLS_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal4)
        
        # UCB
        x5 = [startLoc[sLc, 7], startLoc[sLc, 8]]
        boundVal5 = ((1, 200), (.0001, 10))
        [min_Para[sLc, 7:9], min_LL[sLc, 4]] = fitFunct(x5, UCB_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal5)
   
        # # Gradient
        x6 = [startLoc[sLc, 9], startLoc[sLc, 10]]
        boundVal6 = ((.0001, 10), (.001, 10))
        [min_Para[sLc, 9:11], min_LL[sLc, 5]] = fitFunct(x6, Gradient_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal6)
   
        # KFTS
        x7 = [startLoc[sLc, 11], startLoc[sLc, 12]]
        boundVal7 = ((.1, 1000), (.1, 1000))
        [min_Para[sLc, 11:13], min_LL[sLc, 6]] = fitFunct(x7, KFTS_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal7)
   
        # SM UCB
        x8 = [startLoc[sLc, 13], startLoc[sLc, 14]]
        boundVal8 = ((.00001, 40), (.001, 10))
        [min_Para[sLc, 13:15], min_LL[sLc, 7]] = fitFunct(x8, UCBSM_Lik_Comb, fitType2, rewards, choices, taskParam, boundVal8)

         
    min_LL[min_LL == -0] = 10000   
    ninf = float('-inf')
    min_LL[min_LL == ninf] = 10000    

    
    #Assign Mins to parameters data to parameters
    #Random   
    LLArray[pCt, 0] = min(min_LL[:, 0])
    loc = np.where(min_LL[:, 0] == min_LL[:, 0].min())
    modeledParam[pCt, 0] = min_Para[loc[0][0], 0]

    #E-greedy  
    LLArray[pCt, 1] = min(min_LL[:, 1])
    loc = np.where(min_LL[:, 1] == min_LL[:, 1].min())
    modeledParam[pCt, 1] = min_Para[loc[0][0], 1]
    modeledParam[pCt, 2] = min_Para[loc[0][0], 2]
    
    # #Softmax
    LLArray[pCt, 2] = min(min_LL[:, 2])
    loc = np.where(min_LL[:, 2] == min_LL[:, 2].min())
    modeledParam[pCt, 3] = min_Para[loc[0][0], 3]
    modeledParam[pCt, 4] = min_Para[loc[0][0], 4]
    
    # #WSLS
    LLArray[pCt, 3] = min(min_LL[:, 3])
    loc = np.where(min_LL[:, 3] == min_LL[:, 3].min())
    modeledParam[pCt, 5] = min_Para[loc[0][0], 5]
    modeledParam[pCt, 6] = min_Para[loc[0][0], 6]
    
    # # UCB
    LLArray[pCt, 4] = min(min_LL[:, 4])
    loc = np.where(min_LL[:, 4] == min_LL[:, 4].min())
    modeledParam[pCt, 7] = min_Para[loc[0][0], 7]
    modeledParam[pCt, 8] = min_Para[loc[0][0], 8]
    
    # # Gradient
    LLArray[pCt, 5] = min(min_LL[:, 5])
    loc = np.where(min_LL[:, 5] == min_LL[:, 5].min())
    modeledParam[pCt, 9] = min_Para[loc[0][0], 9]
    modeledParam[pCt, 10] = min_Para[loc[0][0], 10]
    
    # # KFTS
    LLArray[pCt, 6] = min(min_LL[:, 6])
    loc = np.where(min_LL[:, 6] == min_LL[:, 6].min())
    modeledParam[pCt, 11] = min_Para[loc[0][0], 11]
    modeledParam[pCt, 12] = min_Para[loc[0][0], 12]
    
    # # UCBSM
    LLArray[pCt, 7] = min(min_LL[:, 7])
    loc = np.where(min_LL[:, 7] == min_LL[:, 7].min())
    modeledParam[pCt, 13] = min_Para[loc[0][0], 13]
    modeledParam[pCt, 14] = min_Para[loc[0][0], 14]
            
# %%   Save LL Array and modeledParam Stuff as CSVs
np.savetxt('LLArray_Comb.csv', LLArray, delimiter=',', fmt='%2F')
np.savetxt('modeledParam_Comb.csv', modeledParam, delimiter=',', fmt='%2F')
    