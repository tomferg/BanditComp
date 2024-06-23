# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:10:56 2022

@author: Tom
"""

# %% Load Packages
import numpy as np
import os
import sys
import pandas as pd

#Load Data and mat files
#Set up parameters
numPart = 30
startCount_S = 0
matcount = 0
numArms = 2
numTrials = 20
numBlocks = 5
startArray_Size = 10

#Set up arrays
#Loading Data
all_files_S =  [None] * numPart

# adding folders to path.
sys.path.insert(0, './Helper Functions')
sys.path.insert(0, './Trial Classification')

#Assign Path
S_Path = './S_Data/'

#Stationary data
for file in os.listdir(S_Path):
    if file.endswith(".txt"):
        
        all_files_S[startCount_S] = os.path.join(S_Path, file)
        
        startCount_S += 1
#Sort Files
all_file_S = sorted(all_files_S)

#Define Split Function
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


# %% ####### Call Functions #######
# Load Likelihood Functions
import UCB_TC_2_S
import eGreedy_TC_S
import softMax_TC_S
import random_TC_S
import WSLS_TC_S
import gradient_TC_S
import KFTS_TC_S
import UCBSM_TC_S

# %% Trial Classification
#Set up Reward and Choice Arrays
explorationArray = np.zeros(shape=[8, numPart, numBlocks, numTrials])

#Parameter Generation
#Load Arrays
LLArray = np.genfromtxt('LLArray_S.csv', delimiter=',')
modeledParam = np.genfromtxt('modeledParam_S.csv', delimiter=',')


#Average Human data
for partCounter in range(numPart):
    
    print(partCounter)
    
    # File names
    fileName = all_files_S[partCounter]
    
    # Import data and assing to array
    data = np.loadtxt(fileName, delimiter="\t")

    # extract reward values
    rewardCrap_long = data[:,14]
    # extract choices
    choices_long = data[:,10]
    
    # Modify choices for functions
    choices_long[choices_long == 1] = 0
    choices_long[choices_long == 2] = 1
    
    # Reorganize Choice and Reward Arrays
    rewardCrap = np.array(list(split(rewardCrap_long,5)))
    choices =  np.array(list(split(choices_long,5)))

    # Concatinate choices and reward together
    arrayValues = np.zeros(shape=[2,numBlocks,numTrials])
    arrayValues[0,:,:] = rewardCrap
    arrayValues[1,:,:] = choices
    
    # Run Trial Classification for each Model
    # Random
    object = random_TC_S.trialClassification(arrayValues,
                                             modeledParam[partCounter, 0], numArms)  
    randomChoice = object.random()
    explorationArray[0,partCounter,:] = randomChoice
    
    # eGreedy
    object = eGreedy_TC_S.trialClassification(arrayValues,
                                              modeledParam[partCounter, 1], numArms)  
    eGreedy = object.eGreedy()
    explorationArray[1,partCounter,:] = eGreedy
    
    # Softmax
    object = softMax_TC_S.trialClassification(arrayValues,
                                              modeledParam[partCounter, 2], numArms)  
    softVal = object.softmax()
    explorationArray[2,partCounter,:] = softVal
    
    # Win-Stay, Lose-Shift
    object = WSLS_TC_S.trialClassification(arrayValues, 
                                           modeledParam[partCounter, 3], numArms)  
    WSLSchoice = object.WSLS_TC()
    explorationArray[3,partCounter,:] = WSLSchoice
    
    # UCB
    object = UCB_TC_2_S.trialClassification(arrayValues,
                                            modeledParam[partCounter, 4], numArms)  
    confBound = object.UCB1()
    explorationArray[4,partCounter,:] = confBound

    # Gradient
    object = gradient_TC_S.trialClassification(arrayValues,
                                               modeledParam[partCounter, 5], numArms)  
    grad = object.gradient()
    explorationArray[5,partCounter,:] = grad
    
    # KFTS
    object = KFTS_TC_S.trialClassification(arrayValues,
                                               modeledParam[partCounter, 6], numArms)  
    KFTS = object.KFTS()
    explorationArray[6,partCounter,:] = KFTS
    
    # UCBSM
    object = UCBSM_TC_S.trialClassification(arrayValues,
                                               modeledParam[partCounter, 7:9], numArms)  
    UCBSM = object.UCBSM()
    explorationArray[7,partCounter,:] = UCBSM
    


# %% Set Up Overlap Table
overlapTable = np.zeros(shape=[7, 7])

#Overlap Stuff
choiceEG = explorationArray[1, :, :, :]
choiceSM = explorationArray[2, :, :, :]
choiceWSLS = explorationArray[3, :, :, :]
choiceUCB = explorationArray[4, :, :, :]
choiceGrad = explorationArray[5, :, :, :]
choiceKFTS = explorationArray[6, :, :]
choiceUCBSM = explorationArray[7, :, :]

#Egreedy
overlapTable[0, 0] = 1
overlapTable[1, 0] = np.sum(choiceEG == choiceSM) / (numTrials*numPart*numBlocks)
overlapTable[2, 0] = np.sum(choiceEG == choiceWSLS) / (numTrials*numPart*numBlocks)
overlapTable[3, 0] = np.sum(choiceEG == choiceUCB) / (numTrials*numPart*numBlocks)
overlapTable[4, 0] = np.sum(choiceEG == choiceGrad) / (numTrials*numPart*numBlocks)
overlapTable[5, 0] = np.sum(choiceEG == choiceKFTS) / (numTrials*numPart*numBlocks)
overlapTable[6, 0] = np.sum(choiceEG == choiceUCBSM) / (numTrials*numPart*numBlocks)

#SM
overlapTable[1, 1] = 1
overlapTable[2, 1] = np.sum(choiceSM == choiceWSLS) / (numTrials*numPart*numBlocks)
overlapTable[3, 1] = np.sum(choiceSM == choiceUCB) / (numTrials*numPart*numBlocks)
overlapTable[4, 1] = np.sum(choiceSM == choiceGrad) / (numTrials*numPart*numBlocks)
overlapTable[5, 1] = np.sum(choiceSM == choiceKFTS) / (numTrials*numPart*numBlocks)
overlapTable[6, 1] = np.sum(choiceSM == choiceUCBSM) / (numTrials*numPart*numBlocks)

# WSLS
overlapTable[2, 2] = 1
overlapTable[3, 2] = np.sum(choiceWSLS == choiceUCB) / (numTrials*numPart*numBlocks)
overlapTable[4, 2] = np.sum(choiceWSLS == choiceGrad) / (numTrials*numPart*numBlocks)
overlapTable[5, 2] = np.sum(choiceWSLS == choiceKFTS) / (numTrials*numPart*numBlocks)
overlapTable[6, 2] = np.sum(choiceWSLS == choiceUCBSM) / (numTrials*numPart*numBlocks)

# UCB
overlapTable[3, 3] = 1
overlapTable[4, 3] = np.sum(choiceUCB == choiceGrad) / (numTrials*numPart*numBlocks)
overlapTable[5, 3] = np.sum(choiceUCB == choiceKFTS) / (numTrials*numPart*numBlocks)
overlapTable[6, 3] = np.sum(choiceUCB == choiceUCBSM) / (numTrials*numPart*numBlocks)

# Grad
overlapTable[4, 4] = 1
overlapTable[5, 4] = np.sum(choiceGrad == choiceKFTS) / (numTrials*numPart*numBlocks)
overlapTable[6, 4] = np.sum(choiceGrad == choiceUCBSM) / (numTrials*numPart*numBlocks)

# KFTS
overlapTable[5, 5] = 1
overlapTable[6, 5] = np.sum(choiceKFTS == choiceUCBSM) / (numTrials*numPart*numBlocks)

# Last
overlapTable[6, 6] = 1

#Save as numpy array
S_Overlap = np.save('./Arrays_S/S_overlapTable.npy',overlapTable)

# %% Compute Exploration Rate
exploreMean = np.zeros(shape=[numPart,8])

for modelCount in range(8):

    for partcount in range(numPart):    

        exploreMean[partcount,modelCount] = np.count_nonzero(explorationArray[modelCount,partcount,:,:] == 1) / (numTrials*numBlocks)


#Convert to Array
exploreMean_1 = np.zeros(shape=(30, 9))
exploreMean_1[:, 0] = np.linspace(1, 30, 30)
exploreMean_1[:, 1:9] = exploreMean 

#Convert to DF?
exploreDF_Wide = pd.DataFrame(exploreMean_1)
exploreDF_Wide.columns = ["Part", "Random", "eGreedy", "Softmax", "WSLS", "UCB1", "Gradient", "KFTS", "UCBSM"]

#Convert from wide to long?
exploreDF_Long = pd. melt(exploreDF_Wide, id_vars='Part', 
                          value_vars=["Random", "eGreedy", "Softmax", "WSLS", "UCB1", "Gradient", "KFTS", "UCBSM"])
exploreDF_Long['value'] = exploreDF_Long['value'].apply(lambda x: x*100)

exploreDF_Long.to_csv('./Arrays_S/S_EERate.csv') 


# %% Convert LL to BIC
BICArray = np.zeros(shape=[numPart,8])

k = [1,1,1,1,1,1,1,2] #Currently the same for all models
for count in range(8):
    BICArray[:,count] = (k[count]* np.log(numTrials*numBlocks)) + 2*(LLArray[:,count])
#BICArray = k * np.log(numPart) - 2*(-LLArray)

# Subtract baseline BIC
BICBase = 1 - (BICArray / BICArray[:,0,None])
BICBase = np.delete(BICBase,0,1)

#Convert to Array
BICMean_1 = np.zeros(shape=(30,8))
BICMean_1[:,0] = np.linspace(1,30,30)
BICMean_1[:,1:8] =  BICBase

#Convert to DF?
BICDF_Wide = pd.DataFrame(BICMean_1)
BICDF_Wide.columns = ["Part", "eGreedy", "Softmax", "WSLS", "UCB1", "Gradient","KFTS","UCBSM"]

#Convert from wide to long?
BICDF_Long = pd. melt(BICDF_Wide, id_vars='Part', 
                      value_vars=["eGreedy","Softmax", "WSLS", "UCB1", "Gradient","KFTS","UCBSM"])
# Save Data
BICDF_Long.to_csv('./Arrays_S/S_BIC.csv') 