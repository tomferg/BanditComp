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
startCount_NS = 0
matcount = 0
numArms = 4
numTrials = 400

#Set up arrays
#Loading Data
all_files_NS =  [None] * numPart
all_files_Mat = [None] * numPart

# adding folders to path - Environment variables are likely not ideal....
sys.path.insert(0, './Helper Functions/')
sys.path.insert(0, './Trial Classification/')

#Assign Path
NS_Path = 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/NS_Data/'

#Stationary data
for file in os.listdir(NS_Path):
    if file.endswith(".txt"):
        
        all_files_NS[startCount_NS] = os.path.join(NS_Path, file)
        startCount_NS += 1
    
    if file.endswith(".mat"):
        all_files_Mat[startCount_NS] = os.path.join(NS_Path, file)

#Sort Files
all_file_NS = sorted(all_files_NS)



# %% ####### Call Functions #######
import UCB_SW_TC_NS
import eGreedy_TC_NS
import softMax_TC_NS
import random_TC_NS
import WSLS_TC_NS
import gradient_TC_NS
import KFTS_TC_NS
import UCBSM_TC_NS

# %% ####### Trial Class #######
### Non-Stationary ###
#Set up Reward and Choice Arrays
explorationArray = np.zeros(shape=[8, numPart, numTrials])

#Parameter Generation
#Load Arrays
LLArray = np.genfromtxt('LLArray_NS.csv', delimiter=',')
modeledParam = np.genfromtxt('modeledParam_NS.csv', delimiter=',')

arrayValues = np.zeros(shape=[2,numTrials])

#Average Human data
for partCounter in range(numPart):
    
    print(partCounter)
    
    #File names
    fileName = all_files_NS[partCounter]
    fileName_Mat = all_files_Mat[partCounter]
    
    #Import data and assing to array
    data = np.loadtxt(fileName, delimiter="\t")

    data[:,3] = data[:,3] - 1
    data[:,4] = data[:,4] / 100
    
    #Modify NaN's to -1 (for functions)
    data[np.isnan(data)] = -1
    
    #Assign to Array
    arrayValues[0,:] = np.transpose(data[:,3])
    arrayValues[1,:] = np.transpose(data[:,4])
    
    #Random
    object = random_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,0],numArms)  
    randomChoice = object.random()
    explorationArray[0,partCounter,:] = randomChoice

    #Run Trial Classification for each Model
    #eGreedy
    object = eGreedy_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,1:3],numArms)  
    eGreedy = object.eGreedy()
    explorationArray[1,partCounter,:] = eGreedy
    
    #Softmax
    object = softMax_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,3:5],numArms)  
    softVal = object.softmax()
    explorationArray[2,partCounter,:] = softVal
    
    #Win-Stay, Lose-Shift
    object = WSLS_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,5:7],numArms)  
    WSLSchoice = object.WSLS_TC()
    explorationArray[3,partCounter,:] = WSLSchoice

    #UCB
    object = UCB_SW_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,7],numArms)  
    confBound = object.UCB_SW()
    explorationArray[4,partCounter,:] = confBound
    
    #UCB
    object = gradient_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,8:10],numArms)  
    grad = object.gradient()
    explorationArray[5,partCounter,:] = grad[0]    
    
    # KFTS
    object = KFTS_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,10:12],numArms)  
    KFTS = object.KFTS()
    explorationArray[6,partCounter,:] = KFTS 
    
    # UCBSM
    object = UCBSM_TC_NS.trialClassification(arrayValues,modeledParam[partCounter,12:14],numArms)  
    UCBSM = object.UCBSM()
    explorationArray[7,partCounter,:] = UCBSM    
    
# %% Trial Overlap
overlapTable = np.zeros(shape=[7, 7])

#Overlap Stuff
choiceEG = explorationArray[1, :, :]
choiceSM = explorationArray[2, :, :]
choiceWSLS = explorationArray[3, :, :]
choiceUCB = explorationArray[4, :, :]
choiceGrad = explorationArray[5, :, :]
choiceKFTS = explorationArray[6, :, :]
choiceUCBSM = explorationArray[7, :, :]

#Egreedy
overlapTable[0, 0] = 1
overlapTable[1, 0] = np.sum(choiceEG == choiceSM) / (numTrials*numPart)
overlapTable[2, 0] = np.sum(choiceEG == choiceWSLS) / (numTrials*numPart)
overlapTable[3, 0] = np.sum(choiceEG == choiceUCB) / (numTrials*numPart)
overlapTable[4, 0] = np.sum(choiceEG == choiceGrad) / (numTrials*numPart)
overlapTable[5, 0] = np.sum(choiceEG == choiceKFTS) / (numTrials*numPart)
overlapTable[6, 0] = np.sum(choiceEG == choiceUCBSM) / (numTrials*numPart)

#SM
overlapTable[1, 1] = 1
overlapTable[2, 1] = np.sum(choiceSM == choiceWSLS) / (numTrials*numPart)
overlapTable[3, 1] = np.sum(choiceSM == choiceUCB) / (numTrials*numPart)
overlapTable[4, 1] = np.sum(choiceSM == choiceGrad) / (numTrials*numPart)
overlapTable[5, 1] = np.sum(choiceSM == choiceKFTS) / (numTrials*numPart)
overlapTable[6, 1] = np.sum(choiceSM == choiceUCBSM) / (numTrials*numPart)

# WSLS
overlapTable[2, 2] = 1
overlapTable[3, 2] = np.sum(choiceWSLS == choiceUCB) / (numTrials*numPart)
overlapTable[4, 2] = np.sum(choiceWSLS == choiceGrad) / (numTrials*numPart)
overlapTable[5, 2] = np.sum(choiceWSLS == choiceKFTS) / (numTrials*numPart)
overlapTable[6, 2] = np.sum(choiceWSLS == choiceUCBSM) / (numTrials*numPart)

# UCB
overlapTable[3, 3] = 1
overlapTable[4, 3] = np.sum(choiceUCB == choiceGrad) / (numTrials*numPart)
overlapTable[5, 3] = np.sum(choiceUCB == choiceKFTS) / (numTrials*numPart)
overlapTable[6, 3] = np.sum(choiceUCB == choiceUCBSM) / (numTrials*numPart)

# Grad
overlapTable[4, 4] = 1
overlapTable[5, 4] = np.sum(choiceGrad == choiceKFTS) / (numTrials*numPart)
overlapTable[6, 4] = np.sum(choiceGrad == choiceUCBSM) / (numTrials*numPart)

# KFTS
overlapTable[5, 5] = 1
overlapTable[6, 5] = np.sum(choiceKFTS == choiceUCBSM) / (numTrials*numPart)

# Last
overlapTable[6, 6] = 1

# Save as numpy array
NS_Overlap = np.save('./Arrays_NS/NS_overlapTable.npy',overlapTable)

# %% Compute Exploration Rate
exploreMean = np.zeros(shape=[numPart, 8])

for modelCount in range(8):

    for partcount in range(numPart):    

        exploreMean[partcount,modelCount] = np.count_nonzero(explorationArray[modelCount,partcount,:] == 1) / (numTrials)


#Convert to Array
exploreMean_1 = np.zeros(shape=(30,9))
exploreMean_1[:,0] = np.linspace(1,30,30)
exploreMean_1[:,1:9] = exploreMean 

#Convert to DF?
exploreDF_Wide = pd.DataFrame(exploreMean_1)
exploreDF_Wide.columns = ["Part", "Bias", "eGreedy", "Softmax", 
                          "WSLS", "UCB-SW", "Gradient","KFTS", "UCBSM"]

#Convert from wide to long?
exploreDF_Long = pd. melt(exploreDF_Wide, id_vars='Part', 
                          value_vars=["Bias", "eGreedy", "Softmax", 
                            "WSLS", "UCB-SW", "Gradient","KFTS", "UCBSM"])
exploreDF_Long['value'] = exploreDF_Long['value'].apply(lambda x: x*100)

exploreDF_Long.to_csv('./Arrays_NS/NS_EERate.csv') 

# %% Compute Best Model
BICArray = np.zeros(shape=[numPart,8])
#Convert LL to BIC
k = [1,2,2,2,1,2,2,2] #Currently the same for all models
for count in range(8):
    BICArray[:,count] = (k[count]* np.log(numTrials)) + 2*(LLArray[:,count])

# Subtract baseline BIC
BICBase = 1 - (BICArray/ BICArray[:,0,None])
BICBase = np.delete(BICBase,0,1)

# Convert to Array
BICMean_1 = np.zeros(shape=(30,8))
BICMean_1[:,0] = np.linspace(1,30,30)
BICMean_1[:,1:8] =  BICBase

#Convert to DF?
BICDF_Wide = pd.DataFrame(BICMean_1)
BICDF_Wide.columns = ["Part","eGreedy","Softmax","WSLS","UCB-SW","Gradient","KFTS","UCBSM"]

#Convert from wide to long?
BICDF_Long = pd. melt(BICDF_Wide, id_vars='Part', 
                      value_vars=["eGreedy","Softmax","WSLS","UCB-SW","Gradient","KFTS","UCBSM"])
BICDF_Long.to_csv('./Arrays_NS/NS_BIC.csv') 



