# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:57:24 2023

@author: Tom Ferguson, PhD, University of Alberta
"""
import numpy as np
import os
import sys
from scipy import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

#Load Data and mat files
#Set up parameters
numPart = 30
startCount_NS = 0
matcount = 0
numArms_NS = 4
numTrials_NS = 400
numPart = 30
startCount_S = 0
numArms_S = 2
numTrials_S = 20
numBlocks = 5

#Set up arrays
#Loading Data
all_files_NS =  [None] * numPart
all_files_Mat = [None] * numPart

# adding folders to path - Environment variables are likely not ideal....
sys.path.insert(0, 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/Modelling_NS/Trial Classification')
sys.path.insert(0, 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/Modelling_S/Trial Classification')
sys.path.insert(0, 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/Combined_Fitting/Trial Classification')

#Assign Path
S_Path = 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/S_Data/'

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



#Set up arrays
#Loading Data
all_files_S =  [None] * numPart

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
import UCB_TC_2_S
import eGreedy_TC_Comb
import softMax_TC_Comb
import random_TC_S
import WSLS_TC_S
import gradient_TC_S
import KFTS_TC_S
import UCBSM_TC_Comb

#Parameter Generation
#Load Arrays
LLArray = np.genfromtxt('LLArray_Comb.csv', delimiter=',')
modeledParam = np.genfromtxt('modeledParam_Comb.csv', delimiter=',')
explorationArray_NS = np.zeros(shape=[8, numPart, numTrials_NS])

# %% Non Stationary TC

arrayValues_NS = np.zeros(shape=[2, numTrials_NS])

#Average Human data
for partCounter in tqdm(range(numPart)):
        
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
    arrayValues_NS[0,:] = np.transpose(data[:,3])
    arrayValues_NS[1,:] = np.transpose(data[:,4])
    
    #Random
    object = random_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,0],numArms_NS)  
    randomChoice = object.random()
    explorationArray_NS[0,partCounter,:] = randomChoice

    #Run Trial Classification for each Model
    #eGreedy
    object = eGreedy_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,1:3],numArms_NS)  
    eGreedy = object.eGreedy()
    explorationArray_NS[1,partCounter,:] = eGreedy
    
    #Softmax
    object = softMax_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,3:5],numArms_NS)  
    softVal = object.softmax()
    explorationArray_NS[2,partCounter,:] = softVal
    
    #Win-Stay, Lose-Shift
    object = WSLS_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,5:7],numArms_NS)  
    WSLSchoice = object.WSLS_TC()
    explorationArray_NS[3,partCounter,:] = WSLSchoice

    #UCB
    object = UCB_SW_TC_Comb.trialClassification(arrayValues_NS, modeledParam[partCounter,7:9],numArms_NS)  
    confBound = object.UCB_SW()
    explorationArray_NS[4,partCounter,:] = confBound
        
    # Gradient
    object = gradient_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,9:11],numArms_NS)  
    grad = object.gradient()
    explorationArray_NS[5,partCounter,:] = grad[0]    
    
    # KFTS
    object = KFTS_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,11:13],numArms_NS)  
    KFTS = object.KFTS()
    explorationArray_NS[6,partCounter,:] = KFTS 
    
    # UCBSM
    object = UCBSM_TC_NS.trialClassification(arrayValues_NS, modeledParam[partCounter,13:15],numArms_NS)  
    UCBSM = object.UCBSM()
    explorationArray_NS[7,partCounter,:] = UCBSM    


# %% Junk
exploreMean_NS = np.zeros(shape=[numPart, 8])

for modelCount in range(8):

    for partcount in range(numPart):    

        exploreMean_NS[partcount,modelCount] = np.count_nonzero(explorationArray_NS[modelCount,partcount,:] == 1) / (numTrials_NS)


#Convert to Array
exploreMean_NS_1 = np.zeros(shape=(30,9))
exploreMean_NS_1[:,0] = np.linspace(1,30,30)
exploreMean_NS_1[:,1:9] = exploreMean_NS

#Convert to DF?
exploreDF_Wide_NS = pd.DataFrame(exploreMean_NS_1)
exploreDF_Wide_NS.columns = ["Part", "Bias", "eGreedy", "Softmax", 
                          "WSLS", "UCB-SW", "Gradient","KFTS", "UCBSM"]

#Convert from wide to long?
exploreDF_Long_NS = pd. melt(exploreDF_Wide_NS, id_vars='Part', 
                          value_vars=["Bias", "eGreedy", "Softmax", 
                            "WSLS", "UCB-SW", "Gradient","KFTS", "UCBSM"])
exploreDF_Long_NS['value'] = exploreDF_Long_NS['value'].apply(lambda x: x*100)

exploreDF_Long_NS.to_csv('./Arrays_Comb/NS_EERate_Comb.csv') 


# %% Stationary
explorationArray_S = np.zeros(shape=[8, numPart, numBlocks, numTrials_S])

#Average Human data
for partCounter in tqdm(range(numPart)):
        
    #File names
    fileName = all_files_S[partCounter]
    
    #Import data and assing to array
    data = np.loadtxt(fileName, delimiter="\t")

    #extract reward values
    reward_long = data[:,14]
    #extract choices
    choices_long = data[:,10]
    
    #Modify choices for functions
    choices_long[choices_long == 1] = 0
    choices_long[choices_long == 2] = 1
    
    #Reorganize Choice and Reward Arrays
    rewards = np.array(list(split(reward_long,5)))
    choices =  np.array(list(split(choices_long,5)))

    #Concatinate choices and reward together
    arrayValues = np.zeros(shape=[2,numBlocks,numTrials_S])
    arrayValues[0,:,:] = rewards
    arrayValues[1,:,:] = choices
    
    #Run Trial Classification for each Model
    #Random
    object = random_TC_S.trialClassification(arrayValues,
                                             modeledParam[partCounter, 0], numArms_S)  
    randomChoice = object.random()
    explorationArray_S[0,partCounter,:] = randomChoice
    
    #eGreedy
    object = eGreedy_TC_Comb.trialClassification(arrayValues,
                                              modeledParam[partCounter, 1:3], numArms_S)  
    eGreedy = object.eGreedy()
    explorationArray_S[1,partCounter,:] = eGreedy
    
    #Softmax
    object = softMax_TC_Comb.trialClassification(arrayValues,
                                              modeledParam[partCounter, 3:5], numArms_S)  
    softVal = object.softmax()
    explorationArray_S[2,partCounter,:] = softVal
    
    #Win-Stay, Lose-Shift
    object = WSLS_TC_S.trialClassification(arrayValues, 
                                           modeledParam[partCounter, 6], numArms_S)  
    WSLSchoice = object.WSLS_TC()
    explorationArray_S[3,partCounter,:] = WSLSchoice
    
    #UCB
    object = UCB_TC_2_S.trialClassification(arrayValues,
                                            modeledParam[partCounter, 8], numArms_S)  
    confBound = object.UCB1()
    explorationArray_S[4,partCounter,:] = confBound
    
    # Gradient
    object = gradient_TC_S.trialClassification(arrayValues,
                                               modeledParam[partCounter, 10], numArms_S)  
    grad = object.gradient()
    explorationArray_S[5,partCounter,:] = grad
    
    # KFTS
    object = KFTS_TC_S.trialClassification(arrayValues,
                                               modeledParam[partCounter, 12], numArms_S)  
    KFTS = object.KFTS()
    explorationArray_S[6,partCounter,:] = KFTS
    
    #Thomp Samp
    object = UCBSM_TC_Comb.trialClassification(arrayValues,
                                               modeledParam[partCounter, 13:15], numArms_S)  
    UCBSM = object.UCBSM()
    explorationArray_S[7,partCounter,:] = UCBSM

# %% Compute Exploration Rate
exploreMean_S = np.zeros(shape=[numPart,8])

for modelCount in range(8):

    for partcount in range(numPart):    

        exploreMean_S[partcount,modelCount] = np.count_nonzero(explorationArray_S[modelCount,partcount,:,:] == 1) / (numTrials_S*numBlocks)


#Convert to Array
exploreMean_1_S = np.zeros(shape=(30, 9))
exploreMean_1_S[:, 0] = np.linspace(1, 30, 30)
exploreMean_1_S[:, 1:9] = exploreMean_S 

#Convert to DF?
exploreDF_Wide_S = pd.DataFrame(exploreMean_1_S)
exploreDF_Wide_S.columns = ["Part", "Random", "eGreedy", "Softmax", "WSLS", "UCB1", "Gradient", "KFTS", "UCBSM"]

#Convert from wide to long?
exploreDF_Long_S = pd. melt(exploreDF_Wide_S, id_vars='Part', 
                          value_vars=["Random", "eGreedy", "Softmax", "WSLS", "UCB1", "Gradient", "KFTS", "UCBSM"])
exploreDF_Long_S['value'] = exploreDF_Long_S['value'].apply(lambda x: x*100)

exploreDF_Long_S.to_csv('./Arrays_CombS_EERate_Comb.csv') 


# %% Compute Best Model Across All
BICArray = np.zeros(shape=[numPart,8])

#Convert LL to BIC
k = [1,2,2,2,1,2,2,2] 
for count in range(8):
    BICArray[:,count] = (k[count]* np.log(numTrials_NS)) + 2*(LLArray[:,count])

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
BICDF_Long.to_csv('./Arrays_Comb/NS_BIC_Comb.csv') 
