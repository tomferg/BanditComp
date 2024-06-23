# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:09:16 2022

@author: Tom
"""

#Load Packages
import numpy as np
import os
import sys
from scipy.optimize import minimize
import random
from tqdm import tqdm

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
sys.path.insert(0, './Helper Functions/')
sys.path.insert(0, './Likelihood/')

#Assign Path
S_Path = 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/S_Data/'

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
from eGreedy_Lik_V2_S import eGreedy_Lik
from softmax_Lik_V2_S import softmax_Lik
from UCB_Lik_V2_S import UCB1_Lik
from random_Lik_V2_S import random_Lik
from gradient_Lik_S import gradient
from KFTS_Lik_S import KFTS_Lik
from UCBSM_Lik_S import UCBSM_Lik
from WSLS_Lik_S_PR import WSLS_Lik


#Set up Reward and Choice Arrays
rewArrayModel = np.zeros(shape=[8,numPart,numBlocks,numTrials])
choiceArrayModel = np.zeros(shape=[8,numPart,numBlocks,numTrials])

rewArrayPeople = np.zeros(shape=[numPart,numBlocks,numTrials])
choiceArrayPeople = np.zeros(shape=[numPart,numBlocks,numTrials])

#Parameter Generation - start as list?
#Array is set up: 
modeledParam = np.zeros(shape=[numPart, 9])
LLArray = np.zeros(shape=[numPart, 8]) 

#Task Parameters
taskParam = [numArms, numBlocks, numTrials]

# %% ###### Optimize #######
#Parameter Space
startLoc = np.zeros(shape=[9, startArray_Size])

#Loop around participants
for partCounter in tqdm(range(numPart)): #range(1): #

    #load data file
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
    rewards = np.array(list(split(reward_long, 5)))
    choices =  np.array(list(split(choices_long, 5)))
    
    #Assign to Array
    choiceArrayPeople[partCounter,:,:] = choices
    rewArrayPeople[partCounter,:,:] = rewards
    
    #Min arrays
    minArray_LL = np.zeros(shape=[8, startArray_Size])
    minArray_Para = np.zeros(shape=[9, startArray_Size]) 
    #Starting Loc randomization
    startLoc[0,:] = np.random.beta(1.1, 1.1, size=startArray_Size) #Bias Param
    startLoc[1,:] = np.random.beta(1.1, 1.1, size=startArray_Size) #Epsilon Param
    startLoc[2,:] = np.random.gamma(1.2, 5, size=startArray_Size) #Temperature Param
    startLoc[3,:] = np.random.beta(1.1, 1.1, size=startArray_Size) #WinStay Param
    startLoc[4,:] = np.random.gamma(1.1, 2, size=startArray_Size) #UCB - Info Bias Param
    startLoc[5,:] = np.random.beta(1.1, 1.1, size=startArray_Size) # Gradient       
    startLoc[6, :] = np.random.uniform(.01, 1, size=startArray_Size) #KFTS - SigXi
    startLoc[7, :] = np.random.gamma(1.2, 5,size=startArray_Size) #UCBSM - Temp
    startLoc[8, :] =  np.random.uniform(0.01, 10, size=startArray_Size) #UCBSM - Exp
    
    for startLocCounter in range(np.shape(startLoc)[1]):
        
        # Print Message
        print('Currently Finishing '+ str(partCounter) + str(startLocCounter))
        
        #Random
        #Set up parameters
        x0 = startLoc[0,startLocCounter]
        #Find Function to optimize
        res_Rnd = minimize(random_Lik, x0,args=(rewards, choices, 'Stationary', taskParam, 'Binary'),
                        method='trust-constr', bounds=((0.001, 1),))    
        #Add data to parameters    
        minArray_LL[0, startLocCounter] = res_Rnd.fun
        minArray_Para[0, startLocCounter] = res_Rnd.x
        
        #E-Greedy
        #Set up parameters
        x0 = startLoc[1,startLocCounter]
        #Find Function to optimize
        res_EG = minimize(eGreedy_Lik, x0,args=(rewards,choices, taskParam),
                        method='trust-constr',bounds=((0.001,1),))
        minArray_LL[1,startLocCounter] = res_EG.fun
        minArray_Para[1,startLocCounter] = res_EG.x

        #Softmax
        #Set up parameters
        x0 = startLoc[2,startLocCounter]
        #add bounds
        #bnds = ((0.0001, 1), (.0001,80))
        #Find Function to optimize
        res_SM = minimize(softmax_Lik, x0,args=(rewards,choices, taskParam),
                        method='trust-constr',bounds=((0.001, 20),))
        minArray_LL[2,startLocCounter] = res_SM.fun
        minArray_Para[2,startLocCounter] = res_SM.x
        
        #WSLS
        #Set up parameters
        x0 = startLoc[3,startLocCounter]
        #Find Function to optimize
        res_WSLS = minimize(WSLS_Lik, x0,args=(rewards, choices, 'Stationary',taskParam,'Binary'),
                        method='trust-constr',bounds=((0.0001, 1),))    
        #Add data to parameters    
        minArray_LL[3,startLocCounter] = res_WSLS.fun
        minArray_Para[3,startLocCounter] = res_WSLS.x
            
        #UCB
        #Set up parameters
        x0 = startLoc[4,startLocCounter]
        #add bounds
        #bnds = (0.0001, 5)
        #Find Function to optimize
        res_UCB = minimize(UCB1_Lik, x0,args=(rewards,choices, 'Stationary',taskParam,'Binary'),
                        method='trust-constr',bounds=((0.001,15),))    
        minArray_LL[4,startLocCounter] = res_UCB.fun
        minArray_Para[4,startLocCounter] = res_UCB.x
        
        # Gradient
        #Set up parameters
        x0 = startLoc[5,startLocCounter]
        #Find Function to optimize
        res_Grad = minimize(gradient, x0,args=(rewards, choices, taskParam),
                        method='trust-constr',bounds=((0.0001,5),))    
        #Add data to parameters    
        minArray_LL[5, startLocCounter] = res_Grad.fun
        minArray_Para[5, startLocCounter] = res_Grad.x
        
        # KFTS
        # Isolate Parameters
        random.seed(110)
        x0 = startLoc[6,startLocCounter]
        res_KFTS = minimize(KFTS_Lik, x0,args=(rewards, choices, taskParam),
                        method='trust-constr',bounds=((0.001, 2),))  
        minArray_LL[6, startLocCounter] = res_KFTS.fun
        minArray_Para[6, startLocCounter] = res_KFTS.x
        
        # UCB-SM
        x0 = startLoc[7:9,startLocCounter]
        res_UCBSM = minimize(UCBSM_Lik, x0,args=(rewards, choices, taskParam),
                        method='trust-constr',bounds=((0.01, 20),(.001, 10)))  
        minArray_LL[7, startLocCounter] = res_UCBSM.fun
        minArray_Para[7:9, startLocCounter] = res_UCBSM.x
    
    minArray_LL[minArray_LL == -0] = 10000   
    ninf = float('-inf')
    minArray_LL[minArray_LL == ninf] = 10000    

    #Assign Mins to parameters data to parameters
    #for modCount in range(numModels)
    #Add data to parameters    
    LLArray[partCounter,0] = min(minArray_LL[0,:])
    loc = np.where(minArray_LL[0,:] == minArray_LL[0,:].min())
    modeledParam[partCounter,0] = minArray_Para[0,loc[0][0]]

    #Add data to parameters    
    LLArray[partCounter,1] = min(minArray_LL[1,:])
    loc = np.where(minArray_LL[1,:] == minArray_LL[1,:].min())
    modeledParam[partCounter,1] = minArray_Para[1,loc[0][0]]
    
    #Add data to parameters    
    LLArray[partCounter,2] = min(minArray_LL[2,:])
    loc = np.where(minArray_LL[2,:] == minArray_LL[2,:].min())
    modeledParam[partCounter,2] = minArray_Para[2,loc[0][0]]
    
    #print(res.x)
    LLArray[partCounter,3] = min(minArray_LL[3,:])
    loc = np.where(minArray_LL[3,:] == minArray_LL[3,:].min())
    modeledParam[partCounter,3] = minArray_Para[3,loc[0][0]]

    LLArray[partCounter,4] = min(minArray_LL[4,:])
    loc = np.where(minArray_LL[4,:] == minArray_LL[4,:].min())
    modeledParam[partCounter,4] = minArray_Para[4,loc[0][0]]
    
    LLArray[partCounter,5] = min(minArray_LL[5,:])
    loc = np.where(minArray_LL[5,:] == minArray_LL[5,:].min())
    modeledParam[partCounter,5] = minArray_Para[5,loc[0][0]]
    
    LLArray[partCounter,6] = min(minArray_LL[6,:])
    loc = np.where(minArray_LL[6,:] == minArray_LL[6,:].min())
    modeledParam[partCounter,6] = minArray_Para[6,loc[0][0]]
    
    LLArray[partCounter,7] = min(minArray_LL[7,:])
    loc = np.where(minArray_LL[7,:] == minArray_LL[7,:].min())
    modeledParam[partCounter,7] = minArray_Para[7,loc[0][0]]
    modeledParam[partCounter,8] = minArray_Para[8,loc[0][0]]


# %% #Save LL Array and modeledParam Stuff as CSVs
np.savetxt('LLArray_S.csv', LLArray, delimiter=',', fmt='%2F')
np.savetxt('modeledParam_S.csv', modeledParam, delimiter=',', fmt='%2F')
