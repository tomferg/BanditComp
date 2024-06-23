# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:09:16 2022

@author: Tom
"""

# %% Load Packages
import numpy as np
import os
import sys
from scipy.optimize import minimize
from tqdm import tqdm

#Load Data and mat files
#Set up parameters
numPart = 30
startCount_NS = 0
numArms = 4
numTrials = 400
start_Size = 10

#Set up arrays
#Loading Data
all_files_NS =  [None] * numPart

# adding folders to path - Environment variables are likely not ideal....
sys.path.insert(0, './Helper Functions_NS/')
sys.path.insert(0, './Likelihood_NS/')

#Assign Path
NS_Path = 'C:/Users/Tom/Documents/Mac Stuff/RL_Comparison_Paper/NS_Data/'

#Stationary data
for file in os.listdir(NS_Path):
    if file.endswith(".txt"):
        
        all_files_NS[startCount_NS] = os.path.join(NS_Path, file)
        
        startCount_NS += 1
#Sort Files
all_file_NS = sorted(all_files_NS)


# %% ####### Call Functions #######
# Load Likelihood Functions
from random_Lik_V2_NS_PR import random_Lik
from eGreedy_Lik_V2_NS import eGreedy_Lik
from softmax_Lik_V2_NS import softmax_Lik
from WSLS_Lik_NS_PR import WSLS_Lik
from UCB_Slide_Lik_NS import UCB_Slide_Lik_NS
from gradient_Lik_NS import gradient_Lik_NS
from KFTS_Lik_NS import KFTS_Lik
from UCBSM_Lik_NS import UCBSM_Lik

#from DTS_AS_NS_TEST import thompSampGauss

# %% ###### Optimize #######
#Set up Reward and Choice Arrays
rewArrayPeople = np.zeros(shape=[numTrials])
choiceArrayPeople = np.zeros(shape=[numTrials])

#Parameter Generation - start as list?
#Array is set up: 
modeledParam = np.zeros(shape=[numPart, 14])
LLArray = np.zeros(shape=[numPart, 8]) 

#Task Parameters
taskParam = [numArms, numTrials]

# %% Loop around participants
#Parameter Space
startLoc = np.zeros(shape=[14, start_Size])

for partCounter in tqdm(range(numPart)): #range(1): #
        
    #load data file
    #File names
    fileName = all_files_NS[partCounter]
    
    #Import data and assing to array
    data = np.loadtxt(fileName, delimiter="\t")
    
    data[:,3] = data[:,3] - 1
    data[:,4] = data[:,4] / 100
    
    #Modify NaN's to -1 (for functions)
    data[np.isnan(data)] = -1
    
    #Assign to Array
    choiceArrayPeople[:] = data[:,3]
    rewArrayPeople[:] = data[:,4]
    
    #Min arrays
    minArray_LL = np.zeros(shape=[8, start_Size])
    minArray_Para = np.zeros(shape=[14, start_Size]) 
    #Starting Loc randomization
    startLoc[0, :] = np.random.beta(1.1, 1.1, size = start_Size) # N/A - Random - bias#
    startLoc[1, :] = np.random.beta(1.1, 1.1, size = start_Size) # egreedy - epsilon
    startLoc[2, :] = np.random.beta(1.1, 1.1, size = start_Size) # egreedy - learning rate
    startLoc[3, :] = np.random.gamma(1.2, 5, size = start_Size)  # Softmax - Temperature
    startLoc[4, :] = np.random.beta(1.1, 1.1, size = start_Size) # softmax - learning rate 
    startLoc[5, :] = np.random.beta(1.1, 1.1, size = start_Size) # Win-Stay - WSLS
    startLoc[6, :] = np.random.beta(1.1, 1.1, size = start_Size) # Lose-Shift - WSLS
    startLoc[7, :] = np.random.randint(1, 200 , size = start_Size)
    startLoc[8, :] = np.random.beta(1.1, 1.1, size = start_Size) # Gradient - LR
    startLoc[9, :] = np.random.beta(1.1, 1.1, size = start_Size) # Gradient - LR 2
    startLoc[10, :] = np.random.uniform(0.01, 16, size = start_Size) # KFTS - SigX
    startLoc[11, :] =  np.random.uniform(0.01, 10, size = start_Size)  # KFTS - SigEps
    startLoc[12, :] = np.random.gamma(1.2, 5, size = start_Size) # UCBSM - temp
    startLoc[13, :] = np.random.uniform(0.01, 2, size = start_Size) # UCBSM - expPara

    #Modify choices and rewards
    for startLocCounter in range(np.shape(startLoc)[1]):
        
        # Print Message
        print('Currently Finishing '+ str(partCounter) + str(startLocCounter))
        
        # Random
        #Set up parameters
        x0 = startLoc[0, startLocCounter]
        #Find Function to optimize
        res_Rnd = minimize(random_Lik, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr', bounds=((0.001, 1),))    
        #Add data to parameters    
        minArray_LL[0,startLocCounter] = res_Rnd.fun
        minArray_Para[0,startLocCounter] = res_Rnd.x
        
        #E-Greedy
        #Set up parameters
        x0 = startLoc[1:3, startLocCounter]
        #Find Function to optimize
        res_EG = minimize(eGreedy_Lik, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr',bounds=((0.01, .95), (0.01, .95))) #,  
        #Add data to parameters    
        minArray_LL[1, startLocCounter] = res_EG.fun
        minArray_Para[1:3, startLocCounter] = res_EG.x
        
        #Softmax
        #Set up parameters
        x0 = startLoc[3:5, startLocCounter]
        #Find Function to optimize
        res_SM = minimize(softmax_Lik, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr', bounds=((.000001, 40),(0.01, .95)))
        #print(res_SM.x)
        #Add data to parameters    
        minArray_LL[2, startLocCounter] = res_SM.fun
        minArray_Para[3:5, startLocCounter] = res_SM.x
        
        #WSLS
        #Set up parameters
        x0 = startLoc[5:7, startLocCounter]
        #Find Function to optimize
        res_WSLS = minimize(WSLS_Lik, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method="trust-constr", bounds=((0.001, .99), (.001, .99)))    
        #Add data to parameters    
        minArray_LL[3, startLocCounter] = res_WSLS.fun
        minArray_Para[5:7, startLocCounter] = res_WSLS.x
                    
        # UCB
        # Set up parameters
        x0 = startLoc[7,startLocCounter]
        # Find Function to optimize
        res_UCB_SW = minimize(UCB_Slide_Lik_NS, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr', bounds=((1, 400), )) 
        #Add data to parameters    
        minArray_LL[4,startLocCounter] = res_UCB_SW.fun
        minArray_Para[7,startLocCounter] = res_UCB_SW.x
        
        # Gradient
        # Set up parameters
        x0 = startLoc[8:10,startLocCounter]
        # Find Function to optimize
        res_Grad = minimize(gradient_Lik_NS, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr', bounds=((.00001, 5), (.00001, 5))) #
        #Add data to parameters    
        minArray_LL[5, startLocCounter] = res_Grad.fun
        minArray_Para[8:10, startLocCounter] = res_Grad.x
        
        # KL-TS
        # Set up parameters
        x0 = np.log(startLoc[10:12,startLocCounter])

        # Find Function to optimize
        res_KFTS = minimize(KFTS_Lik, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr', bounds=((-6, 8), (-6, 8))) #
        #Add data to parameters    
        minArray_LL[6, startLocCounter] = res_KFTS.fun
        minArray_Para[10:12, startLocCounter] = np.exp(res_KFTS.x)
        
        
        # UCB-SM
        # Set up parameters
        x0 = startLoc[12:14, startLocCounter]
        # Find Function to optimize
        res_UCBSM = minimize(UCBSM_Lik, x0,
                        args=(rewArrayPeople, choiceArrayPeople, taskParam),
                        method='trust-constr', bounds=((.01, 40), (.00001, 10))) #
        #Add data to parameters    
        minArray_LL[7, startLocCounter] = res_UCBSM.fun
        minArray_Para[12:14, startLocCounter] = res_UCBSM.x
        
    
    minArray_LL[minArray_LL == -0] = 10000   
    ninf = float('-inf')
    minArray_LL[minArray_LL == ninf] = 10000    

    
    #Assign Mins to parameters data to parameters
    #Random   
    LLArray[partCounter, 0] = min(minArray_LL[0, :])
    loc = np.where(minArray_LL[0, :] == minArray_LL[0, :].min())
    modeledParam[partCounter, 0] = minArray_Para[0, loc[0][0]]

    #E-greedy  
    LLArray[partCounter, 1] = min(minArray_LL[1, :])
    loc = np.where(minArray_LL[1, :] == minArray_LL[1, :].min())
    modeledParam[partCounter, 1] = minArray_Para[1, loc[0][0]]
    modeledParam[partCounter, 2] = minArray_Para[2, loc[0][0]]
    
    #Softmax
    LLArray[partCounter, 2] = min(minArray_LL[2, :])
    loc = np.where(minArray_LL[2,:] == minArray_LL[2, :].min())
    modeledParam[partCounter, 3] = minArray_Para[3, loc[0][0]]
    modeledParam[partCounter, 4] = minArray_Para[4, loc[0][0]]
    
    #WSLS
    LLArray[partCounter, 3] = min(minArray_LL[3, :])
    loc = np.where(minArray_LL[3, :] == minArray_LL[3, :].min())
    modeledParam[partCounter, 5] = minArray_Para[5, loc[0][0]]
    modeledParam[partCounter, 6] = minArray_Para[6, loc[0][0]]
    
    #UCB
    LLArray[partCounter, 4] = min(minArray_LL[4,:])
    loc = np.where(minArray_LL[4, :] == minArray_LL[4, :].min())
    modeledParam[partCounter, 7] = minArray_Para[7, loc[0][0]]
    
    # Gradient
    LLArray[partCounter, 5] = min(minArray_LL[5, :])
    loc = np.where(minArray_LL[5, :] == minArray_LL[5, :].min())
    modeledParam[partCounter,8] = minArray_Para[8, loc[0][0]]
    modeledParam[partCounter,9] = minArray_Para[9, loc[0][0]]
    
    # KFTS
    LLArray[partCounter, 6] = min(minArray_LL[6, :])
    loc = np.where(minArray_LL[6, :] == minArray_LL[6, :].min())
    modeledParam[partCounter,10] = minArray_Para[10, loc[0][0]]
    modeledParam[partCounter,11] = minArray_Para[11, loc[0][0]]
    
    # UCBSM
    LLArray[partCounter, 7] = min(minArray_LL[7, :])
    loc = np.where(minArray_LL[7, :] == minArray_LL[7, :].min())
    modeledParam[partCounter,12] = minArray_Para[12, loc[0][0]]
    modeledParam[partCounter,13] = minArray_Para[13, loc[0][0]]

# %%   Save LL Array and modeledParam Stuff as CSVs
np.savetxt('LLArray_NS.csv', LLArray, delimiter=',', fmt='%2F')
np.savetxt('modeledParam_NS.csv', modeledParam, delimiter=',', fmt='%2F')
