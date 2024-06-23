# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:21:43 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def Bias_Lik_Comb(parameters,
            rewardVal,
            choices,              
            taskParam):
    
    import numpy as np
        
    # %% Task Set up
    numArms_NS = int(taskParam[0])
    numTrials_NS = int(taskParam[1])
    numArms_S = int(taskParam[2])
    numTrials_S = int(taskParam[3])
    numBlocks = int(taskParam[4])
    
    #Initialize Parameters
    biasParam = parameters
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]

# %% 
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
    
    # Assign Bias Parameter to one arm?
    probStuff = [biasParam, (1 - biasParam) / (numArms_NS-1),\
                 (1 - biasParam) / (numArms_NS-1),(1 - biasParam) / (numArms_NS-1)]
    
    #Shuffle bias
    np.random.shuffle(probStuff)
    
    for trial in range(numTrials_NS):
                
        if trial == -1:
            
            liklihoodArray_NS[trial] = 1
            
        else:
        
            # Select action according to UCB Criteria
            selection = int(choices_NS[trial])
                
            #Compute Liklihood
            liklihoodArray_NS[trial] = probStuff[selection]
            
            #print(probArray[selection]) 
            if liklihoodArray_NS[trial] == 0:
                liklihoodArray_NS[trial] = 1


    # Deal with Zeros
    liklihoodArray_NS[liklihoodArray_NS <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_NS[np.isnan(liklihoodArray_NS)] = 1e+300
        
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))


# %% Stationary

    #Create Liklihood Array
    liklihoodArray_S = np.zeros(shape=[numBlocks, numTrials_S])
            
    #Assign Bias Parameter to one arm?
    probStuff = [biasParam, 1 - biasParam]
    
    for block in range(numBlocks):
    
        #Initialize samples
        selectCount = np.ones(shape=numArms_S)
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms_S)
        
        #Set Up Arrays
        trialReward = reward_S[block,:]
        choiceBlock = choices_S[block,:]

        for trial in range(numTrials_S):
            
            # Select action according to UCB Criteria
            selection = int(choiceBlock[trial])
                
            #reward = np.random.normal() + rewVals[a]
            #Get Reward
            #Compute Reward
            reward = trialReward[trial]
                
            #Update Arm Count!
            selectCount[selection] += 1  
            
            #Compute Prediction Error
            predError = reward - qValue[selection]
            
                
            qValue[selection] = qValue[selection] + (1 / selectCount[selection]) * predError
            
            #Compute Liklihood
            liklihoodArray_S[ block, trial] = probStuff[selection]
            
            #print(probArray[selection]) 
            if liklihoodArray_S[block,trial] == 0:
                liklihoodArray_S[block,trial] = 1e-5


    # Deal with Zeros
    liklihoodArray_S[liklihoodArray_S <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_S[np.isnan(liklihoodArray_S)] = 1e+300
    
    #Update Liklihood Values
    liklihoodSum_S = -np.sum(np.log(liklihoodArray_S))


# %% Likelihood Junk (sum both)
    
    liklihoodSum = ((liklihoodSum_NS/400) + (liklihoodSum_S/100))*500
    #liklihoodSum = liklihoodSum_NS + liklihoodSum_S

    return liklihoodSum 