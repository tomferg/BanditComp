# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:52:01 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def UCBSM_Lik(parameters,
            rewardVal,
            choices,              
            trialParam):
    
    import numpy as np
    
    # Task Parameters
    numTrials = trialParam[1]
    numArms = trialParam[0]

    # Isolate Parameters
    tempParam = parameters[0]
    expParam = parameters[1]
    lrParam = .5
    
    #Initialize Arm sample history
    selectCount = np.ones(shape=numArms)
    
    # Needed so trial one isn't 0 in the SM equation
    n = 1
    
    #Initialize Q Values
    qValue = np.zeros(shape=numArms)+1e-5
    
    # Initialize Uncertainty
    uncert = np.zeros(shape=numArms)+1e-5
    
    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numTrials])
    
    for tCt in range(numTrials):
        
        # Find selection and reward
        selection = int(choices[tCt])
        reward = rewardVal[tCt]
        
        if selection == -1:
            
            liklihoodArray[tCt] = 1
            
        else:
        
            # Version from UCB1
            #uncert = expParam * np.sqrt(np.log(n) / (selectCount))
            
            # Version from S&K 2015
            lastCount = np.zeros(shape=numArms)
            for cTc in range(numArms):
                x = np.array(np.where(choices[0:tCt] == cTc))
                if x.size == 0:
                    lastCount[cTc] = 0    
                else:
                    lastCount[cTc] = np.amax(x)
            uncert = (expParam * (n - lastCount)) / 100
            
            # Calculate Softmax Probabilities 
            num = np.exp(np.multiply(qValue,tempParam) + uncert)
            denom = sum(np.exp(np.multiply(qValue,tempParam) + uncert));
            
            # Actual Softmax
            softmaxResult = num/denom
            
            # Selection
            selection = int(choices[tCt])
            
            # Reward
            reward = rewardVal[tCt]
            
            #Update Arm Count!
            selectCount[selection] += 1
            n += 1
            
            #Compute Prediction Error
            predError = reward - qValue[selection]
            
            #Update Reward - Non Stationary
            qValue[selection] = qValue[selection] + lrParam * predError
            
            # Assign Likelihood
            liklihoodArray[tCt] = softmaxResult[selection]
            

    # Deal with Zeros
    liklihoodArray[liklihoodArray <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray[np.isnan(liklihoodArray)] = 1e+300
    
    # Sum LL
    liklihoodSum = -np.sum(np.log(liklihoodArray))
        
    # print('hello')
    
    return liklihoodSum