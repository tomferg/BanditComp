# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:58:33 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def UCBSM_Lik(parameters, rewards_LL, choices_LL, taskParam):
    
    import numpy as np
    
    # Task Parameters
    numArms = taskParam[0]
    numTrials = taskParam[2]
    numBlocks = taskParam[1]
    
    # Isolate Parameters
    tempParam = parameters[0]
    expParam = parameters[1]
    learningRate = .2
    
    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numBlocks, numTrials])
    
    for bCt in range(numBlocks):
    
        #Initialize Arm sample history
        selectCount = np.ones(shape=numArms)
        
        # Needed so trial one isn't 0 in the SM equation
        n = 1
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms)+1e-5
        
        # Initialize Uncertainty
        uncert = np.zeros(shape=numArms)+1e-5
    
        for tCt in range(numTrials):
            
            # Find selection and reward
            selection = int(choices_LL[bCt, tCt])
                
            if selection == -1:
                
                liklihoodArray[bCt, tCt] = 1
                
            else:
            
                # Version from UCB1
                uncert = expParam * np.sqrt(np.log(n) / (selectCount))
                
                # Calculate Softmax Probabilities 
                num = np.exp(np.multiply(qValue,tempParam) + uncert)
                denom = sum(np.exp(np.multiply(qValue,tempParam) + uncert));
                
                # Actual Softmax
                softmaxResult = num/denom
                
                # Selection
                #selection = int(choices_LL[bCt, tCt])
                
                # Reward
                reward = rewards_LL[bCt, tCt]
                
                #Update Arm Count!
                selectCount[selection] += 1
                n += 1
                
                #Compute Prediction Error
                predError = reward - qValue[selection]
                
                #Update Reward - Non Stationary
                # qValue[selection] = qValue[selection] + lrParam * predError
                qValue[selection] = qValue[selection] + learningRate * predError
    
                # Assign Likelihood
                liklihoodArray[bCt, tCt] = softmaxResult[selection]
            

    # Deal with Zeros
    liklihoodArray[liklihoodArray <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray[np.isnan(liklihoodArray)] = 1e+300
    
    # Sum LL
    liklihoodSum = -np.sum(np.log(liklihoodArray))
        
    # print('hello')
    
    return liklihoodSum