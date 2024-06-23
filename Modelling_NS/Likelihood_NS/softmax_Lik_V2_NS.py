# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:57:48 2022

@author: Tom
"""

def softmax_Lik(parameters,
            rewardVal,
            choices,
            trialParam):
    
    import numpy as np


    #Initialize Parameters
    numArms = int(trialParam[0])
    numTrials = int(trialParam[1])
    
    learningRate = parameters[1]
    temperature = parameters[0]
    
    #Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numTrials])
    
    #Initialize samples
    selectCount = np.ones(shape=numArms)
    
    #Initialize Q Values
    qValue = np.zeros(shape=numArms)+.5
    
    #Loop around trials
    for trial in range(numTrials):
 
        #Extract Participant choice
        selection = int(choices[trial])
        
        
        if selection == -1:
            
            liklihoodArray[trial] = 1
            
        else:
        
            #Compute Softmax values
            num = np.exp( np.multiply(qValue,temperature))
            denom = sum(np.exp(np.multiply(qValue,temperature)))
            
            #Find softmax result
            softmaxResult = num/denom
            
            #Update Arm Count!
            selectCount[selection] += 1  
            
            #Compute Reward
            reward = rewardVal[trial]
            
            #Compute Prediction Error
            predError = reward - qValue[selection]
            
            #Update Reward - Non Stationary
            qValue[selection] = qValue[selection] + learningRate * predError              
            
            #Note this is needed to avoid overly large values which numpy can't handle
            if qValue[selection] > 1:
                qValue[selection] = 1
            if qValue[selection] < -1:
                qValue[selection] = -1
            
            #Compute Liklihood
            liklihoodArray[trial] = softmaxResult[selection]
            
            if liklihoodArray[trial] <= 0 or np.isnan(liklihoodArray[trial]):
                liklihoodArray[trial] = 1e300
    
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))

      
    return liklihoodSum



