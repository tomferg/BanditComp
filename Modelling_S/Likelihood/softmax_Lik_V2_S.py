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
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
    

    temperature = parameters
    learningRate = .2
    
    #Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numBlocks,numTrials])
    
    for block in range(numBlocks):
    
        #Initialize samples
        selectCount = np.ones(shape=numArms)
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms)+.00001
        
        #Set Up Arrays
        trialReward = rewardVal[block,:]
        choiceBlock = choices[block,:]
        
        #Loop around trials
        for trial in range(numTrials):
 
            #Extract Participant choice
            selection = int(choiceBlock[trial])
            
            
            if selection == -1:
                
                liklihoodArray[block,trial] = 1
                
            else:
            
                #Compute Softmax values
                num = np.exp( np.multiply(qValue, temperature))
                denom = sum(np.exp(np.multiply(qValue, temperature)))
                
                #Find softmax result
                softmaxResult = num/denom
                
                #Update Arm Count!
                selectCount[selection] += 1  
                
                #Compute Reward
                reward = trialReward[trial]
                
                #Compute Prediction Error
                predError = reward - qValue[selection]
                
                #Update Reward - Non Stationary
                qValue[selection] = qValue[selection] + learningRate * predError            
                    
                #Compute Liklihood
                liklihoodArray[block,trial] = softmaxResult[selection]
                
                if liklihoodArray[block,trial] <= 0:
                    liklihoodArray[block,trial] = 1e300
        
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))

          
    return liklihoodSum
    


