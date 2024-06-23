# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:22:34 2022

@author: Tom
"""

def eGreedy_Lik(parameters,
            rewardVal,
            choices,              
            trialParam):
    
    import numpy as np
    
    #Initialize Parameters
    learningRate = parameters[1]
    expParam = parameters[0]
            
    numArms = int(trialParam[0])
    numTrials = int(trialParam[1])
    
    #Initialize Liklihood Array
    liklihoodArray = np.zeros(shape=[numTrials])

    #Initialize samples
    selectCount = np.ones(shape=numArms)
    
    #Initialize Q Values
    qValue = np.zeros(shape=numArms) #+ .5
    
    for trial in range(numTrials):
        
        #print(choiceBlock)
        
        #Find actual selection
        selection = int(choices[trial])
        
        if selection == -1:
            
            liklihoodArray[trial] = 1
            
        else:
            
            #Compute e-greedy values - make sure it sums to one
            greedyResult = expParam/(len(qValue)-1)* np.ones(shape=len(qValue)) 

            #COnvert Q Value array to list
            qValueList = qValue.tolist()
    
            #Find Max Choice
            maxLoc = qValueList.index(max(qValueList))
       
            greedyResult[maxLoc] = 1 - expParam             
            
            #Update Arm Count!
            selectCount[selection] += 1  
            
            #Compute Reward
            reward = rewardVal[trial]

            #Compute Prediction Error
            predError = reward - qValue[selection]
            
            #Update Reward - Non Stationary
            qValue[selection] = qValue[selection] + learningRate * predError
            
            # #Note this is needed to avoid overly large values which numpy can't handle
            # if qValue[selection] > 1:
            #     qValue[selection] = 1
            # if qValue[selection] < -1:
            #     qValue[selection] = -1
            
            #Compute Liklihood
            liklihoodArray[trial] = greedyResult[selection]
            
            if liklihoodArray[trial] <= 0 or np.isnan(liklihoodArray[trial]):
                liklihoodArray[trial] = 1e300
    
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))
      
    return liklihoodSum



