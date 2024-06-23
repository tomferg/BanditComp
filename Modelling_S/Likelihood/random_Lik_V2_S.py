# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:57:04 2022

@author: Tom
"""


            
def random_Lik(parameters,
         rewardVal,
         choices,
         enviroType,
         trialParam,
         rewardType):
    
    import numpy as np
    
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
    
    if enviroType == 'Non-stationary':
        biasParam = parameters[0]
        learningRate = parameters[1]
    else:
        #Initialize Parameters
        biasParam = parameters
        learningRate = 'NaN'
    
    #Create Liklihood Array
    liklihoodArray = np.zeros(shape=[numBlocks,numTrials])
    
    #Assign Bias Parameter to one arm?
    probStuff = [biasParam, 1 - biasParam]
    
    for block in range(numBlocks):
            
        #Shuffle Bias for each block?
        np.random.shuffle(probStuff)
    
        #Initialize samples
        selectCount = np.ones(shape=numArms)
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms)
        
        #Set Up Arrays
        trialReward = rewardVal[block,:]
        choiceBlock = choices[block,:]

        for trial in range(numTrials):
            
            # Select action according to UCB Criteria
            selection = int(choiceBlock[trial])
            
            if selection == -1:
                
                liklihoodArray[block,trial] = 1
                
            else:
            
                #reward = np.random.normal() + rewVals[a]
                #Get Reward
                #Compute Reward
                if rewardType == "Points":
                    reward = trialReward[trial] / 100
                else:
                    reward = trialReward[trial]
                    
                #Update Arm Count!
                selectCount[selection] += 1  
                
                #Compute Prediction Error
                predError = reward - qValue[selection]
                
                #Update Reward - Non Stationary
                if enviroType == 'Non-Stationary':
                    
                    qValue[selection] = qValue[selection] + learningRate * predError
                    
                #stationary Update Instead (no learning rate)
                else:
                    
                    qValue[selection] = qValue[selection] + (1 / selectCount[selection]) * predError
                
                #Compute Liklihood
                liklihoodArray[block,trial] = probStuff[selection]
                
                #print(probArray[selection]) 
                if liklihoodArray[block,trial] <= 0:
                    liklihoodArray[block,trial] = 1e300
            
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))
 
    return liklihoodSum
    