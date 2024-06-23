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
    expParam  = parameters
    learningRate = .2
            
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
    
    #Initialize Liklihood Array
    liklihoodArray = np.zeros(shape=[numBlocks,numTrials])
    
    
    for block in range(numBlocks):
        
        #Initialize samples
        selectCount = np.ones(shape=numArms)
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms) + .00001
        
        trialReward = rewardVal[block,:]
        choiceBlock = choices[block,:]
        
        for trial in range(numTrials):
            
            #print(choiceBlock)
            
            #Find actual selection
            selection = int(choiceBlock[trial])
            
            if selection == -1:
                
                liklihoodArray[block,trial] = 1
                
            else:

                #Compute e-greedy values - make sure it sums to one
                greedyResult = (expParam/(len(qValue))) * np.ones(shape=len(qValue)) 
        
                #COnvert Q Value array to list
                qValueList = qValue.tolist()
        
                #Find Max Choice
                maxLoc = qValueList.index(max(qValueList))
           
                greedyResult[maxLoc] = (1-expParam) + (expParam/len(qValue))
                 
                #Update Arm Count!
                selectCount[selection] += 1  
                
                #Compute Reward
                reward = trialReward[trial]
                    
                #Compute Prediction Error
                predError = reward - qValue[selection]
                
                #Update Reward - Non Stationary
                qValue[selection] = qValue[selection] + learningRate * predError
                
                #Compute Liklihood
                liklihoodArray[block,trial] = greedyResult[selection]
                
                if liklihoodArray[block,trial] <= 0:
                    liklihoodArray[block,trial] = 1e300
        
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))

          
    return liklihoodSum
    


