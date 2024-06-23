# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:57:48 2022

@author: Tom

"""

def WSLS_Lik(parameters,
         rewardVal,
         choices,
         enviroType,
         trialParam,
         rewardType):
    
    import numpy as np
    
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
        
    #Initialize Parameters
    winStay = .5
    loseShift = parameters
    
    
    liklihoodArray = np.zeros(shape=[numBlocks,numTrials])
    
    for block in range(numBlocks):
        
        #Set Up Arrays
        trialReward = rewardVal[block,:]
        choiceBlock = choices[block,:]
        p = np.zeros(shape = 2)
        selection = []
        
        for trial in range(numTrials):
                        
            if trial == 0 and block == 0:
                
                p = [.5,.5]
                
            else:
                
                if trialReward[trial-1] == 1:
                
                    p = (winStay / numArms) * np.ones(numArms)
                    p[selection] = 1-winStay/numArms
                        
                else:
                    
                    p = (loseShift /numArms) * np.ones(numArms)
                    p[selection] = 1- loseShift / numArms
                    
            
            selection = int(choiceBlock[trial])
           
            liklihoodArray[block,trial] = p[selection]             
                
            if liklihoodArray[block,trial] <= 0:
                liklihoodArray[block,trial] = 1
            

    liklihoodSum = -np.sum(np.log(liklihoodArray))
          
    return liklihoodSum


