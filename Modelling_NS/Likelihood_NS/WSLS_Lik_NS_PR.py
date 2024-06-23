# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:57:48 2022

@author: Tom

"""

def WSLS_Lik(parameters,
         rewardVal,
         choices,
         trialParam):
    
    import numpy as np
    
    numArms = int(trialParam[0])
    numTrials = int(trialParam[1])
        
    #Initialize Parameters
    winStay = parameters[0]
    loseShift = parameters[1]

    liklihoodArray = np.zeros(shape=[numTrials])
        
    for trial in range(numTrials):
                    
        if trial == 0:
            
            liklihoodArray[trial] = 1
            
        else:
            
            
            if rewardVal[trial-1] >= .50:
                
                if choices[trial] == choices[trial-1]:
                    
                    liklihoodArray[trial] = winStay
                    
                else:
                    
                    liklihoodArray[trial] = (1 - winStay) / (numArms - 1)
                
                
            else:
                
                if choices[trial] != choices[trial-1]:
                    
                    liklihoodArray[trial] = loseShift
                    
                else:
                    
                    liklihoodArray[trial] = (1 - loseShift) / (numArms - 1)
                
                           
        if liklihoodArray[trial] <= 0 or np.isnan(liklihoodArray[trial]):
            liklihoodArray[trial] = 1e300
            

    liklihoodSum = -np.sum(np.log(liklihoodArray))
          
    return liklihoodSum


