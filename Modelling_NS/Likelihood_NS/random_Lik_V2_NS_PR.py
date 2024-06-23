# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:57:04 2022

@author: Tom
"""


            
def random_Lik(parameters,
         rewardVal,
         choices,
         trialParam,):
    
    import numpy as np
    
    numArms = int(trialParam[0])
    numTrials = int(trialParam[1])
    
    biasParam = parameters
    
    #Create Liklihood Array
    liklihoodArray = np.zeros(shape=[numTrials])
    
    #Assign Bias Parameter to one arm?
    probStuff = [biasParam, (1 - biasParam) / (numArms-1),\
                 (1 - biasParam) / (numArms-1),(1 - biasParam) / (numArms-1)]
    
    #Shuffle bias
    np.random.shuffle(probStuff)
    
    for trial in range(numTrials):
                
        if trial == -1:
            
            liklihoodArray[trial] = 1
            
        else:
        
            # Select action according to UCB Criteria
            selection = int(choices[trial])
                
            #Compute Liklihood
            liklihoodArray[trial] = probStuff[selection]
            
            #print(probArray[selection]) 
            if liklihoodArray[trial] <= 0 or np.isnan(liklihoodArray[trial]):
                liklihoodArray[trial] = 1e300
            
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))
 
    return liklihoodSum
    