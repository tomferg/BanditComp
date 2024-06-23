# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:04:02 2022

@author: Tom
"""

import numpy as np

#Action Selection
class trialClassification:
    
    '''
    Code classifies human trial by trial data as exploration or exploitation
    Runs a random approach (baseline model)
    
    Takes input:
        (1) Array Values - reward and choices from human data
        (2) parameters - the optimized model parameters
        (3) numArms = the number of bandit arms
    
    Returns:
        (1) Exploration Matrix where 1 is explore, 0 is exploit
    
    '''
    def __init__(self,
                 arrayVal,
                 parameters,
                 numArms):
        
        self.arrayVal = arrayVal
        self.parameters = parameters
        self.numTrials = np.shape(arrayVal)[1]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numTrials])
            
    def random(self):
        
        #Initialize Parameters
        biasParam = self.parameters

        #Assign Bias Parameter to one arm?
        #probStuff = [biasParam, 1 - biasParam]
        probStuff = np.zeros(shape=self.numArms)
        probStuff[1] = (1 - biasParam) / (self.numArms-1)
        probStuff[2] = (1 - biasParam) / (self.numArms-1)
        probStuff[3] = (1 - biasParam) / (self.numArms-1)
        probStuff[0] = biasParam
        
        
        np.random.shuffle(probStuff)    
        
        #Initialize Reward Array
        #rewardBlock = self.rewardVal[0,block,:] 
        choiceBlock = self.arrayVal[0,:]
            
        for trial in range(self.numTrials):
            
            
            if trial == 0 or choiceBlock[trial] == -1:
                
                trialType = -1
                
            else:
                
                biasMax = np.argmax(probStuff)
                
                if biasMax == choiceBlock[trial]:
                    
                    trialType = 0
                            
                else:
                    
                    trialType = 1
                    
            #Update Exploration Array
            self.explorationMat[trial] = trialType
          
        return self.explorationMat
    


    