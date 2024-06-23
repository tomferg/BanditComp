# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:42:20 2022

@author: Tom
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:51:41 2022

@author: Tom
"""
import numpy as np

#Action Selection
class trialClassification:
    
    '''
    Code classifies human trial by trial data as exploration or exploitation
    Runs a win-stay, lose-shift approach
    
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
        self.numBlocks = np.shape(arrayVal)[1]
        self.numTrials = np.shape(arrayVal)[2]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numBlocks,self.numTrials])
        
        
    def WSLS_TC(self):
        
        
        for block in range(self.numBlocks):
            
            rewardBlock = self.arrayVal[0,block,:]
            choiceBlock = self.arrayVal[1,block,:]
            
            
            #Then loop around trials?
            for trial in range(self.numTrials):
                
                trialType = []
                
                if trial == 0 or choiceBlock[trial] == -1:
                    
                    trialType = -1
                    
                    
                else:
                    
                    #Previous trial Win
                    if rewardBlock[trial-1] == 1:
                        
                        if choiceBlock[trial] == choiceBlock[trial-1]:
                            trialType = 0
                        else:
                            trialType = 1

                    #Previous Trial Loss  
                    else:
                        
                        #Lose Shift
                        if choiceBlock[trial] != choiceBlock[trial-1]:
                            trialType = 0
                        else:
                            trialType = 1
      
                #Update Exploration Array
                self.explorationMat[block,trial] = trialType
              
        return self.explorationMat
    