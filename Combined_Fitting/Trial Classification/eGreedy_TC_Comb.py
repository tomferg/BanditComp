# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 17:28:41 2022

@author: Tom
"""

import numpy as np

#Action Selection
class trialClassification:
    
    '''
    Code classifies human trial by trial data as exploration or exploitation
    Runs an e-greedy approach
    
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

        
    # def greedy(self):
    #     raise Exception('No Model Ready!')
            
    def eGreedy(self):
        
        #Initialize Parameters
        expParam = self.parameters[0]
        learningRate = self.parameters[1]
        
        
        for block in range(self.numBlocks):
    
            #Initialize samples
            selectCount = np.ones(shape=self.numArms)
            
            #Initialize Q Values
            qValues = np.zeros(shape=self.numArms) + .000001
            
            #Initialize Reward Array
            rewardBlock = self.arrayVal[0,block,:] 
            choiceBlock = self.arrayVal[1,block,:]
        
            for trial in range(self.numTrials):
                
                maxBandit = []
                
                humanSelect = int(choiceBlock[trial])
                
                if trial == 0 or humanSelect == -1:
                    
                    trialType = -1
                    
                else:
                    
                    greedyResult = (expParam / (self.numArms)) * np.ones(self.numArms)
                    
                    maxBandit = np.argmax(qValues)
                    
                    greedyResult[maxBandit] = (1-expParam) + (expParam / (self.numArms))
                  
                
                    if maxBandit == humanSelect:
                        
                        trialType = 0
                        
                    else:
                        
                        trialType = 1
                     
                    #Update Selection count
                    selectCount[humanSelect] += 1
                    
                    #Compute Reward                    
                    if rewardBlock[trial] == 1:
                        
                        reward = 1
                    
                    else:
                        
                        reward = 0 
                    
                    #Compute Prediction Error
                    predError = reward - qValues[humanSelect]
                                        
                    qValues[humanSelect] = qValues[humanSelect] + learningRate * predError
                
                #Update Exploration Array
                self.explorationMat[block,trial] = trialType
              
        return self.explorationMat     
        


    