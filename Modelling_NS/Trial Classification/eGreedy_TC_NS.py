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
        self.numTrials = np.shape(arrayVal)[1]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numTrials])

        
    # def greedy(self):
    #     raise Exception('No Model Ready!')
            
    def eGreedy(self):
        
        #Initialize Parameters
        expParam = self.parameters[0]
        learningRate = self.parameters[1]
        
        #Initialize samples
        selectCount = np.ones(shape=self.numArms)
        
        #Initialize Q Values
        qValues = np.zeros(shape=self.numArms) + .000001
        
        #Initialize Reward Array
        rewardBlock = self.arrayVal[1,:]
        choiceBlock = self.arrayVal[0,:]
        
        for trial in range(self.numTrials):
            
            maxBandit = []
            
            humanSelect = int(choiceBlock[trial])
            
            if trial == 0 or humanSelect == -1:
                
                trialType = -1
                
            else:
                
                greedyResult = (expParam / (self.numArms)) * np.ones(self.numArms)
                
                maxBandit = np.argmax(qValues)
                
                greedyResult[maxBandit] = (1-expParam) + (expParam / self.numArms)
              
            
                if maxBandit == humanSelect:
                    
                    trialType = 0
                    
                else:
                    
                    trialType = 1
                 
                #Update Selection count
                selectCount[humanSelect] += 1
                
                #Compute Reward                    
                reward = rewardBlock[trial]
                
                #Compute Prediction Error
                predError = reward - qValues[humanSelect]
                                    
                qValues[humanSelect] = qValues[humanSelect] + learningRate  * predError
                
                if qValues[humanSelect] > 1:
                    qValues[humanSelect] = 1
                if qValues[humanSelect] < -1:
                    qValues[humanSelect] = -1

            
            #Update Exploration Array
            self.explorationMat[trial] = trialType
              
        return self.explorationMat     
        


    