# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:11:45 2022

@author: Tom
"""

import numpy as np

#Action Selection
class trialClassification:
    
    '''
    Code classifies human trial by trial data as exploration or exploitation
    Runs a softmax approach
    
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
        
        
    def softmax(self):
        
            
        temperature  = self.parameters[0]
        learningRate = self.parameters[1]
                    
        #Initialize Q Values
        qValue = np.zeros(shape=self.numArms) + .00001
        
        #Initialize Reward Array
        rewardBlock = self.arrayVal[1,:]
        choiceBlock = self.arrayVal[0,:]
        
        #Selection Count
        selectCount = np.zeros(self.numArms)
        
        #Loop around trials
        for trial in range(self.numTrials):
            
            modelChoice = []
            
            humanSelect = int(choiceBlock[trial])
            
            if trial == 0 or humanSelect == -1:
                
                trialType = -1
                
            else:
                
                #Compute Softmax values
                num = np.exp(np.multiply(qValue,temperature))
                
                denom = sum(np.exp(np.multiply(qValue,temperature)));
            
                #Find softmax result
                softmaxResult = num/denom
            
            
                modelChoice = np.argmax(softmaxResult)
                
                if modelChoice == humanSelect:
                    
                    trialType = 0
                    
                else:
                    
                    trialType = 1
                 
                #Update Selection count
                selectCount[humanSelect] += 1
                
                #Compute Reward                    
                reward =  rewardBlock[trial]
                
                #Compute Prediction Error
                predError = reward - qValue[humanSelect]
                                    
                qValue[humanSelect] = qValue[humanSelect] + learningRate * predError
                
                if qValue[humanSelect] > 1:
                    qValue[humanSelect] = 1
                if qValue[humanSelect] < -1:
                    qValue[humanSelect] = -1

            
            #Update Exploration Array
            self.explorationMat[trial] = trialType
          
        return self.explorationMat        


