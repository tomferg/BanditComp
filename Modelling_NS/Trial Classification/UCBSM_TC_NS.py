# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:19:37 2023

@author: Tom Ferguson, PhD, University of Alberta
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
        
        
    def UCBSM(self):
        
        # Isolate Parameters
        tempParam = self.parameters[0]
        expParam = self.parameters[1]
        lrParam = .5
        
        #Initialize Reward Array
        rewardBlock = self.arrayVal[1,:]
        choiceBlock = self.arrayVal[0,:]
        
        # Needed so trial one isn't 0 in the SM equation
        n = 1
        
        #Initialize Q Values
        qValue = np.zeros(shape=self.numArms)+1e-5
        
        # Initialize Uncertainty
        uncert = np.zeros(shape=self.numArms)+1e-5
        
        
        for tCt in range(self.numTrials):
            
            modelChoice = []
            
            humanSelect = int(choiceBlock[tCt])
            
            reward = rewardBlock[tCt]
            
            
            if tCt == 0 or humanSelect == -1:
                
                trialType = -1
                
            else:
            
                # Version from UCB1
                #uncert = expParam * np.sqrt(np.log(n) / (selectCount))
                
                # Version from S&K 2015
                lastCount = np.zeros(shape=self.numArms)
                for cTc in range(self.numArms):
                    x = np.array(np.where(choiceBlock[0:tCt] == cTc))
                    if x.size == 0:
                        lastCount[cTc] = 0    
                    else:
                        lastCount[cTc] = np.amax(x)
                uncert = (expParam * (n - lastCount)) / 100
                
                # Calculate Softmax Probabilities 
                num = np.exp(np.multiply(qValue,tempParam) + uncert)
                denom = sum(np.exp(np.multiply(qValue,tempParam) + uncert));
                
                # Actual Softmax
                softmaxResult = num/denom
                
                # Selection
                modelChoice = np.argmax(softmaxResult)
                
                if modelChoice == humanSelect:
                    
                    trialType = 0
                    
                else:
                    
                    trialType = 1
                
                #Update Arm Count!
                n += 1
                
                #Compute Prediction Error
                predError = reward - qValue[humanSelect]
                
                #Update Reward - Non Stationary
                qValue[humanSelect] = qValue[humanSelect] + lrParam * predError
                

            #Update Exploration Array
            self.explorationMat[tCt] = trialType
          
        return self.explorationMat         
        