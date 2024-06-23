# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:53:03 2023

@author: Tom Ferguson, PhD, University of Alberta
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
        self.numBlocks = np.shape(arrayVal)[1]
        self.numTrials = np.shape(arrayVal)[2]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numBlocks,self.numTrials])
    
    def UCBSM(self):
                        
        # Isolate Parameters
        tempParam = self.parameters[0]
        expParam = self.parameters[1]
        
        learningRate = .2
        
        for bCt in range(self.numBlocks):
        
            #Initialize Arm sample history
            selectCount = np.ones(shape=self.numArms)
            
            # Needed so trial one isn't 0 in the SM equation
            n = 1
            
            #Initialize Q Values
            qValue = np.zeros(shape=self.numArms)-1
            
            # Initialize Uncertainty
            uncert = np.zeros(shape=self.numArms)+1e-5
            
            #Initialize Reward Array
            rewardBlock = self.arrayVal[0, bCt, :]
            choiceBlock = self.arrayVal[1, bCt, :]
        
        
            for tCt in range(self.numTrials):
                
                modelChoice = []
                
                humanSelect = int(choiceBlock[tCt])
                
                    
                if tCt == 0 or humanSelect == -1:
                    
                    trialType = -1
                    
                else:
                
                    # Version from UCB1
                    uncert = expParam * np.sqrt(np.log(n) / (selectCount))
                    
                    # Calculate Softmax Probabilities 
                    num = np.exp(np.multiply(qValue,tempParam) + uncert)
                    denom = sum(np.exp(np.multiply(qValue,tempParam) + uncert));
                    
                    #Find softmax result
                    softmaxResult = num/denom
                
                    #Find cumulative sum
                    softmaxSum = np.cumsum(softmaxResult)
                    
                    #Assign Values to softmax options
                    softmaxOptions = softmaxSum > np.random.rand()
                
                    #Find arm choice
                    if not any(softmaxOptions):
                        
                        #Randomly Select If no Options True
                        modelChoice = np.random.randint(len(qValue))   
                   
                    else:
                        #If not use Softmax
                        modelChoice = np.argmax(softmaxOptions)
                    
                    
                    if modelChoice == humanSelect:
                        
                        trialType = 0
                        
                    else:
                        
                        trialType = 1
                        
                    #Compute Reward                    
                    if rewardBlock[tCt] == 1:
                        
                        reward = 1
                    
                    else:
                        reward = 0 
                    
                    #Update Arm Count!
                    selectCount[humanSelect] += 1
                    n += 1
                    
                    #Compute Prediction Error
                    predError = reward - qValue[humanSelect]
                    
                    #Update Reward - Non Stationary
                    # qValue[selection] = qValue[selection] + lrParam * predError
                    qValue[humanSelect] = qValue[humanSelect] + learningRate * predError
        
        
                    #Update Exploration Array
                    self.explorationMat[bCt, tCt] = trialType
          
        return self.explorationMat  