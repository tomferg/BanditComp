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
        self.numBlocks = np.shape(arrayVal)[1]
        self.numTrials = np.shape(arrayVal)[2]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numBlocks,self.numTrials])
        
        
    def softmax(self):
        
            
        temperature  = self.parameters[0]
        learningRate = self.parameters[1]
        
        for block in range(self.numBlocks):
            
            #Initialize Q Values
            qValue = np.zeros(shape=self.numArms) + .00001
            
            #Initialize Reward Array
            rewardBlock = self.arrayVal[0,block,:]
            choiceBlock = self.arrayVal[1,block,:]
            
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
                     
                    #Update Selection count
                    selectCount[humanSelect] += 1
                    
                    #Compute Reward                    
                    if rewardBlock[trial] == 1:
                        
                        reward = 1
                    
                    else:
                        reward = 0 
                    
                    #Compute Prediction Error
                    predError = reward - qValue[humanSelect]
                                        
                    qValue[humanSelect] = qValue[humanSelect] + learningRate * predError

                
                #Update Exploration Array
                self.explorationMat[block,trial] = trialType
              
        return self.explorationMat        


    