# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 23:07:47 2022

@author: Tom
"""

import numpy as np

#Action Selection
class trialClassification:
    
    '''
    Code classifies human trial by trial data as exploration or exploitation
    Runs a UCB-1 approach
    
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
        self.numArms =numArms
        self.explorationMat = np.zeros([self.numBlocks,self.numTrials])
        
        
    def UCB1(self):
        
        #Initialize Parameters
        expParam = self.parameters
        
        for block in range(self.numBlocks):
            
            rewardBlock = self.arrayVal[0,block,:]
            choiceBlock = self.arrayVal[1,block,:]

           
            # Step count
            n = 1
            # Step count for each arm
            stepCount = np.ones(self.numArms)
            # Total mean reward
            #mean_reward = 0
            # Mean reward for each arm
            qValues = np.zeros(self.numArms) + 1e-5
                       
            for trial in range(self.numTrials):
                
                maxBandit = []
                humanSelect = int(choiceBlock[trial])
                
                if trial == 0 or humanSelect == -1:
                    
                    trialType = -1
                    
                else:
                    
                    #Select action according to UCB Criteria
                    maxBandit = np.argmax(qValues + expParam * np.sqrt(
                            (np.log(n)) / stepCount))
                    
                    
                    if maxBandit == humanSelect:
                    
                        trialType = 0
                        
                    else:
                        
                        trialType = 1
                    
                    # Update counts
                    n += 1
                    stepCount[humanSelect] += 1
                    
                    # # Update total
                    # mean_reward = mean_reward + (
                    #     reward - mean_reward) / n                
                
                    #Compute Reward                    
                    if rewardBlock[trial] == 1:
                        
                        reward = 1
                    
                    else:
                        reward = 0 
                    
                    #Compute Prediction Error
                    predError = reward - qValues[humanSelect]
                                        
                    qValues[humanSelect] = qValues[humanSelect] + (1 / stepCount[humanSelect]) * predError
    
                    
                #Update Exploration Array
                self.explorationMat[block,trial] = trialType
              
        return self.explorationMat   

    def UCBSlide(self):
        
        raise Exception('No Model Ready!')


    