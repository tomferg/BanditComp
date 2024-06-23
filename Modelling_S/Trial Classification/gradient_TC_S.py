# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:45:37 2022

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
        self.numBlocks = np.shape(arrayVal)[1]
        self.numTrials = np.shape(arrayVal)[2]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numBlocks,self.numTrials])

        
    # def greedy(self):
    #     raise Exception('No Model Ready!')
            
    def gradient(self):
        
        #Initialize Parameters
        learningRate = self.parameters
        
        
        for block in range(self.numBlocks):
    
            # Start at 0?
            H = np.zeros(shape=self.numArms)
            
            # Start timer and reward
            time = 1
            baseline = .5
            
            #Initialize Reward Array
            rewardBlock = self.arrayVal[0,block,:] 
            choiceBlock = self.arrayVal[1,block,:]
        
            for trial in range(self.numTrials):
                
                maxBandit = []
                
                humanSelect = int(choiceBlock[trial])
                
                if trial == 0 or humanSelect == -1:
                    
                    trialType = -1
                    
                else:
                    
                    # Calculate Softmax
                    num = np.exp(H)
                    
                    denom = np.sum(num)
                    
                    sm = num / denom
                    
                    maxBandit = np.argmax(sm)
                    
                    
                    if maxBandit == humanSelect:
                        
                        trialType = 0
                        
                    else:
                        
                        trialType = 1
                    
                    #Compute Reward                    
                    if rewardBlock[trial] == 1:
                        
                        reward = 1
                    
                    else:
                        
                        reward = 0 
                    
                    one_hot = np.zeros(self.numArms)
                    one_hot[humanSelect] = 1
                    
                    # Update Chosen H Values
                    H +=  learningRate *  (reward - baseline) * (one_hot - sm)
                                    
                    # Update Times
                    time += 1
                
                #Update Exploration Array
                self.explorationMat[block,trial] = trialType
              
        return self.explorationMat     
        


    