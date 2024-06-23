# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:13:41 2022

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
        self.numTrials = np.shape(arrayVal)[1]
        self.numArms = numArms
        self.explorationMat = np.zeros([self.numTrials])
        self.modelMat = np.zeros([self.numTrials]) 
        
        
    def gradient(self):
        
        learningRate = self.parameters[0]
        traceRate = self.parameters[1]
                    
        #Initialize Reward Array
        rewardBlock = self.arrayVal[1,:]
        choiceBlock = self.arrayVal[0,:]
        
        # Start at 0?
        H = np.zeros(shape=self.numArms) + 10
        
        # Start timer and reward
        time = 1
        baseline = 0.65
        trace = 0
        
        # Loop around trials
        for trial in range(self.numTrials):
     
            #Extract Participant choice
            humanSelect = int(choiceBlock[trial])
            
            if humanSelect == -1:
                
                trialType = -1
                modelChoice = -1
                
            else:
            
                # Calculate Softmax
                num = np.exp(H)
                
                denom = np.sum(num)
                
                sm = num / denom
                
                modelChoice = np.random.choice(np.flatnonzero(sm == sm.max()))
                
                # Reward
                reward = rewardBlock[trial] #* 100
                
                if humanSelect == modelChoice:
                    
                    trialType = 0
                
                else:
                    
                    trialType = 1
                
                
                # Average reward for baseline
                if trace == 0:
                
                    traceRate2 = 0
                
                else:
                
                    traceRate2 = traceRate / trace # modify LR for nonstationary
                
                baseline += (reward - baseline) * traceRate2
                
                # Update trace
                trace = trace + (traceRate*(1 - trace))
                
                one_hot = np.zeros(self.numArms)
                
                one_hot[humanSelect] = 1
                
                # Update Chosen H Values
                if trial == 200:
                
                    baseline = .65
                    H = np.zeros(shape=self.numArms) + 10
                
                else:
                
                    H +=  learningRate * (reward - baseline) * (one_hot - sm)
                                
                # Update Times
                time += 1
            
            #Update Exploration Array
            self.explorationMat[trial] = trialType
            self.modelMat[trial] = modelChoice
          
        return self.explorationMat, self.modelMat        


