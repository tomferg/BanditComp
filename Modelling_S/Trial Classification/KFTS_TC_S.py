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
    
    def KFTS(self):
                        
        # Isolate Parameters
        sigXi = self.parameters
        
        mu0 = 5
        v0 = 100
        numSamples = 1
        sigEps = .001
        
        # Extract Posterior means and variances from KF calculation
        m = np.zeros(shape=[self.numBlocks, self.numTrials+1, self.numArms]) + mu0
        v = np.zeros(shape=[self.numBlocks, self.numTrials+1, self.numArms]) + v0
        
        for bCt in range(self.numBlocks):
            
            #Initialize Reward Array
            rewardBlock = self.arrayVal[0, bCt, :]
            choiceBlock = self.arrayVal[1, bCt, :]
        
            for tCt in range(self.numTrials):
                
                modelChoice = []
                
                humanSelect = int(choiceBlock[tCt])
                
                    
                if tCt == 0 or humanSelect == -1:
                    
                    trialType = -1
                    
                    # Update Mean and Variances
                    m[bCt, tCt+1] = m[bCt, tCt]
                    v[bCt, tCt+1] = v[bCt, tCt]
                    
                else:
                    
                    #Compute Reward                    
                    if rewardBlock[tCt] == 1:
                        
                        reward = 1
                    
                    else:
                        reward = 0 
                    
                    
                    #prob = np.zeros(shape=self.numArms)
                                
                    # Zero Kalman Gain Set up
                    kt = np.zeros(shape=self.numArms)
                    
                    # Calculate kalman Gain
                    kt[humanSelect] = (v[bCt, tCt, humanSelect] + sigXi) / (v[bCt, tCt, humanSelect] + sigXi + sigEps)
                                
                    # Update Mean and Variances
                    m[bCt, tCt+1] = m[bCt, tCt] + kt*(reward - m[bCt, tCt])
                    v[bCt, tCt+1] = (1 - kt) * (v[bCt, tCt] + sigXi)
                                
                    #Find Prob for sampling          
                    # Set up prior and sample from Normal distribution
                    priorSamp = np.random.normal(loc=m[bCt, tCt, :], scale=np.sqrt(v[bCt, tCt, :]),
                                                size = ( numSamples, self.numArms))
                    # Determine Max Arms
                    maxArm = np.argmax(priorSamp,axis=1)
                    # # Count Max 
                    # unique, counts = np.unique(maxArm, return_counts=True)  
                    # # Divide by number of samples
                    # prob[unique] = counts / numSamples
                        
                    #If not use Softmax
                    modelChoice = maxArm
                    
                    if modelChoice == humanSelect:
                        
                        trialType = 0
                        
                    else:
                        
                        trialType = 1                    
        
                    #Update Exploration Array
                    self.explorationMat[bCt, tCt] = trialType
          
        return self.explorationMat  