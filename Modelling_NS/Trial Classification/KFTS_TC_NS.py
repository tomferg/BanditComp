# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:19:36 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

import numpy as np
# # UNCLEAR IF NEEDED - CLEAN LATER?!?
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
mvtnorm = importr('mvtnorm')
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
r = robjects.r
numpy2ri.activate()
pandas2ri.activate()
    
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
        
        
    def KFTS(self):
        
        # Parameters
        sigXi = self.parameters[0]
        sigEps = self.parameters[1]
        
        mu0 = 5
        v0 = 100
                
        #Initialize Reward Array
        rewardBlock = self.arrayVal[1,:]
        choiceBlock = self.arrayVal[0,:]
        
        m = np.zeros(shape=[self.numTrials+1, self.numArms]) + mu0
        v = np.zeros(shape=[self.numTrials+1, self.numArms]) + v0
        
        for tCt in range(self.numTrials):
                        
            humanSelect = int(choiceBlock[tCt])
            
            reward = rewardBlock[tCt]*100
            
            if humanSelect == -1:
                
                # Update Mean and Variances
                m[tCt+1,] = m[tCt,]
                v[tCt+1,] = v[tCt,]
                
            else:
            
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=self.numArms)
                
                # Calculate kalman Gain
                kt[humanSelect] = (v[tCt, humanSelect] + sigXi) / (v[tCt, humanSelect] + sigXi + sigEps)
                
                # Update Mean and Variances
                m[tCt+1] = m[tCt] + kt*(reward - m[tCt])
                v[tCt+1] = (1 - kt) * (v[tCt] + sigXi)
                
        
        # Set up matrix for multiplication
        A1 = np.array([[1, -1, 0 ,0],[1,0,-1,0], [1,0,0,-1]])
        
        # Create Array to assign junk - NOTE COLUMN 2nd here!
        A = np.zeros(shape =[4,3,4])
        
        # Assign A1 to the A array
        A[0, :, :] = A1
        
        # Shuffle each and just assign to the matrix
        idx1 = [1, 0, 2, 3]
        idx2 = [1, 2, 0, 3]
        idx3 = [1, 2, 3, 0]
        
        A[1, : ,:] = A1[:, idx1]
        A[2, : ,:] = A1[:, idx2]
        A[3, : ,:] = A1[:, idx3]
            
        # initialize a matrix for the choice probabilities
        prob = np.zeros(shape =[self.numTrials, self.numArms])+1e-3
        
        # Loop across trials?
        for tCt in range(self.numTrials):
            
            # Loop across Options
            for aCt in range(self.numArms):
                
                # newM is the mean vector of the difference scores
                newM = A[aCt, :, :] @ m[tCt, ]
                
                # Convert to numeric vector?
                newM = robjects.vectors.FloatVector(newM)
                
                # newV is the diag
                newV = A[aCt, :, :] @ r.diag(v[tCt,]) @ np.transpose(A[aCt, :, :])
                
                # Create thing so MIWA goes longer?
                aSG = mvtnorm.Miwa(steps = 1280)
                
                # Assign Junk
                prob[tCt, aCt] = mvtnorm.pmvnorm(lower=r.c(0,0,0), mean = newM, sigma = newV, algorithm=aSG)
                
                # Check if too small?
                for cTc in range(self.numArms):
                    
                    if prob[tCt, cTc] <= 1e-4:
                        
                        prob[tCt, cTc] = .001
        
        # Find Max Arm on each trial
        modelMax = np.argmax(prob, axis=1)
        
        # Compare Max Model Choices to Humans
        for trial in range(self.numTrials):
            
            if trial == 0 or int(choiceBlock[trial]) == -1:
                
                trialType = -1
            
            
            elif int(modelMax[trial]) == int(choiceBlock[trial]):
            
                trialType = 0
            
            else:
                
                trialType = 1
                
            #Update Exploration Array
            self.explorationMat[trial] = trialType
        
        
        
              
        return self.explorationMat     
        