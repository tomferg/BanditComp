# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:13:40 2022

@author: Tom Ferguson, PhD, University of Alberta
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
        self.numTrials = np.shape(arrayVal)[1]
        self.numArms =numArms
        self.explorationMat = np.zeros([self.numTrials])
        
    def UCB_SW(self):
        
        #Initialize Parameters
        #learningRate = self.parameters[0]
        #expParam = parameters[1]
        #expParam = parameters[1]
        expParam = .1
        tau = self.parameters
        xi = 1
        
        #Initialize Reward Array
        rewardBlock = self.arrayVal[1,:]
        choiceBlock = self.arrayVal[0,:]
        
        # Step count
        n = 0
        # Step count for each arm
        X = np.zeros(shape=self.numArms)
        N = np.zeros(shape=self.numArms)
        c = np.zeros(shape=self.numArms)
    
        history = [] # history of played arms, of size t
        history_bool = None # history of played arms as booleans, of size (t,nbArms)
        reward_history = []
        bound = []
        n_T = None
    
        for trial in range(self.numTrials):
                    
            humanSelect = int(choiceBlock[trial])
            
            if trial == 0 or humanSelect == -1:
                
                trialType = -1
                
            else:
                
                    
                n_T = None
                
                if tau > trial:
                    
                    startTrial = 0
                
                else:
                
                    startTrial = len(history)- (tau+1)
                 
                startTrial = int(startTrial)
                
                n_T = history[startTrial:]
                
                reward_recent = reward_history[startTrial:]
                
                if np.array(reward_recent).size > 0:
                    
                    for i in range(self.numArms):
                        
        
                        temp = np.where(np.array(n_T )==i, True, False)
                        
                        reward_received = reward_recent[temp]
                        
                        N[i] = np.sum(temp)
                        
                        if N[i] == 0:
                            
                            X[i] = 0
                            
                        else:
                        
                            X[i] = (1/N[i]) * np.sum((reward_received))
                    
                        if N[i] == 0:
                        
                            c[i] = c[i] + .01 #np.inf # origin np.inf
                        
                        else:
                         
                            c[i] = expParam * np.sqrt (( xi * min(trial, tau))  / N[i])
                    
                bound = (X+c)
                
                # Find max
                modelBest = np.random.choice(np.flatnonzero(X == X.max()))
                
                
                if humanSelect == modelBest:
                    trialType = 0
                else:
                    trialType = 1
                
                reward = rewardBlock[trial] #/ 100
            
                history = np.append(history, humanSelect)
                reward_history = np.append(reward_history, reward)
                arm_bool = np.zeros(self.numArms)
                arm_bool[humanSelect] = 1
                
                if history_bool is None : 
                    
                    history_bool = arm_bool
                
                else : 
                    
                    history_bool = np.vstack((history_bool, arm_bool))

                                    
                # Update counts
                n += 1
                                
            #Update Exploration Array
            self.explorationMat[trial] = trialType
          
        return self.explorationMat   
