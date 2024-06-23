# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:58:29 2022

@author: Tom Ferguson, PhD, University of Alberta
"""

def UCB_Slide_Lik_NS(parameters,
         rewardVal,
         choices,
         trialParam):
    
    import numpy as np
    
    numArms = int(trialParam[0])
    numTrials = int(trialParam[1])
        
    #Initialize Parameters
    #learningRate = self.parameters[0]
    #expParam = parameters[1]
    #expParam = parameters[1]
    expParam = .1
    tau= parameters
    xi = 1
    
    liklihoodArray = np.zeros(shape=[numTrials])
    
    # Step count
    n = 0
    # Step count for each arm
    X = np.zeros(shape=numArms)
    N = np.zeros(shape=numArms)
    c = np.zeros(shape=numArms)

    history = [] # history of played arms, of size t
    history_bool = None # history of played arms as booleans, of size (t,nbArms)
    reward_history = []
    bound = []
    n_T = None
    UCBValues  = np.zeros(shape=numArms) + .25

    for trial in range(numTrials):
                
        selection = int(choices[trial])
        
        if selection == -1:
            
            liklihoodArray[trial] = 1
            
        else:
            
            #print(block,trial)
            if trial < numArms:
                        
                UCBValues  = np.zeros(shape=numArms) + .25
                        
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
                    
                    for i in range(numArms):
                        
        
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
                    
            reward = rewardVal[trial] * 100 #/ 100
        
            history = np.append(history, selection)
            reward_history = np.append(reward_history, reward)
            arm_bool = np.zeros(numArms)
            arm_bool[selection] = 1
            
            if history_bool is None : 
                
                history_bool = arm_bool
            
            else : 
                
                history_bool = np.vstack((history_bool, arm_bool))
            
            if len(bound) > 0:
                
                bound[bound == 0] = .001
            
            if trial < numArms or np.array(reward_recent).size == 0:
                
                UCBValues = np.zeros(shape=numArms) + .25
            
            else:
                
                UCBValues = bound / sum(bound)
                
                # Test
                #UCBValues = np.exp(bound) / sum(np.exp(bound))
                
            # Update counts
            n += 1
                            
            # Compute Liklihood
            liklihoodArray[trial] = UCBValues[selection]
    
            if liklihoodArray[trial] <= 0 or np.isnan(liklihoodArray[trial]):
                liklihoodArray[trial] = 1e300
                
    
    liklihoodSum = -np.sum(np.log(liklihoodArray))
          
    return liklihoodSum
