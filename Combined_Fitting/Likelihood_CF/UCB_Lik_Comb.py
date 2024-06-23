# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:45:09 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def UCB_Lik_Comb(parameters,
            rewardVal,
            choices,              
            taskParam):
    
    import numpy as np
        
    # %% Task Set up
    numArms_NS = int(taskParam[0])
    numTrials_NS = int(taskParam[1])
    numArms_S = int(taskParam[2])
    numTrials_S = int(taskParam[3])
    numBlocks = int(taskParam[4])
    
    #Initialize Parameters
    tau = parameters[0]
    expParam = parameters[1]
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]

# %%
    expParam2 = expParam
    xi = 1
     
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
     
    # Step count
    n = 0
    # Step count for each arm
    X = np.zeros(shape=numArms_NS)
    N = np.zeros(shape=numArms_NS)
    c = np.zeros(shape=numArms_NS)
    
    history = [] # history of played arms, of size t
    history_bool = None # history of played arms as booleans, of size (t,nbArms)
    reward_history = []
    bound = []
    n_T = None
    UCBValues  = np.zeros(shape=numArms_NS) + .25
    
    for trial in range(numTrials_NS):
                 
         selection = int(choices_NS[trial])
         
         if selection == -1:
             
             liklihoodArray_NS[trial] = 1
             
         else:
             
             #print(block,trial)
             if trial < numArms_NS:
                         
                 UCBValues  = np.zeros(shape=numArms_NS) + .25
                         
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
                     
                     for i in range(numArms_NS):
                         
         
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
                          
                             c[i] = expParam2 * np.sqrt (( xi * min(trial, tau))  / N[i])
                     
         
                 bound = (X+c)
                     
             reward = reward_NS[trial] * 100 #/ 100
         
             history = np.append(history, selection)
             reward_history = np.append(reward_history, reward)
             arm_bool = np.zeros(numArms_NS)
             arm_bool[selection] = 1
             
             if history_bool is None : 
                 
                 history_bool = arm_bool
             
             else : 
                 
                 history_bool = np.vstack((history_bool, arm_bool))
             
             if len(bound) > 0:
                 
                 bound[bound == 0] = .001
             
             if trial < numArms_NS or np.array(reward_recent).size == 0:
                 
                 UCBValues = np.zeros(shape=numArms_NS) + .25
             
             else:
                 
                 UCBValues = bound / sum(bound)
                 
                 # Test
                 #UCBValues = np.exp(bound) / sum(np.exp(bound))
                 
             # Update counts
             n += 1
                             
             # Compute Liklihood
             liklihoodArray_NS[trial] = UCBValues[selection]
     
             if liklihoodArray_NS[trial] <= 0 or np.isnan(liklihoodArray_NS[trial]):
                 liklihoodArray_NS[trial] = 1
    
    # Deal with Zeros
    liklihoodArray_NS[liklihoodArray_NS <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_NS[np.isnan(liklihoodArray_NS)] = 1e+300
            
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))


# %% Stationary
    liklihoodArray_S = np.zeros(shape=[numBlocks, numTrials_S])
    
    
    for block in range(numBlocks):
        
        # Step count
        n = 1
        # Step count for each arm
        stepCount = np.ones(numArms_S)
        # Total mean reward
        mean_reward = 0
        #reward = np.zeros(numTrials)
        # Mean reward for each arm
        qValues = np.zeros(numArms_S) + 1e-5 #this is here to avoid inf values
        #Set Up Arrays
        trialReward = reward_S[block,:]
        choiceBlock = choices_S[block,:]
        
        for trial in range(numTrials_S):
            # Select action according to UCB Criteria
            selection = int(choiceBlock[trial])
            
            
            if selection == -1:
                
                liklihoodArray_S[block,trial] = 1
                
            else:
                        
                uncertainty = expParam * np.sqrt(
                    (np.log(n) / (stepCount +  1e-5)))
            
                #Compute Softmax values
                num = qValues + uncertainty
                denom = sum(qValues + uncertainty)
                
            
                #UCB Array
                if num.all() == 0 and denom == 0:
                    UCBValues = np.zeros(numArms_S)
                else:
                    UCBValues = num / denom
            
                    
                #Get Reward
                #Compute Reward
                reward = trialReward[trial]
              
                # Update counts
                n += 1
                stepCount[selection] += 1
                
                # Update total
                mean_reward = mean_reward + (
                    reward - mean_reward) / n
                
                # Update results for a_k
                qValues[selection] = qValues[selection] + (
                    reward - qValues[selection]) / stepCount[selection]
    
                #Compute Liklihood
                liklihoodArray_S[block,trial] = UCBValues[selection]
    
                if liklihoodArray_S[ block,trial] <= 0:
                    liklihoodArray_S[ block,trial] = 1e-5
                    
    # Deal with Zeros
    liklihoodArray_S[liklihoodArray_S <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_S[np.isnan(liklihoodArray_S)] = 1e+300
    
    #Update Liklihood Values
    liklihoodSum_S = -np.sum(np.log(liklihoodArray_S))
    
    
    # %% Likelihood Junk (sum both)
        
    liklihoodSum = ((liklihoodSum_NS/400) + (liklihoodSum_S/100))*500
    #liklihoodSum = liklihoodSum_NS + liklihoodSum_S
    
    return liklihoodSum 