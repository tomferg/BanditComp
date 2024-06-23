# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:07:08 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def Gradient_Lik_Comb(parameters,
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
    learningRate = parameters[0]
    traceRate = parameters[1]
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]

# %% 
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
    
    # Start at 0?
    H = np.zeros(shape=numArms_NS)+10
    
    # Start timer and reward
    time = 1
    baseline = .5
    trace = 0
    
    # Loop around trials
    for trial in range(numTrials_NS):
 
        #Extract Participant choice
        selection = int(choices_NS[trial])
        
        if selection == -1:
            
            liklihoodArray_NS[trial] = 1
            
        else:
        
            # Calculate Softmax
            num = np.exp(H)
            
            denom = np.sum(num)
            
            sm = num / denom
            
            # Reward
            reward = reward_NS[trial]  / 100
            
            # Average reward for baseline
            if trace == 0:
                
                traceRate2 = 0
            
            else:
        
                traceRate2 = traceRate / trace # modify LR for nonstationary
                
            baseline += (reward - baseline) * traceRate2
            
            # Update trace
            trace = trace + (traceRate*(1 - trace))
            
            # Average reward for baseline
            #baseline += (reward - baseline) / time
            one_hot = np.zeros(numArms_NS)
            one_hot[selection] = 1
            
            # Update Chosen H Values
            if trial == 200:
                
                baseline = .5
                H = np.zeros(shape=numArms_NS) + 10
                
            else:
                
                H +=  learningRate * (reward - baseline) * (one_hot - sm)
                            
            # Update Times
            time += 1
            
            # else:
 
            #     # Test
            #     sm2 = H + abs(min(H))
            #     sm3= (sm2) / sum(sm2)
            
            #Compute Liklihood
            liklihoodArray_NS[trial] = sm[selection]
            
            if liklihoodArray_NS[trial] <= 0:
                liklihoodArray_NS[trial] = 1
    
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))


# %% Stationary
    liklihoodArray_S = np.zeros(shape=[numBlocks, numTrials_S])
    
    
        
    for block in range(numBlocks):
    
        # Start at 0?
        H = np.zeros(shape=numArms_S)
        
        # Start timer and reward
        time = 1
        baseline = 4
        
        # Loop around trials
        for trial in range(numTrials_S):
     
            #Extract Participant choice
            selection = int(choices_S[block, trial])
            
            if selection == -1:
                
                liklihoodArray_S[block, trial] = 1
                
            else:
            
                # Calculate Softmax
                num = np.exp(H)
                
                denom = np.sum(num)
                
                sm = num / denom
                
                # Reward
                reward = reward_S[block, trial]  #/ 100

                baseline += (reward - baseline) / time
                
                one_hot = np.zeros(numArms_S)
                one_hot[selection] = 1
                
                # Update Chosen H Values
                H +=  learningRate *  (reward - baseline) * (one_hot - sm)
                                
                # Update Times
                time += 1
        
                #Compute Liklihood
                liklihoodArray_S[block, trial] = sm[selection]
                
                if liklihoodArray_S[block, trial] <= 0:
                    liklihoodArray_S[block, trial] = 1
    

    #Update Liklihood Values
    liklihoodSum_S = -np.sum(np.log(liklihoodArray_S))


# %% Likelihood Junk (sum both)
    
    liklihoodSum = ((liklihoodSum_NS/400) + (liklihoodSum_S/100))*500
    #liklihoodSum = liklihoodSum_NS + liklihoodSum_S
    
    return liklihoodSum 