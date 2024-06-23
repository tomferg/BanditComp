# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:08:40 2022

@author: Tom Ferguson, PhD, University of Alberta
"""

def gradient_Lik_NS(parameters,
            rewardVal,
            choices,
            trialParam):
    
    import numpy as np

    # Initialize Parameters
    numArms = int(trialParam[0])
    numTrials = int(trialParam[1])
    
    # Model Parameters
    learningRate = parameters[0]
    traceRate = parameters[1]
    
    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numTrials])
    
    # Start at 0?
    H = np.zeros(shape=numArms)+10
    
    # Start timer and reward
    time = 1
    baseline = .65
    trace = 0
    
    # Loop around trials
    for trial in range(numTrials):
 
        #Extract Participant choice
        selection = int(choices[trial])
        
        if selection == -1:
            
            liklihoodArray[trial] = 1
            
        else:
        
            # Calculate Softmax
            num = np.exp(H)
            
            denom = np.sum(num)
            
            sm = num / denom
            
            # Reward
            reward = rewardVal[trial]  / 100
            
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
            one_hot = np.zeros(numArms)
            one_hot[selection] = 1
            
            # Update Chosen H Values
            if trial == 200:
                
                baseline = .65
                H = np.zeros(shape=numArms) + 10
                
            else:
                
                H +=  learningRate * (reward - baseline) * (one_hot - sm)
                            
            # Update Times
            time += 1
            
            #Compute Liklihood
            liklihoodArray[trial] = sm[selection]
            
            if liklihoodArray[trial] <= 0 or np.isnan(liklihoodArray[trial]):
                liklihoodArray[trial] = 1e300
    
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))

    return liklihoodSum



