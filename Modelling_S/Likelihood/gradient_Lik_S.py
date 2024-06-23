# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:11:26 2022

@author: Tom Ferguson, PhD, University of Alberta
"""
import numpy as np

#Action Selection
    
'''
Code runs different action selection approaches

Arguments:
    arg(1) = reward values, needs to be Arms by Trials
    arg(2) = parameters: learningRate & eGreedy exploration
    arg(3) = continuous or binomial reward
    
Returns:

'''  
        
def gradient(parameters,
         rewardVal,
         choices,
         trialParam):

    # Initialize Parameters
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])

    # Model Parameters
    learningRate = parameters
    
    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numBlocks, numTrials])
    
    for block in range(numBlocks):
    
        # Start at 0?
        H = np.zeros(shape=numArms)
        
        # Start timer and reward
        time = 1
        baseline = 4
        
        # Loop around trials
        for trial in range(numTrials):
     
            #Extract Participant choice
            selection = int(choices[block, trial])
            
            if selection == -1:
                
                liklihoodArray[block, trial] = 1
                
            else:
            
                # Calculate Softmax
                num = np.exp(H)
                
                denom = np.sum(num)
                
                sm = num / denom
                
                # Reward
                reward = rewardVal[block, trial]  #/ 100

                baseline += (reward - baseline) / time
                
                one_hot = np.zeros(numArms)
                one_hot[selection] = 1
                
                # Update Chosen H Values
                H +=  learningRate *  (reward - baseline) * (one_hot - sm)
                                
                # Update Times
                time += 1
        
                #Compute Liklihood
                liklihoodArray[block, trial] = sm[selection]
                
                if liklihoodArray[block, trial] <= 0:
                    liklihoodArray[block, trial] = 1
        
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray))

    return liklihoodSum



