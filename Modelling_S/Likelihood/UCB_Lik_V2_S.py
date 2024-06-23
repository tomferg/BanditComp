# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:57:48 2022

@author: Tom

"""

def UCB1_Lik(parameters,
         rewardVal,
         choices,
         enviroType,
         trialParam,
         rewardType):
    
    import numpy as np
    
    numArms = int(trialParam[0])
    numTrials = int(trialParam[2])
    numBlocks = int(trialParam[1])
        
    #Initialize Parameters
    expParam = parameters
    
    liklihoodArray = np.zeros(shape=[numBlocks,numTrials])
    
    for block in range(numBlocks):
        
        # Step count
        n = 1
        # Step count for each arm
        stepCount = np.ones(numArms)
        # Total mean reward
        mean_reward = 0
        #reward = np.zeros(numTrials)
        # Mean reward for each arm
        qValues = np.zeros(numArms) + 1e-5 #this is here to avoid inf values
        #Set Up Arrays
        trialReward = rewardVal[block,:]
        choiceBlock = choices[block,:]
        
        for trial in range(numTrials):
            # Select action according to UCB Criteria
            selection = int(choiceBlock[trial])
            
            
            if selection == -1:
                
                liklihoodArray[block,trial] = 1
                
            else:
                        
                uncertainty = expParam * np.sqrt(
                    (np.log(n) / (stepCount +  1e-5)))
            
                #Compute Softmax values
                num = qValues + uncertainty
                denom = sum(qValues + uncertainty)
                
            
                #UCB Array
                if num.all() == 0 and denom == 0:
                    UCBValues = np.zeros(numArms)
                else:
                    UCBValues = num / denom
            
                    
                #Get Reward
                #Compute Reward
                if rewardType == "Points":
                    reward = trialReward[trial] / 100
                else:
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
                liklihoodArray[block,trial] = UCBValues[selection]
    
                if liklihoodArray[block,trial] <= 0:
                    liklihoodArray[block,trial] = 1e300
                

    liklihoodSum = -np.sum(np.log(liklihoodArray))
          
    return liklihoodSum


