# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:52:16 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def softmax_Lik_Comb(parameters,
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
    temperature = parameters[0]
    learningRate = parameters[1]
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]
    
# %% Non-stationary
    #Initialize Liklihood array
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
    
    #Initialize samples
    selectCount = np.ones(shape=numArms_NS)
    
    #Initialize Q Values
    qValue = np.zeros(shape=numArms_NS)+1e-5
    
    #Loop around trials
    for trial in range(numTrials_NS):
 
        #Extract Participant choice
        selection = int(choices_NS[trial])
        
        
        if selection == -1:
            
            liklihoodArray_NS[trial] = 1
            
        else:
        
            #Compute Softmax values
            num = np.exp( np.multiply(qValue,temperature))
            denom = sum(np.exp(np.multiply(qValue,temperature)))
            
            #Find softmax result
            softmaxResult = num/denom
            
            #Update Arm Count!
            selectCount[selection] += 1  
            
            #Compute Reward
            reward = reward_NS[trial]
            
            #Compute Prediction Error
            predError = reward - qValue[selection]
            
            #Update Reward - Non Stationary
            qValue[selection] = qValue[selection] + learningRate * predError              
            
            # #Note this is needed to avoid overly large values which numpy can't handle
            # if qValue[selection] > 1:
            #     qValue[selection] = 1
            # if qValue[selection] < -1:
            #     qValue[selection] = -1
            
            #Compute Liklihood
            liklihoodArray_NS[trial] = softmaxResult[selection]
            
            if liklihoodArray_NS[trial] <= 0:
                liklihoodArray_NS[trial] = 1e300
                
    # Deal with Zeros
    liklihoodArray_NS[liklihoodArray_NS <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_NS[np.isnan(liklihoodArray_NS)] = 1e+300
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))


# %% Stationary
    
    #Initialize Liklihood array
    liklihoodArray_S = np.zeros(shape=[ numBlocks,numTrials_S])
    
        
    for block in range(numBlocks):
    
        #Initialize samples
        selectCount = np.ones(shape=numArms_S)
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms_S)+.00001
        
        #Set Up Arrays
        trialReward = reward_S[block,:]
        choiceBlock = choices_S[block,:]
        
        #Loop around trials
        for trial in range(numTrials_S):
    
            #Extract Participant choice
            selection = int(choiceBlock[trial])
            
            
            if selection == -1:
                
                liklihoodArray_S[block,trial] = 1
                
            else:
            
                #Compute Softmax values
                num = np.exp( np.multiply(qValue,temperature))
                denom = sum(np.exp(np.multiply(qValue,temperature)))
                
                #Find softmax result
                softmaxResult = num/denom
                
                #Update Arm Count!
                selectCount[selection] += 1  
                
                #Compute Reward
                reward = trialReward[trial]
                
                #Compute Prediction Error
                predError = reward - qValue[selection]
                
                #Update Reward - Non Stationary                    
                qValue[selection] = qValue[selection] + learningRate  * predError #(1 / selectCount[selection]) * predError                
                    
                #Compute Liklihood
                liklihoodArray_S[block,trial] = softmaxResult[selection]
                
                if liklihoodArray_S[block,trial] <= 0:
                    liklihoodArray_S[block,trial] = 1e300
     
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