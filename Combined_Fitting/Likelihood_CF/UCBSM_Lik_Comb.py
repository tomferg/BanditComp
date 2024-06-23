# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:25:45 2023

@author: Tom Ferguson, PhD, University of Alberta
"""
def UCBSM_Lik_Comb(parameters,
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
    tempParam = parameters[0]
    expParam = parameters[1]
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]

# %% 
    
    # Likelihood Array
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
        
    #Initialize Arm sample history
    selectCount = np.ones(shape=numArms_NS)
    
    # Needed so trial one isn't 0 in the SM equation
    n = 1
    
    #Initialize Q Values
    qValue = np.zeros(shape=numArms_NS)+1e-5
    
    # Initialize Uncertainty
    uncert = np.zeros(shape=numArms_NS)+1e-5
    
    # LR Parameter
    lrParam = .4
    
    for tCt in range(numTrials_NS):
        
        # Find selection and reward
        selection = int(choices_NS[tCt])
        reward = reward_NS[tCt]
        
        if selection == -1:
            
            liklihoodArray_NS[tCt] = 1
            
        else:
        
            # Version from UCB1
            #uncert = expParam * np.sqrt(np.log(n) / (selectCount))
            
            # Version from S&K 2015
            lastCount = np.zeros(shape=numArms_NS)
            for cTc in range(numArms_NS):
                x = np.array(np.where(choices_NS[0:tCt] == cTc))
                if x.size == 0:
                    lastCount[cTc] = 0    
                else:
                    lastCount[cTc] = np.amax(x)
            uncert = (expParam * (n - lastCount)) / 100
            
            # Calculate Softmax Probabilities 
            num = np.exp(np.multiply(qValue,tempParam) + uncert)
            denom = sum(np.exp(np.multiply(qValue,tempParam) + uncert));
            
            # Actual Softmax
            softmaxResult = num/denom
            
            # Selection
            selection = int(choices_NS[tCt])
            
            # Reward
            reward = reward_NS[tCt]
            
            #Update Arm Count!
            selectCount[selection] += 1
            n += 1
            
            #Compute Prediction Error
            predError = reward - qValue[selection]
            
            #Update Reward - Non Stationary
            qValue[selection] = qValue[selection] + lrParam * predError
            
            # Assign Likelihood
            liklihoodArray_NS[tCt] = softmaxResult[selection]
            

    # Deal with Zeros
    liklihoodArray_NS[liklihoodArray_NS <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_NS[np.isnan(liklihoodArray_NS)] = 1e+300
    
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))


# %% Stationary

    # #Create Liklihood Array
    # liklihoodArray_S = np.zeros(shape=[numBlocks, numTrials_S])
    
    #Initialize Liklihood array
    liklihoodArray_S = np.zeros(shape=[numBlocks,numTrials_S])
    
    
    for bCt in range(numBlocks):
    
        #Initialize Arm sample history
        selectCount = np.ones(shape=numArms_S)
        
        # Needed so trial one isn't 0 in the SM equation
        n = 1
        
        #Initialize Q Values
        qValue = np.zeros(shape=numArms_S)+1e-3
        
        # Initialize Uncertainty
        uncert = np.zeros(shape=numArms_S)+1e-3
    
        for tCt in range(numTrials_S):
            
            # Find selection and reward
            selection = int(choices_S[bCt, tCt])
                
            if selection == -1:
                
                liklihoodArray_S[bCt, tCt] = 1
                
            else:
            
                # Version from UCB1
                uncert = expParam * np.sqrt(np.log(n) / (selectCount))
                
                # Calculate Softmax Probabilities 
                num = np.exp(np.multiply(qValue,tempParam) + uncert)
                denom = sum(np.exp(np.multiply(qValue,tempParam) + uncert));
                
                # Actual Softmax
                softmaxResult = num/denom
                
                # Selection
                #selection = int(choices_LL[bCt, tCt])
                
                # Reward
                reward = reward_S[bCt, tCt]
                
                #Update Arm Count!
                selectCount[selection] += 1
                n += 1
                
                #Compute Prediction Error
                predError = reward - qValue[selection]
                
                #Update Reward - Non Stationary
                qValue[selection] = qValue[selection] + lrParam * predError
                #qValue[selection] = qValue[selection] + (1 / (selectCount[selection])) * predError
    
                # Assign Likelihood
                liklihoodArray_S[bCt, tCt] = softmaxResult[selection]
            
    
    # Deal with Zeros
    liklihoodArray_S[liklihoodArray_S <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_S[np.isnan(liklihoodArray_S)] = 1e+300
    
    #Update Liklihood Values
    liklihoodSum_S = -np.sum(np.log(liklihoodArray_S))


# %% Likelihood Junk (sum both)
    
    #liklihoodSum = liklihoodSum_NS + liklihoodSum_S
    liklihoodSum = ((liklihoodSum_NS/400) + (liklihoodSum_S/100))*500

    return liklihoodSum 