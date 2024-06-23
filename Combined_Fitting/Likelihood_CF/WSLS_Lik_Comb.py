# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:35:56 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def WSLS_Lik_Comb(parameters,
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
    winStay = parameters[0]
    loseShift = parameters[1]
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]

# %%
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
     
    #Set Up Arrays
    selection = []
         
    for trial in range(numTrials_NS):
                 
     if trial == 0:
         
         liklihoodArray_NS[trial] = 1
         
     else:
         
         if reward_NS[trial-1] > .50:
             
             if choices_NS[trial] == choices_NS[trial-1]:
                 
                 liklihoodArray_NS[trial] = winStay
                 
             else:
                 
                 liklihoodArray_NS[trial] = (1 - winStay) / (numArms_NS - 1)
             
             
         else:
             
             if choices_NS[trial] != choices_NS[trial-1]:
                 
                 liklihoodArray_NS[trial] = loseShift
                 
             else:
                 
                 liklihoodArray_NS[trial] = (1 - loseShift) / (numArms_NS - 1)
             
                        
     if liklihoodArray_NS[trial] <= 0:
         liklihoodArray_NS[trial] = 1e-5
    
    # Deal with Zeros
    liklihoodArray_NS[liklihoodArray_NS <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_NS[np.isnan(liklihoodArray_NS)] = 1e+300
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))
    
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))

# %%

    liklihoodArray_S = np.zeros(shape=[numBlocks, numTrials_S])
    
    
    for block in range(numBlocks):
        
        #Set Up Arrays
        trialReward = reward_S[block,:]
        choiceBlock = choices_S[block,:]
        p = np.zeros(shape = 2)
        selection = []
        
        for trial in range(numTrials_S):
                        
            if trial == 0 and block == 0:
                
                p = [.5,.5]
                
            else:
                
                if trialReward[trial-1] == 1:
                
                    p = (winStay / numArms_S) * np.ones(numArms_S)
                    p[selection] = 1-winStay/numArms_S
                        
                else:
                    
                    p = (loseShift /numArms_S) * np.ones(numArms_S)
                    p[selection] = 1- loseShift / numArms_S
                    
            
            selection = int(choiceBlock[trial])
           
            liklihoodArray_S[ block,trial] = p[selection]             
                
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