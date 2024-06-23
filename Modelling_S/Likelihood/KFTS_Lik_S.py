# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:58:31 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def KFTS_Lik(parameters, 
             rewards_LL, 
             choices_LL, 
             trialParam):

    
    import numpy as np
    
    # # Isolate Parameters?
    mu0 = 5
    v0 = 100
    numSamples = 1000
    sigXi = parameters
    sigEps = .001
    
    # Other junk
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
    

    # Extract Posterior means and variances from KF calculation
    m = np.zeros(shape=[numBlocks, numTrials+1, numArms]) + mu0
    v = np.zeros(shape=[numBlocks, numTrials+1, numArms]) + v0
    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numBlocks, numTrials])
        
    for block in range(numBlocks):
            
        for trial in range(numTrials):

            # Find selection and reward
            selection = int(choices_LL[block, trial])
                
            if selection == -1:
                
                liklihoodArray[block, trial] = 1
                
                # Calculate kalman Gain
                #kt[selection] = (v[bCt, tCt, selection] + sigXi) / (v[bCt, tCt, selection] + sigXi + sigEps)
                            
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial]
                v[block, trial+1] = v[block, trial]
                
            else:
            
                prob = np.zeros(shape=numArms)
        
                reward = rewards_LL[block, trial]
                
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms)
                
                # Calculate kalman Gain
                kt[selection] = (v[block, trial, selection] + sigXi) / (v[block, trial, selection] + sigXi + sigEps)
                            
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial] + kt*(reward - m[block, trial])
                v[block, trial+1] = (1 - kt) * (v[block, trial] + sigXi)
                            
                #Find Prob for sampling          
                # Set up prior and sample from Normal distribution
                priorSamp = np.random.normal(loc=m[block, trial, :], scale=np.sqrt(v[block, trial, :]),
                                            size = ( numSamples, numArms))
                # Determine Max Arms
                maxArm = np.argmax(priorSamp,axis=1)
                # Count Max 
                unique, counts = np.unique(maxArm, return_counts=True)  
                # Divide by number of samples
                prob[unique] = counts / numSamples
                
                liklihoodArray[block, trial] = prob[selection]
        
    #if liklihoodArray <= 0:
    liklihoodArray[liklihoodArray <= 0] = 1e-300
    # Sum LL
    liklihoodSum = -np.sum(np.log(liklihoodArray))

    return liklihoodSum