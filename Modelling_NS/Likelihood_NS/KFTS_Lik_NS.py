# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:51:59 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def KFTS_Lik(parameters,
            rewardVal,
            choices,              
            trialParam):
    
    import numpy as np
    
    # # # UNCLEAR IF NEEDED - CLEAN LATER?!?
    # import rpy2.robjects as robjects
    # from rpy2.robjects.packages import importr
    # # utils = importr('utils')
    # # #base = importr('base')
    # # #utils.install_packages('mvtnorm')
    # mvtnorm = importr('mvtnorm')
    # from rpy2.robjects import pandas2ri
    # from rpy2.robjects import numpy2ri
    # numpy2ri.activate()
    # pandas2ri.activate()

    # # This imports crap into R - might be quick than defining everything
    # r = robjects.r
    
    # # Isolate Parameters?
    mu0 = 5
    v0 = 100
    sigXi = np.exp(parameters[0])
    sigEps = np.exp(parameters[1])
    numSamples = 10000
    
    numTrials = trialParam[1]
    numArms = trialParam[0]
    
    # Extract Posterior means and variances from KF calculation
    # nT = numTrials
    # nO = numArms
    m = np.zeros(shape=[numTrials+1, numArms]) + mu0
    v = np.zeros(shape=[numTrials+1, numArms]) + v0
    
    # Loop around trials
    for tCt in range(0, len(choices)):
        
        # Find selection and reward
        selection = int(choices[tCt])
        reward = rewardVal[tCt]*100
        
        if selection == -1:
            
            # Update Mean and Variances
            m[tCt+1,] = m[tCt,]
            v[tCt+1,] = v[tCt,]
            
        else:
        
            # Zero Kalman Gain Set up
            kt = np.zeros(shape=numArms)
            
            # Calculate kalman Gain
            kt[selection] = (v[tCt, selection] + sigXi) / (v[tCt, selection] + sigXi + sigEps)
            
            # Update Mean and Variances
            m[tCt+1] = m[tCt] + kt*(reward - m[tCt])
            v[tCt+1] = (1 - kt) * (v[tCt] + sigXi)
        
    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numTrials])
    
    # # Set up matrix for multiplication
    # A1 = np.array([[1, -1, 0 ,0],[1,0,-1,0], [1,0,0,-1]])
    
    # # Create Array to assign junk - NOTE COLUMN 2nd here!
    # A = np.zeros(shape =[4,3,4])
    
    # # Assign A1 to the A array
    # A[0, :, :] = A1
    
    # # Shuffle each and just assign to the matrix
    # idx1 = [1, 0, 2, 3]
    # idx2 = [1, 2, 0, 3]
    # idx3 = [1, 2, 3, 0]
    
    # A[1, : ,:] = A1[:, idx1]
    # A[2, : ,:] = A1[:, idx2]
    # A[3, : ,:] = A1[:, idx3]
    
    # # initialize a matrix for the choice probabilities
    # prob = np.zeros(shape =[numTrials, numArms])+1e-3
    
    # # Loop across trials?
    # for tCt in range(0, len(choices)):
    #     # Loop across Options
    #     for aCt in range(0, numArms):
            
    #         # newM is the mean vector of the difference scores
    #         newM = A[aCt, :, :] @ m[tCt, ]
            
    #         # Convert to numeric vector?
    #         newM = robjects.vectors.FloatVector(newM)
            
    #         # newV is the diag
    #         newV = A[aCt, :, :] @ r.diag(v[tCt,]) @ np.transpose(A[aCt, :, :])
            
    #         # Create thing so MIWA goes longer?
    #         aSG = mvtnorm.Miwa(steps = 1280)
            
    #         # Assign Junk
    #         prob[tCt, aCt] = mvtnorm.pmvnorm(lower=r.c(0,0,0), mean = newM, sigma = newV, algorithm=aSG)
            
    #         # Check if too small?
    #         for cTc in range(0, numArms):
                
    #             if prob[tCt, cTc] <= 1e-4:
                    
    #                 prob[tCt, cTc] = .001
                    
            
            
    # initialize a matrix for the choice probabilities
    prob = np.zeros(shape =[numTrials, numArms])+1e-3

    for tCt in range(0, len(choices)):
        
        # ALTERNATIVE APPROACH...
        #Find Prob for sampling          
        # Set up prior and sample from Normal distribution
        priorSamp = np.random.normal(loc=m[tCt, :], scale=np.sqrt(v[tCt, :]),
                                    size = ( numSamples, numArms))
        # Determine Max Arms
        maxArm = np.argmax(priorSamp,axis=1)
        # Count Max 
        unique, counts = np.unique(maxArm, return_counts=True)  
        # Divide by number of samples
        prob[tCt, unique] = counts / numSamples
        
        #liklihoodArray[tCt] = prob[selection]
    
    
    # Convert choices in integer
    choices_int = choices.astype(int)
    
    #Compute Liklihood
    liklihoodArray = prob[np.indices(choices_int.shape)[0], choices_int]
    
    liklihoodArray[choices_int == -1] = 1
    
    #if liklihoodArray <= 0:
    liklihoodArray[liklihoodArray <= 0] = 1e+300
    
    # Sum LL
    liklihoodSum = -np.sum(np.log(liklihoodArray))
    
    return liklihoodSum
