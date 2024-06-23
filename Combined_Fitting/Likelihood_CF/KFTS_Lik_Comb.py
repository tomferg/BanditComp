# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:42:55 2023

@author: Tom Ferguson, PhD, University of Alberta
"""

def KFTS_Lik_Comb(parameters,
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
    sigXi = parameters[0]
    sigEps = parameters[1]
    
    # Isolate Reward and Choices for testing
    reward_NS = rewardVal[0]
    choices_NS = choices[0]
    
    reward_S = rewardVal[1]
    choices_S = choices[1]
    
# %% 
    
    # Likelihood Array
    liklihoodArray_NS = np.zeros(shape=[numTrials_NS])
    
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

    # This imports crap into R - might be quick than defining everything
    #r = robjects.r
    
    # # Isolate Parameters?
    mu0 = 5
    v0 = 500
    numSamples = 1000
    

    # Extract Posterior means and variances from KF calculation
    # nT = numTrials
    # nO = numArms
    m = np.zeros(shape=[numTrials_NS+1, numArms_NS]) + mu0
    v = np.zeros(shape=[numTrials_NS+1, numArms_NS]) + v0
    
    # Loop around trials
    for tCt in range(0, len(choices_NS)):
        
        # Find selection and reward
        selection = int(choices_NS[tCt])
        reward = reward_NS[tCt]*100
        
        if selection == -1:
            
            # Update Mean and Variances
            m[tCt+1,] = m[tCt,]
            v[tCt+1,] = v[tCt,]
            
        else:
        
            # Zero Kalman Gain Set up
            kt = np.zeros(shape=numArms_NS)
            
            # Calculate kalman Gain
            kt[selection] = (v[tCt, selection] + sigXi) / (v[tCt, selection] + sigXi + sigEps)
            
            # Update Mean and Variances
            m[tCt+1] = m[tCt] + kt*(reward - m[tCt])
            v[tCt+1] = (1 - kt) * (v[tCt] + sigXi)
        
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
    prob = np.zeros(shape =[numTrials_NS, numArms_NS])+1e-3

    for tCt in range(0, len(choices_NS)):
        
        # ALTERNATIVE APPROACH...
        #Find Prob for sampling          
        # Set up prior and sample from Normal distribution
        priorSamp = np.random.normal(loc=m[tCt, :], scale=np.sqrt(v[tCt, :]),
                                    size = ( numSamples, numArms_NS))
        # Determine Max Arms
        maxArm = np.argmax(priorSamp,axis=1)
        # Count Max 
        unique, counts = np.unique(maxArm, return_counts=True)  
        # Divide by number of samples
        prob[tCt, unique] = counts / numSamples
        
        # liklihoodArray[tCt] = prob[selection]
    
    
    # Convert choices in integer
    choices_int = choices_NS.astype(int)
    
    #Compute Liklihood
    liklihoodArray_NS = prob[np.indices(choices_int.shape)[0], choices_int]
    
    liklihoodArray_NS[choices_int == -1] = 1
    
    # Deal with Zeros
    liklihoodArray_NS[liklihoodArray_NS <= 0] = 1e+300
    
    # Deal with NaNs
    liklihoodArray_NS[np.isnan(liklihoodArray_NS)] = 1e+300
    
    #Update Liklihood Values
    liklihoodSum_NS = -np.sum(np.log(liklihoodArray_NS))


# %% Stationary

    #Create Liklihood Array
    liklihoodArray_S = np.zeros(shape=[numBlocks, numTrials_S])
    
    # # Isolate Parameters?
    mu0 = 0
    v0 = 100
    numSamples = 1000
    #sigXi = parameters
    sigEps2 = sigEps
    
        
    # Extract Posterior means and variances from KF calculation
    m = np.zeros(shape=[numBlocks, numTrials_S+1, numArms_S]) + mu0
    v = np.zeros(shape=[numBlocks, numTrials_S+1, numArms_S]) + v0
        
    for bCt in range(numBlocks):
    
        # Loop around trials
        for tCt in range(numTrials_S):
               
            # Find selection and reward
            selection = int(choices_S[bCt, tCt])
                
            if selection == -1:
                
                liklihoodArray_S[bCt, tCt] = 1
                
                # Calculate kalman Gain
                #kt[selection] = (v[bCt, tCt, selection] + sigXi) / (v[bCt, tCt, selection] + sigXi + sigEps)
                            
                # Update Mean and Variances
                m[bCt, tCt+1] = m[bCt, tCt]
                v[bCt, tCt+1] = v[bCt, tCt]
                
            else:
            
                prob = np.zeros(shape=numArms_S)
        
                reward = reward_S[bCt, tCt]
                
                if reward_S[bCt, tCt] == 1:
                    reward = 1
                else:
                    reward = 0
                    
                
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms_S)
                
                # Calculate kalman Gain
                kt[selection] = (v[bCt, tCt, selection] + sigXi) / (v[bCt, tCt, selection] + sigXi + sigEps2)
                            
                # Update Mean and Variances
                m[bCt, tCt+1] = m[bCt, tCt] + kt*(reward - m[bCt, tCt])
                v[bCt, tCt+1] = (1 - kt) * (v[bCt, tCt] + sigXi)
                            
                #Find Prob for sampling          
                # Set up prior and sample from Normal distribution
                priorSamp = np.random.normal(loc=m[bCt, tCt, :], scale=np.sqrt(v[bCt, tCt, :]),
                                            size = ( numSamples, numArms_S))
                # Determine Max Arms
                maxArm = np.argmax(priorSamp,axis=1)
                # Count Max 
                unique, counts = np.unique(maxArm, return_counts=True)  
                # Divide by number of samples
                prob[unique] = counts / numSamples
                
                liklihoodArray_S[bCt, tCt] = prob[selection]
        
    
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
