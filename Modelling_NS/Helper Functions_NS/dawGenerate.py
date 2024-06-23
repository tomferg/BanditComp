# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:55:48 2022

@author: Tom
"""
import numpy as np

def dawGenerate(arms,Trials):
    
    #Parameters Taken from Daw Paper
    decayParam = 0.9836;
    decayCen = 50;
    diffNoiseSD = 2.8;
    payoutSD = 4;
    
    #Createe Blank Arrays
    payouts = np.zeros(shape=[arms,Trials])
    actualPayouts = np.zeros(shape=[arms,Trials])
    
    #Initialize Reward Value
    payouts[:,0] = 100*np.random.rand(4)
    
    #Compute Mean Payouts
        
    #Start at trial 2
    for trialCount in range(1,Trials):
    
       payouts[:,trialCount] = decayParam*payouts[:,trialCount-1] + (1-decayParam)*decayCen + diffNoiseSD * np.random.normal(size=arms);
        
    #Constrain Mean Payouts
    payouts[payouts < 1] = 1
    payouts[payouts > 100] = 100
    
    #Compute Actual Payouts
    noiseValue = payoutSD * np.random.normal(size=[arms,Trials])
    
    #Round Values
    actualPayouts = np.round(payouts+noiseValue)
    
    #Constain actual Payouts
    actualPayouts[actualPayouts < 1] = 1
    actualPayouts[actualPayouts > 100] = 100
        
    return(actualPayouts)
