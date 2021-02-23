# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:05:38 2021

@author: Adin
"""
import random
class NoiseCreator:
    def createNoise(array, noise):
        noisyarray=[]
        for index in array:
            noisyarray.append(index+random.uniform(-noise,noise))
        
        return noisyarray
