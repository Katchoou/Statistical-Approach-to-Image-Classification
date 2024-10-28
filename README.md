# Image-classification

import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
import pathlib
from glob import glob
from scipy import stats
import math
from skimage.feature import graycomatrix, graycoprops
import cv2
import seaborn as sns


import numpy as np
#1
nbit=8
L=2**nbit
def probabilities(hist, nbre_pixel_image):
    prob=[card_B/nbre_pixel_image for card_B in hist]
    return prob
    
#2
def expectation(prob, levels):
    e=0
    for i in range(L-1):
        e+=prob[i]*levels[i]
    return e
    
#3
def variance(e, prob):
    var=0
    for i in range(L-1):
        var+=prob[i]*(i-e)**2
    return var

#4
def asymetry(e, prob):
    asy=0
    for i in range(L-1):
        asy+=prob[i]*(i-e)**3
    return asy

#5
def entropy(prob):
    entropy=0
    for i in range(L-1):
        if prob[i]>0:
            entropy+=prob[i]*math.log2(prob[i])
    return -1*entropy

#6
def excess(e, prob):
    excess=0
    for i in range(L-1):
        excess+=prob[i]*(i-e)**4
    return excess

#7
def homogeneity(prob):
    homo=0
    for i in range(L-1):
        homo+=prob[i]**2
    return homo

#8
def skewness(variance, asymetry):
    sigma=np.sqrt(variance)
    skew=asymetry/sigma**3
    return skew

#9
def kurtosis(var, excess):
    return excess/(var**2)  
