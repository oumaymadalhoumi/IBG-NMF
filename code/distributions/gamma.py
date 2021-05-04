"""
Class representing a gamma distribution, allowing us to sample from it, 
and compute the expectation and the expectation of the log.
"""
import math
from scipy.special import psi as digamma
from numpy.random import gamma


# Gamma draws
def gamma_draw(alpha,beta):       
    shape = float(alpha)
    scale = 1.0 / float(beta)
    return gamma(shape=shape,scale=scale,size=None)
        
# Gamma expectation
def gamma_expectation(alpha,beta): 
    alpha, beta = float(alpha), float(beta)      
    return alpha / beta
        
# Gamma variance
def gamma_expectation_log(alpha,beta):   
    alpha, beta = float(alpha), float(beta)      
    return digamma(alpha) - math.log(beta)
   
# Gamma mode
def gamma_mode(alpha,beta):
    alpha, beta = float(alpha), float(beta)
    return (alpha-1) / beta

