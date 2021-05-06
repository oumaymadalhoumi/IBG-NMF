"""
Class representing a gamma distribution, allowing us to sample from it, 
and compute the expectation and the expectation of the log.
"""
import math
from scipy.special import psi as digamma
from numpy.random import beta


# Gamma draws
def ib_draw(alpha,beta):       
    shape = float(alpha)
    scale = 1.0 / float(beta)
    X = beta(shape=shape , scale = scale, size=None)
    return  X /(1-X)
        
# Gamma expectation
def inverse_beta_expectation(alpha,beta): 
    alpha, beta = float(alpha), float(beta)      
    return alpha / (beta -1)