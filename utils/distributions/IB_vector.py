"""
Class representing a Truncated Normal distribution, with a=0 and b-> inf, 
allowing us to sample from it, and compute the expectation and the variance.
This is the special case that we want to compute the expectation and variance
for multiple independent variables - i.e. mu and tau are vectors.
truncnorm: a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
           loc, scale = mu, sigma
           
We get efficient draws using the library rtnorm by C. Lassner, from:
    http://miv.u-strasbg.fr/mazet/rtnorm/
We compute the expectation and variance ourselves - note that we use the
complementary error function for 1-cdf(x) = 0.5*erfc(x/sqrt(2)), as for large
x (>8), cdf(x)=1., so we get 0. instead of something like n*e^-n.
As mu gets lower (negative), and tau higher, we get draws and expectations that
are closer to an exponential distribution with scale parameter | mu * tau |.
The draws in this case work effectively, but computing the mean and variance
fails due to numerical errors. As a result, the mean and variance go to 0 after
a certain point.
This point is: -38 * std.
This means that we need to use the mean and variance of an exponential when
|mu| gets close to 38*std.
Therefore we use it when |mu| < 30*std.
"""
import math, numpy, time
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
from scipy.special import erfc
import inverted_beta


# TN draws
def IB_vector_draw(alphas,betas):
    draws = []
    for (alpha,beta) in zip(alphas,betas):
        d = inverted_beta.ib_draw(alpha,beta)
        d = d if (d >= 0. and d != numpy.inf and d != -numpy.inf and not numpy.isnan(d)) else 0.
        draws.append(d)   
    return draws           
       
# IB expectation    
def IB_vector_expectation(alphas,betas):
    exp = numpy.divide(alphas, betas - numpy.float64(1.0) )
    return [v if (v >= 0.0 and v != numpy.inf and v != -numpy.inf and not numpy.isnan(v)) else 0. for v in exp]
    

# TN variance
def TN_vector_variance(alphas,betas):
    numerator = numpy.multiply(alphas, alphas+betas-numpy.float64(1.0))
    denomenator = numpy.multiply(betas - numpy.float64(2.0),betas - numpy.float64(1.0))
    var = numpy.divide ( numerator, denomenator)
    return [v if (v >= 0.0 and v != numpy.inf and v != -numpy.inf and not numpy.isnan(v)) else 0. for v in var]      
       
# TN mode
def TN_vector_mode(mus):
    numerator = alphas - numpy.float64(1.0)
    denomenator = betas + numpy.float64(1.0)
    mode = numpy.divide ( numerator, denomenator)
    return [v if (v >= 0.0 and v != numpy.inf and v != -numpy.inf and not numpy.isnan(v)) else 0. for v in mode]      