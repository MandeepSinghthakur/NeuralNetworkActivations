#########################################
#This File includes most of the mathematical operations related to activations.
#########################################


##TODO - Tensors
### Convert to methods to a class Activations

import numpy as np


#####
#Sigmoid Function Activations
#####


def sigmoid(x):
    """
    Computes as per formula y = 1/(1+exp(-x))
    
    """

    return 1 / (1 + np.exp(-x))

def logSigmoid(x):
   # return np.log(np.exp(x))
    return np.log(1 / (1 + np.exp(-x)))
    #TO-DO ----->

#####
#Relu Function
#####
def relu(x):
    return np.max(x,0,x)

####
#Softmax Function
#####
def softmax(x):
    e_p_max = np.exp(x-np.max(x))
    return e_p_max/ e_p_max.sum(axis = 0)
##TODO 