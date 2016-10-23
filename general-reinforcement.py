import numpy as np
import gym

def DataShape(raw_input):
    """ DataShape takes in the raw input and returns an array of integers where each value is the length of each dimension."""
    return raw_input.shape

def DataSize(raw_input):
    """ DataSize takes in the raw input and returns the product of all dimensions"""
    return raw_input.size

def Downsampler(number_of_dimensions):
    """ Downsample reduces the resolution of a given environment so that it is below the given threshold."""

    def handler(raw_input, resolution_threshold):
        environmentResolution = DataSize(raw_input)
        downsampledInput = raw_input.copy()

        # Divide the difference bewteen the environmentResolution by the
        # diff to know how much resolution needs to be downsampled.
        diff = environmentResolution - resolution_threshold
        differenceMultiple = np.ceil(environmentResolution / diff)

        if number_of_dimensions == 1:
            return downsampledInput[::differenceMultiple]
        if number_of_dimensions == 2:
            return downsampledInput[::,::differenceMultiple]
        if number_of_dimensions == 3:
            return downsampledInput[::,::,::differenceMultiple]
        if number_of_dimensions == 4:
            return downsampledInput[::,::,::,::differenceMultiple]
        return raw_input

    return handler

def Echo(raw_input, resolution_threshold):
    """ Echo returns the raw input unchanged"""
    return raw_input

def PreprocessFunction(raw_input, resolution_threshold):
    """ PreprocessFunction selects the appropriate function to pre-process the raw input to achieve a reasonable amount of input resolution."""
    environmentResolution = DataSize(raw_input)
    
    if environmentResolution > resolution_threshold:

        numDimensions = len(DataShape(raw_input))
        if numDimensions == 1:
            return Downsampler(1)
        if numDimensions == 2:
            return Downsampler(2)
        if numDimensions == 3:
            return Downsampler(3)
        if numDimensions == 4:
            return Downsampler(4)

    return Echo


def Reshape(input_data, desired_shape):
    '''Reshape n-dimensional input_data into desired shape given as tuple'''
    return numpy.reshape(input_data, desired_shape)

def FeedForward(neural_net, shaped_input):
    '''Feed properly shaped input through neural net returning scalar output and hidden state'''
    hidden_state = shaped_input.copy()

    # fully connected neural net
    for i in xrange(0, neural_net.num_layers - 2):
        hidden_state = numpy.multiply(hidden_state, neural_net[i])
        hidden_state[hidden_state < 0] = 0 # ReLU nonlinearity

    # dot product returns scalar from two arrays of same dimensions
    y = numpy.dot(hidden_state, neural_net[-1])
    y = Sigmoid(y)
    
    return y, hidden_state

def CalculateAction(y):
    '''Weighted toss of coin of weight y'''
    return random.uniform(0, 1) >= y

def Sigmoid(y): 
    '''Sigmoid "squashing" function to interval [0,1]'''
    return 1.0 / (1.0 + np.exp(y))
