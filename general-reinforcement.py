import numpy as np
import gym

def DataShape(raw_input):
    """ DataShape takes in the raw input and returns an array of integers where each value is the length of each dimension."""
    return raw_input.shape

def DataSize(raw_input):
    """ DataSize takes in the raw input and returns the product of all dimensions"""
    return raw_input.size

def Downsample(raw_input, resolution_threshold):
    """ Downsample reduces the resolution of a given environment so that it is below the given threshold."""

    environmentResolution = DataSize(raw_input)
    downsampledInput = raw_input.copy()

    # Divide the difference bewteen the environmentResolution by the
    # diff to know how much resolution needs to be downsampled.
    diff = environmentResolution - resolution_threshold
    differenceMultiple = np.ceil(environmentResolution / diff)

    return downsampledInput[::,::differenceMultiple]

def Echo(raw_input, resolution_threshold):
    """ Echo returns the raw input unchanged"""
    return raw_input

def PreprocessFunction(raw_input, resolution_threshold):
    """ PreprocessFunction selects the appropriate function to pre-process the raw input to achieve a reasonable amount of input resolution."""
    environmentResolution = DataSize(raw_input)
    
    if environmentResolution > resolution_threshold:
        return Downsample
    
    return Echo
