import numpy as np

def split_signal(x):
    unsigned = x + 2**15

    coarse = unsigned // 256    
    fine = unsigned % 256

    return coarse, fine

def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15