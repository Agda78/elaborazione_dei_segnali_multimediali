# -*- coding: utf-8 -*-
"""
Functions to get and set bit-planes.
Created on Tue Mar 23 17:25:56 2021

@author: Davide
"""

def bitget(x, index):
    """
    Returns the values of the bits at a given position of a unsigned integer array.

    Parameters
    ----------
    x : unsigned integer array 
        Extract the bit-plane from this array.
    index : int
        Posizion of bit-plane to extract.
        It must be less than number of bits in the integer class of x.

    Returns
    -------
    boolean array of the same shape of x
        Extracted bit-plane.
    """
    
    import numpy
    if x.dtype not in [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
        raise ValueError("Only uint8, uint16, uint32, and uint64 are supported as dtype!")
    return (x & (1 << index)) != 0

def bitset(x, index, v):
    """
    Return an unsigned integer array modifying a bit-plane of a given array.

    Parameters
    ----------
    x : unsigned integer array 
        Modify the bit-plane from this array.
    index : int
        Posizion of bit-plane to modify.
        It must be less than number of bits in the integer class of x.
    v : 0 or 1 or boolean arry
        Value or values used to modify the bit-plane.
        The bits where v is False are set to 0 (off), and
        The bits where v is True are set to 1 (on).

    Returns
    -------
    unsigned integer array of the same shape of x
        Array with the modified bit-plane.

    Example
    -------
    a = bitset(x,0,1)
    
    """
    
    import numpy
    if x.dtype not in [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
        raise ValueError("Only uint8, uint16, uint32, and uint64 are supported as dtype!")
    if numpy.isscalar(v):
        v = numpy.asarray([v!=0,], dtype=x.dtype)
    else:
        v = numpy.asarray(v!=0, dtype=x.dtype)
    vp = v << index
    vn = ~((1-v) << index)
    
    return (x & vn) | vp 
    