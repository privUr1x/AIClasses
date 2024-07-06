#!/usr/bin/python3 
"""Initializer for the module"""

try: 
    from numpy import (
        float16,
        float32,
        float64,
        float_,
        int16,
        int32,
        int64,
        int8,
        ndarray,
        uint16,
        uint32,
        uint64,
        uint8,
    )

global nptypes

nptypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    float_,
)
