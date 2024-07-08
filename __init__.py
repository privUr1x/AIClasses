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

except Exception as e:
    print("Error", e)
