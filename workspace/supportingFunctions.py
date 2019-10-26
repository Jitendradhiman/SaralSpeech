#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:55:52 2019

@author: ranu
"""

from scipy.io.wavfile import read
import os
import numpy as np

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def wavread(filename):
	"""
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: sampling rate of file, x: floating point array
	"""

	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		raise ValueError("Input file is wrong")

	fs, x = read(filename)

	if (len(x.shape) !=1):                                   # raise error if more than one channel
		raise ValueError("Audio file should be mono")

	#if (fs !=44100):                                         # raise error if more than one channel
		#raise ValueError("Sampling rate of input sound should be 44100")

	#scale down and convert audio into floating point number in range of -1 to 1
	x = np.float32(x)/norm_fact[x.dtype.name]
	return fs, x
def myframes(wav,fs,winD,shiftD,win):
    winL = np.floor(winD*fs).astype(int)
    shiftL = np.floor(shiftD*fs).astype(int)
    nFr = (wav.size - winL)/shiftL +1
    Frames = np.zeros((nFr,winL))
    FE = 0
    c  = 0
    while FE < len(wav)-winL:
        FB = c*shiftL+1
        FE = FB+winL
        Frames[c,:] = wav[FB:FE]*win
        c = c+1
    return Frames        
     
