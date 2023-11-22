#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:55:52 2019

@author: Jitendra
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
	returns fs: sampling rate of file (int64), x: floating point array
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
def myframes(wav,**config):    
    """
    # returns frames of the signal
    # wav: waveform
    # fs: sampling rate
    # winD: window duration (s)
    # shiftD: hop duration (s)
    """
    fs, winD, shiftD = config['fs'], config['windowDuration'], config['hopDuration']
    winType = config['windowType']
    # returns frames
    winL = np.floor(winD*fs).astype(int)
    shiftL = np.floor(shiftD*fs).astype(int)
    if winType == 'hamming':
        win = np.hamming(winL)
    else:
        print('Error: Window type is not supported')
    nFr = (wav.size - winL)//shiftL +1
    Frames = np.zeros((nFr,winL))
    FE = 0
    c  = 0
    while FE < len(wav)-winL:
        FB = c*shiftL
        FE = FB+winL
        Frames[c,:] = wav[FB:FE]*win # each row is a frame
        c = c+1
    return Frames

def getSTFT(Frames, nfft=1024):
    # returns STFT
    X      = np.zeros((Frames.shape[0],nfft),dtype=complex)
    for i in range(Frames.shape[0]):
        X[i,:] = np.fft.fft(Frames[i,:],n=nfft)
    return X

def plotWavAndSpec(wav, X, fs, shiftD):
    # plot wav and spectrogram
    # wav: waveform
    # X: STFT
    # fs: sampling rate
    nfft = X.shape[1]
    mX = np.abs(X)**2
    mX = mX[:,0:nfft//2+1]
    mXdb   = 10.0*np.log10(mX + 10**(-20))
    dbdown = 60
    vmin   = mXdb.max() - dbdown
    vmax   = mXdb.max()

    #%% PLOTS
    taxis = np.arange(wav.size)*(1.0/fs)
    naxis = shiftD  * np.arange(mX.shape[0])
    faxis = fs/nfft * np.arange(nfft/2+1)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    lplt1 = ax1.plot(taxis,wav)
    mesh1 = ax2.pcolormesh(naxis,faxis,mXdb.T)
    mesh1.set_clim(vmin,vmax)
#     fig.colorbar(mesh1,ax=ax2)
    plt.tight_layout()
    ax2.set_xlabel('TIME (s)')
    ax2.set_ylabel('FREQUENCY (Hz)')

#     fig.colorbar(lplt1,axis=ax1)
#     fig.colorbar(mesh1,axis=ax2)

    ax1.set_ylabel('AMPLITUDE')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax2.autoscale(enable=True,axis='x',tight=True)
    
def makeLengthsEqual(s1, L2):
    # makes length of signal s2 equal to length L2
    if len(s1) > L2:
        s1 = s1[0:L2]
    else:
        ldiff = L2 - len(s1)
        s1 = np.concatenate((s1, np.zeros(ldiff, )))
    return s1

def OLA1D(Frames, **config):
    # OLA using LSE
    fs, winType, winD, shiftD = config['fs'], config['windowType'], config['windowDuration'], config['hopDuration']
    winL = np.floor(winD * fs).astype(int)
    if winType == 'hamming':
        win = np.hamming(winL)
    else:
        print('Window type not supported')
    shiftlocs = Frames.shape[0]
    shiftL = np.floor(shiftD * fs).astype(int)
    Len  = winL + (shiftlocs - 1) * shiftL
    fn = np.zeros((Len, ))
    fd = np.zeros((Len, ))
    for i in range(shiftlocs):
        idx = np.arange(i * shiftL, i * shiftL + winL)
        sig = Frames[i,:]
        fn[idx] = fn[idx] + sig[0:winL] * win
        fd[idx] = fd[idx] + win*win
    rs = fn / fd
    rs = np.real(rs)
    rs[np.isnan(rs)] = 0
    rs[np.isinf(rs)] = 0
    return rs

