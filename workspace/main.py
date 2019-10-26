#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:36:16 2019

@author: ranu
"""

# read a wavefile and compute spectrogram
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import supportingFunctions as SF

matplotlib.pyplot.jet()
sys.path.append('../sounds/')
fpath = '../sounds/slt/'
fname ='arctic_a0001'
fpathFull = fpath+fname+'.wav'
fs,wav = SF.wavread(fpathFull)

winD   = 0.02    # window duration
shiftD = 0.001   # window shift
winL   = np.floor(fs*winD).astype(int)
win    = np.hamming(winL)
Frames = SF.myframes(wav,fs,winD,shiftD,win)
nfft   = 512
X      = np.zeros((Frames.shape[0],nfft),dtype=complex)
for i in range(Frames.shape[0]):
    X[i,:] = np.fft.fft(Frames[i,:],n=nfft) 
    

mX = np.abs(X)**2
mX = mX[:,0:nfft/2+1]

mXdb   = 10.0*np.log10(mX+10**(-20))
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
#fig.colorbar(mesh1,ax=ax2)
plt.tight_layout()
ax2.set_xlabel('TIME (s)')
ax2.set_ylabel('FREQUENCY (Hz)')

#fig.colorbar(lplt1,axis=ax1)
#fig.colorbar(mesh1,axis=ax2)

ax1.set_ylabel('AMPLITUDE')
ax1.autoscale(enable=True, axis='x', tight=True)
ax2.autoscale(enable=True,axis='x',tight=True)
