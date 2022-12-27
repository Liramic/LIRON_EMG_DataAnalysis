import os
from datetime import datetime
import gc
import cupy as xp
import numpy as np
import pywt
from scipy import ndimage, signal

def log(string):
    print(f"{datetime.now().strftime('%H:%M:%S')} : {string}")

def toFilePath(sessionFolder, fname):
    return os.path.join(sessionFolder, fname)

def getPathsFromSessionFolder(sessionFolder, correctionA = 0.4, correctionB = 0.4):
    files = [x for x in os.listdir(sessionFolder) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    return [(toFilePath(sessionFolder, fileA), correctionA),(toFilePath(sessionFolder, fileB), correctionB)]

def getFirstCsvFile(path):
    for x in os.listdir(path):
        if x.endswith(".csv"):
            return toFilePath(path, x)

def isMyAnnotation(annotation):
    parts = [".ogg", ".png", "smile_", "angry_", "blink_"]
    s = str(annotation).lower()
    for part in parts:
        if ( part in s):
            return True
    return False

def cleanSpace():
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()

def matrixCorrelation(mat1, mat2):
    cols1 = mat1.shape[1]
    cols2 = mat2.shape[1]
    cols = min(cols1, cols2)
    return np.mean(np.corrcoef(mat1[:, 0:cols], mat2[:, 0:cols]))


def xwt_coherence(x1, x2, fs, nNotes=12, detrend=True, normalize=True):
    
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2),   "error: arrays not same size"
    
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x1 = signal.detrend(x1,type='linear')
        x2 = signal.detrend(x2,type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2

    ###########################################################################
    # Define some parameters of our wavelet analysis. 

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef1, freqs1=pywt.cwt(x1,scales,'cmor1.5-1.0')
    coef2, freqs2=pywt.cwt(x2,scales,'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # Calculates the cross transform of xs1 and xs2.
    coef12 = coef1 * np.conj(coef2)

    ###########################################################################
    # coherence
    scaleMatrix = np.ones([1, N]) * scales[:, None]
    S1 = ndimage.gaussian_filter((np.abs(coef1)**2 / scaleMatrix), sigma=2)
    S2 = ndimage.gaussian_filter((np.abs(coef2)**2 / scaleMatrix), sigma=2)
    S12 = ndimage.gaussian_filter((np.abs(coef12 / scaleMatrix)), sigma=2)
    WCT = S12**2 / (S1 * S2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return WCT, times, frequencies, coif