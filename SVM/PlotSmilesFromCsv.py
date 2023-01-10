import matplotlib.pyplot as plt
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.stats import zscore
from skimage.metrics import structural_similarity

def ArrayReverse(arr):
    return arr[::-1]

def TakeOneChannel(sig, channels):
    if channels == 1:
        return sig.reshape(-1,1)
    return sig[0][:,0].reshape(-1,1)

def envelope(sig,a=0,b=0):
    return np.abs(signal.hilbert(sig))

def moving_average(values, n) :
    return np.average(sliding_window_view(values, window_shape = n), axis=(1,1))


def ComputeCrossCorrToEnvelope(sig1, sig2):
    sig1 = zscore(sig1)
    sig2 = zscore(sig2)
    
    corr = signal.correlate(sig1,sig2,mode ="full", method="fft");
    return np.max(corr)/len(sig1)

def plot(y_val, X, totalSize, index, title, color, ticks): #, maximun_val):
    plot = plt.subplot2grid(totalSize, index, rowspan=1)
    plot.plot(X,y_val, color = color)
    plot.set_title(title)
    for tick in ticks:
        plot.axvline(tick, linestyle='--')
    plt.gca().axes.get_xaxis().set_visible(False)


def plotChunkData(X, ticks):
    y_axis = X
    total_size = (2,1)
    #maximun_val = np.max(y_axis) # much better
    for i in range(0,2):
        plot(y_axis[i], range(0,len(y_axis[i])) , total_size, (i, 0), f"participant {i}", 'r', ticks[i])
    plt.show()

def window_rms_single(a, window_size = int(0.25*4000)):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))

def getPairwiseZip(arr):
    return list(zip(arr[0::2], arr[1::2]))

def loadJson(filename):
    data = None
    ticks = []
    titles = []
    with open(filename, "r") as f: 
        data = json.load(f)
    for item in data:
        ticks.append(item["start"])
        ticks.append(item["end"])
        titles.append(item["key"])
    return ticks, titles

A=0
B=1

titles = []
data = []
ticks = []
folder = r"C:\Liron\DataEmg\Done\20122022_0830"
stateToAnalyze = "smile"
svmType = "svr" # "svc"

path = fr"{folder}\{stateToAnalyze}\{svmType}"

ticksA, titles = loadJson(fr"{path}\A.list")
ticks.append(ticksA)
ticksB, titles = loadJson(f"{path}\B.list") 
ticks.append(ticksB)

data.append(np.loadtxt(f"{path}\A.csv", delimiter=","))
data.append(np.loadtxt(f"{path}\B.csv", delimiter=","))
print(f"tick len: {[len(x) for x in ticks]}")
data = [ window_rms_single(x) for x in data]

ticksPairWise = [getPairwiseZip(x) for x in ticks]
for i in range(0,len(ticksPairWise[0])):
    A_start = ticksPairWise[A][i][0]
    A_end = ticksPairWise[A][i][1]
    B_start = ticksPairWise[B][i][0]
    B_end = ticksPairWise[B][i][1]
    sizeOfPart = min(A_end-A_start, B_end - B_start)
    corr = structural_similarity(data[A][A_start:A_start+sizeOfPart], data[B][B_start:B_start+sizeOfPart], gaussian_weights = False,
                        sigma = 1.5, use_sample_covariance = False, multichannel=False)
    print(f"{titles[i]} : {corr}")


plotChunkData(data, ticks)
