import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import pearsonr


#mat = np.random.randn(10,1000)
#plt.imshow(mat, cmap='hot', aspect='auto', interpolation='gaussian')
#plt.show()
if False:
    y = np.random.randn(5,5) #np.array([[1,2],[3,4]])
    m = np.mean(y, axis=1)

    print(y)
    print(m)
    print( y - np.matrix(m).T)

# a = np.random.randn(1,4000*60).flatten()
# b = np.random.randn(1,4000*60).flatten()

# print ( xwt_coherence(a,b,4000) )

# from os import path
# import json
# p = path.join(r"C:\Liron\neuroscience\PHD\First year experiment - EMG\DataAnalysis\WSL", "20112022_1545", "correct.txt")
# with open(p, "r") as f:
#     corrections = json.load(f)
# print(type(corrections["A"]))
# print(corrections["B"])


# def GetTrapezoid(start, end, windowInMs = 50):
#     d = windowInMs*4
#     y = np.array([])
#     slope = np.arange(0,d)/d
#     y = np.append(y, slope)
#     y = np.append(y, np.ones((1, end-start-2*d)))
#     y = np.append(y, np.flip(slope))

#     return y


# start = 3000
# end = 3000 + 3*4000

# xrange = np.arange(start,end)
# yrange = GetTrapezoid(start,end,500)

# import matplotlib.pyplot as plt
# plt.ylim(top = 1.1)
# plt.plot(xrange, yrange)
# plt.show()

chunk_data = [[], []]
A=0
B=1
downsampleWindowInMs = 10
num_channels = 16

chunk_data[A] = np.random.randn(16, (60*1000//downsampleWindowInMs) + 3)
chunk_data[B] = np.random.randn(16, (60*1000//downsampleWindowInMs) +4 )

tempfilename = "1.png"

def SavePlotToTempFile():
    plt.savefig(tempfilename)
    return tempfilename


for lag_ind in np.arange(0,11): #00 - 1000 ms lag
    lag = lag_ind * 100//downsampleWindowInMs # 100 ms at a lag
    end = min(chunk_data[A].shape[1], chunk_data[B].shape[1])
    matA = chunk_data[A][:, :end-lag]
    matB = chunk_data[B][:, lag:end]

    windowForCorrInMs = 100
    numOfPixelsInCorrMat = windowForCorrInMs//downsampleWindowInMs
    corrMat = np.array([])
    shape1 = min(matA.shape[1], matB.shape[1])
    for s in range(0, shape1 - numOfPixelsInCorrMat , numOfPixelsInCorrMat):
        e = s + numOfPixelsInCorrMat
        #vec = np.sum(matA[:, s:e] * matB[:, s:e], axis=1)
        vec = np.array([pearsonr(matA[i, s:e], matB[i, s:e])[0] for i in range(matA.shape[0])])
        corrMat = np.append(corrMat, vec)
    
    corrMat = np.reshape(corrMat, (num_channels, len(corrMat)//num_channels))
    norm = Normalize(vmin=0, vmax=0.5)
    sns.heatmap(corrMat, norm=norm, cmap='viridis_r')
    plt.xlabel("Time")
    plt.ylabel("Components")
    #plt.show()
    SavePlotToTempFile()
    plt.cla()
    exit()