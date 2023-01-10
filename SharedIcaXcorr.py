import os
from General.HelperFunctions import GetSessions, cleanSpace
import numpy as np
from EDF.SharedIcaRunner import LoadComponents
from scipy.stats import zscore
from skimage.metrics import structural_similarity
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from General import HtmlTablePlot

#data_path = r'/mnt/c/Liron/DataEmg/Done'
data_path = r"C:\Liron\DataEmg\Done"
tempfilename = "lastplot.png"

def SavePlotToTempFile():
    plt.savefig(tempfilename)
    return tempfilename


num_channels = 16
A = 0
B = 1
downsampleWindowInMs = 10 # downsamples to 100HZ
orig_freq = 4000

sessions = GetSessions(data_path)

for current_session in [sessions[14]]:
    print(current_session)
    independentComponents, chunks = LoadComponents(data_path, current_session, downsampleWindowInMs)
    
    output_dic = HtmlTablePlot.initHtmlDictionary(["session", "key", "lag", "meanCorr", "maxMeanCompCor","plot"])
    print("Zscore")
    for p in [A, B]:
       independentComponents[p] = zscore(independentComponents[p], axis=1)
    cleanSpace()

    freq = orig_freq // (4*downsampleWindowInMs)

    print("Go over chunks")
    for key in chunks[0]:
        chunk_data = [[],[]]
        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        windowForCorrInMs = 100
        numOfPixelsInCorrMat = windowForCorrInMs//downsampleWindowInMs
        for lag_ind in np.arange(0,11): #00 - 1000 ms lag
            lag = lag_ind * numOfPixelsInCorrMat # 100 ms at a lag
            end = min(chunk_data[A].shape[1], chunk_data[B].shape[1])
            mat1 = chunk_data[B][:, :end-lag]
            mat2 = chunk_data[A][:, lag:end]

            corrMat = np.array([])
            shape1 = min(mat1.shape[1], mat2.shape[1])
            for s in range(0, shape1 - numOfPixelsInCorrMat , numOfPixelsInCorrMat):
                e = s + numOfPixelsInCorrMat
                #vec = np.sum((mat1[:, s:e] * mat2[:, s:e])/(e-s), axis=1)
                vec = np.array([pearsonr(mat1[i, s:e], mat2[i, s:e])[0] for i in range(mat1.shape[0])])
                corrMat = np.append(corrMat, vec)
            
            corrMat = np.reshape(corrMat, (num_channels, len(corrMat)//num_channels))
            matMeanCorr = np.mean(corrMat)
            maxCompMean = max(np.mean(corrMat, axis=1))

            norm = Normalize(vmin=0, vmax=0.5)
            sns.heatmap(corrMat, norm=norm, cmap='viridis_r')
            plt.xlabel("Time")
            plt.ylabel("Components")
            plt.title(f"{key} A synchronize B, lag: {lag}")
            HtmlTablePlot.addRow(output_dic, [HtmlTablePlot.toTextHtmlCol(current_session), 
                HtmlTablePlot.toTextHtmlCol(key), HtmlTablePlot.toTextHtmlCol(lag),
                HtmlTablePlot.toTextHtmlCol(matMeanCorr), HtmlTablePlot.toTextHtmlCol(maxCompMean),
                HtmlTablePlot.toImageHtmlColFromPath(SavePlotToTempFile())])
            
            plt.close()
            plt.cla()
    HtmlTablePlot.SaveHtmlFile(output_dic, os.path.join(data_path, current_session, "B_Corr_A_pearson.html"))
    print("finished session " + current_session)
