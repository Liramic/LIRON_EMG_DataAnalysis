import os
from General.HelperFunctions import GetSessions, getPathsFromSessionFolder, cleanSpace, getICACsvFileNameWithPath, getWeightsFileNameWithPath
import pyedflib
from EDF.EdfAnalyzer import EdfAnalyzer
import numpy as np
from General.StatisticsFunctions import movingSsim, movingMatrixCorr, movingSsimSpecificComponents, specificSsim, matrixCorrelation, orb_sim, ssim
from EDF.SharedIcaRunner import LoadComponents
from scipy.stats import zscore
from General.PlotFunctions import plotDataFromBothChunks
from skimage.metrics import structural_similarity
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.pyplot as plt


#data_path = r'/mnt/c/Liron/DataEmg/Done'
data_path = r"C:\Liron\DataEmg\Done"

num_channels = 16
A = 0
B = 1
downsampleWindowInMs = 100 # downsamples to 10HZ
orig_freq = 4000

sessions = GetSessions(data_path)

for current_session in sessions:
    print(current_session)
    independentComponents, chunks = LoadComponents(data_path, current_session, downsampleWindowInMs)

    print("Zscore")
    for p in [A, B]:
       independentComponents[p] = zscore(independentComponents[p], axis=1)
    cleanSpace()

    freq = orig_freq // (4*downsampleWindowInMs)

    print("Go over chunks")
    for key in [list(chunks[0].keys())[10], list(chunks[0].keys())[11]]: #chunks[0]:
        chunk_data = [[],[]]
        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        #smile_comps = np.array([6,7,9,14]) - 1
        #print(f"key : {key} value: {movingSsim(chunk_data[A], chunk_data[B], lagMs=500, freq=10)}")
        #plotDataFromBothChunks(np.diff(chunk_data[A]), np.diff(chunk_data[B]), title = str(key))
        #movingSsimValue = movingSsimSpecificComponents(chunk_data[A], chunk_data[B], smile_comps, lagMs=500,  freq=freq)
        lag = int(0.5*10)
        #value = matrixCorrelation(np.diff(chunk_data[A][:, lag:]), np.diff(chunk_data[B]), smile_comps)
        #print(f"key : {key}, moving SSIM Value: {value}")
        for lag in np.arange(0,13): #00 - 1200 ms lag
            end = min(chunk_data[A].shape[1], chunk_data[B].shape[1])
            ssim_full_res = structural_similarity(chunk_data[A][:, :end-lag], chunk_data[B][:, lag:end], full=True, win_size=3)
            ssim_full = ssim_full_res[1]
            print(f"{key} A synchronize B, lag: {lag} score: {ssim_full_res[0]}")
            norm = Normalize(vmin=0, vmax=0.5)
            sns.heatmap(ssim_full, norm=norm, cmap='viridis_r')
            plt.xlabel("Time")
            plt.ylabel("Components")
            plt.title(f"{key} A synchronize B, lag: {lag}")
            plt.show()
            plt.cla()
    
    print("finished session " + current_session)
