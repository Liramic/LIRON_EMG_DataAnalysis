# the idea is stringing vectors from eachother's electrode
# then perform PCA
# then split again to the different componenets.

# After that, we can either corrolate or visuallize the result.

import pyedflib
from matplotlib.widgets import Slider #for horizontal scrolling in time plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
#from torch import maximum
from EdfAnalyzer import EdfAnalyzer

num_components = 16

############## functions #####################

def plot(y_val, X, totalSize, index, title, color, ticks): #, maximun_val):
    plot = plt.subplot2grid(totalSize, index, rowspan=2, colspan=2)
    plot.plot(X,y_val, color = color)
    plot.set_title(title)
    for tick in ticks:
        plot.axvline(tick, linestyle='--')
    plt.gca().axes.get_xaxis().set_visible(False)


def plotChunkData(X, title, ticks, subtitles, maximum_values):
    y_axis = X
    x_axis = range(0, y_axis.shape[1])
    total_size = (num_components*2,2)
    #maximun_val = np.max(y_axis) # much better
    for i in range(0,num_components):
        plot(y_axis[i], x_axis, total_size, (2*i, 0), subtitles[i], 'r', ticks) #, np.float(maximum_values[i]))
    plt.suptitle(title)
    plt.show()

def fixTicks(ticks):
    s = ticks[0]
    return [ x - s for x in ticks]

def plotDataFromBothChunks(X1, X2, ticks1, ticks2, title="smiles"):
    total_graph_size = (num_components*2,4)
    ticks1 = fixTicks(ticks1)
    ticks2 = fixTicks(ticks2)

    for index in range(16):
        y1 = X1[index]
        y2 = X2[index]
        plot(y1, range(0,y1.shape[0]), total_graph_size,(index*2, 0), "component %d" % index, 'r', ticks1 )
        plot(y2, range(0,y2.shape[0]), total_graph_size,(index*2, 2), "component %d" % index, 'b', ticks2 )
        index = index+1
        
    plt.suptitle(title)
    plt.show()


#################### main #########################

num_channels = 16
A = 0
B = 1
signals = []
chunks = []
freq = 0

# read both EDF files
#paths = ["29052022_1230_A_Recording_00_SD.edf", "29052022_1230_b_Recording_00_SD.edf"]
paths = [r"C:\Liron\neuroscience\PHD\First year experiment - EMG\DataAnalysis\EMG\06062022_1100\06062022_1100_A_Recording_00_SD.edf", r"C:\Liron\neuroscience\PHD\First year experiment - EMG\DataAnalysis\EMG\06062022_1100\06062022_1100_B_Recording_00_SD.edf"]
for path in paths:
    f = pyedflib.EdfReader(path)
    Y, freq = EdfAnalyzer.readEdf(f, doButter=False)
    signals.append(Y)
    #EdfAnalyzer.reduceMeanFromEachCol(Y)
    chunks.append(EdfAnalyzer.getAnnotationChunks(f))
    f.close()

# Append Signals
appendedSignal = []
for i in range(num_channels):
    appendedSignal.append(np.append(signals[A][i], signals[B][i]))

# ICA 
w, x = EdfAnalyzer.ICA(np.matrix(appendedSignal), num_channels) # W*Y = X
# x = EdfAnalyzer.window_rms(x) - perform the RMS after the split

# seperate components :
independentComponents = [[], []]
for i in range(num_channels):
    splitValue = len(signals[A][i])
    independentComponents[A].append(x[i][:splitValue])
    independentComponents[B].append(x[i][splitValue:])

# RMS
#independentComponents = [ EdfAnalyzer.window_rms(comps) for comps in independentComponents]

# take only smiles chunk.
# smile_chunks = [[],[]]
# for i in range(3):
#     for participant in [A,B]:
#         chunk = chunks[participant]["smile__%d" % i]
#         chunk_start = int(chunk.Start.Time*freq)
#         chunk_end = int(chunk.End.Time*freq)
#         chunk.data = [independentComponents[participant][c][chunk_start:chunk_end] for c in range(num_channels)]
#         smile_chunks[participant].append(chunk)

states = ["smile", "angry", "blink"]
for state in states :
    ticks = []
    smile_whole_chunks = []
    for particiapnt in [A,B]:
        ticks.append(EdfAnalyzer.getCallibrationTicks(chunks[particiapnt], freq, state))
        start = ticks[particiapnt][0]
        end = ticks[particiapnt][-1]
        smile_whole_chunks.append([x[i][start:end] for i in range(num_channels)])
    # print both smile chunks ICA marked (2*16 graphs):
    plotDataFromBothChunks(smile_whole_chunks[A], smile_whole_chunks[B], ticks[A], ticks[B])


