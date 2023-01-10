import os
from General.HelperFunctions import GetSessions, getPathsFromSessionFolder, cleanSpace, getICACsvFileNameWithPath, getWeightsFileNameWithPath 
import pyedflib
from EDF.EdfAnalyzer import EdfAnalyzer
import numpy as np


num_channels = 16
A = 0
B = 1

def SaveSessionIcaToDisk(current_session, data_path):
    print(current_session)

    signals = []
    chunks = []
    freq = 0
    independentComponents = [[], []]

    isICLoadedFromDisk = False

    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    print("Load Files")
    
    if (os.path.exists(getICACsvFileNameWithPath(data_path, current_session)) ):
        return
    
    for p in [A,B]:
        pathProperties = paths[p]
        f = pyedflib.EdfReader(pathProperties[0])
        chunks.append(EdfAnalyzer.getAnnotationChunks(f, pathProperties[1]))
        Y, freq = EdfAnalyzer.readEdf(f, doButter=True)
        Y = EdfAnalyzer.RemoveDataUntilStart(chunks, Y, freq)
        signals.append(Y)
        f.close()
        cleanSpace()

    appendedSignal = np.append(signals[A], signals[B], axis=1)
    print("ICA")
    # ICA 
    w, x = EdfAnalyzer.ICA(appendedSignal, num_channels) # W*Y = X

    # seperate components :
    splitValue = len(signals[A][0])
    independentComponents[A] = x[:, :splitValue]
    independentComponents[B] = x[:, splitValue:]

    np.savetxt(getWeightsFileNameWithPath(data_path, current_session), w, delimiter=",")
    
    del w
    del x
    del Y
    del signals

    cleanSpace()
        
    np.savez(getICACsvFileNameWithPath(data_path, current_session), independentComponents[A], independentComponents[B])
        
    del independentComponents
    cleanSpace()


def LoadComponents(data_path, current_session, downsampleWindowInMs):
    chunks = []
    independentComponents = [[], []]

    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    print("Load Files")
    
    if (not os.path.exists(getICACsvFileNameWithPath(data_path, current_session)) ):
        SaveSessionIcaToDisk(current_session, data_path)
    
    loaded = np.load(getICACsvFileNameWithPath(data_path, current_session))
    independentComponents[A] = loaded["arr_0"]
    independentComponents[B] = loaded["arr_1"]
    del loaded
    cleanSpace()
    
    for p in [A,B]:
        pathProperties = paths[p]
        f = pyedflib.EdfReader(pathProperties[0])
        chunks.append(EdfAnalyzer.getAnnotationChunks(f, pathProperties[1], downsampleWindowInMs))
        f.close()
        cleanSpace()

    print("RMS")
    for p in [A, B]:
       independentComponents[p] = EdfAnalyzer.window_rms_downsample_no_cut_faster(independentComponents[p], downsampleWindowInMs)
    cleanSpace()

    return independentComponents, chunks


if __name__ == "__main__":
    data_path = r"C:\Liron\DataEmg\Done"
    sessions = GetSessions(data_path)
    for current_session in sessions:
        SaveSessionIcaToDisk(current_session, data_path)

    print("Finished")
    exit()