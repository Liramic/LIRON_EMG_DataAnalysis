from General.HelperFunctions import toFilePath, getPathsFromSessionFolder, getFirstCsvFile
import pandas as pd
import numpy as np

def GetPsychopyAnnotationsRelativeToPsychoPyStart(filename, freq=4000):
    dataColumns = []
    df = pd.read_csv(filename)
    for i in range(4):
        idx = df[f"Trigger{i}"].notnull()
        dataColumns.append((df[f"Trigger{i}"][idx].values, df[f"TriggerTime{i}"][idx].values))
    startExperimentTime = np.double(dataColumns[0][1][0])
    for i in range(4):
        for j in range(len(dataColumns[i][0])):
            temp = dataColumns[i][1]
            dataColumns[i][1][j] = np.float64(temp[j]) - np.float64(startExperimentTime)

    x = [[],[]]
    for i in range(4):
        x[0] = np.append(x[0],dataColumns[i][0])
        x[1] = np.append(x[1],dataColumns[i][1])
    
    x = zip(x[0],x[1])
    x = sorted(x, key=lambda y: y[1])
    titles, times = zip(*x)
    
    return titles, times


if (__name__ == "__main__"):
    folderSessionPath = r"C:\Liron\Code\LIRON_EMG_Data\06122022_1215"
    paths = getPathsFromSessionFolder(folderSessionPath, 0, 0)
    csvPath = getFirstCsvFile(folderSessionPath)

