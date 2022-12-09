import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
#from cuml.svm import LinearSVC
from cuml.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix
import cupy as xp
import os
from cupy import savetxt
from EdfAnalyzer import EdfAnalyzer
import gc
from datetime import datetime
import pandas as pd
import json


class UserChoice:
    def __init__(self, story1, story2, choiceA, choiceB, rtA, rtB, isOther) -> None:
        self.story1 = story1
        self.story2 = story2
        self.choiceA = choiceA
        self.choiceB = choiceB
        self.rtA = rtA
        self.rtB = rtB
        self.isOther = isOther
    
    def __str__(self):
        return f"story1 : {self.story1}, story2:{self.story2}, choiceA:{self.choiceA}, choiceB:{self.choiceB}, rtA:{self.rtA}, rtB:{self.rtB}, isOther:{self.isOther}"
        
userChoiceDict = dict()
def GenerateUserChoice(df, i):
    s1 = str(df["StoryOrder1"][i])
    s2 = str(df["StoryOrder2"][i])
    choiceA = int(df["UserANumberChoice"][i])
    choiceB = int(df["UserBNumberChoice"][i])
    rtA = str(df["UserA_choice.rt"][i])
    rtB = str(df["UserB_choice.rt"][i])
    isOther = "other" in str(df["AudioInstruction"][i]).lower()
    return UserChoice(s1, s2, choiceA, choiceB, rtA, rtB,isOther)

def toFilePath(sessionId, fname):
    return fr"{sessionId}/{fname}"

def readCsv(sessionsId):
    fname = [x for x in os.listdir(sessionsId) if x.endswith(".csv")][0]
    csvPath = toFilePath(sessionId, fname)
    df = pd.read_csv(csvPath)
    storyIndexes = np.where(df["trialId"].notnull())[0]
    for i in storyIndexes:
        uc = GenerateUserChoice(df, i)
        key = f"{uc.story1.split('.')[1].lower()}_{int(df['trialId'][i])}"
        userChoiceDict[key] = uc

def log(string):
    print(f"{datetime.now().strftime('%H:%M:%S')} : {string}")

def cleanSpace():
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()

def getSvmTypeString(isSvc):
    if ( isSvc):
        return "svc"
    return "svr"

def getParticiapntString(particiapnt):
    if( particiapnt == 0):
        return "A"
    return "B"

def getPaths(sessionId, correctionA = 0.4, correctionB = 0.4):
    files = [x for x in os.listdir(sessionId) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    return [(toFilePath(sessionId, fileA), correctionA),(toFilePath(sessionId, fileB), correctionB)]

    # if ( sessionId == "20112022_1545"):
    #     return [(toFilePath(sessionId,"20112022_1545_A_Recording_00_SD.edf"),0.4), (toFilePath(sessionId,"20112022_1545_B_Recording_00_SD.edf"), 0.4)]
    # if( sessionId == "06062022_1100"):
    #     return [(toFilePath(sessionId,"06062022_1100_A_Recording_00_SD.edf"), 0.4),(toFilePath(sessionId,"06062022_1100_B_Recording_00_SD.edf"), 0.4)]
    
    
    

num_components = 16
num_channels = 16
states = ["smile", "angry", "blink"]

# ------------------- Parameters ------------------- #
isICA       = True    #othererwise PCA.
isSVC       = True    #otherwise SVR.
sessionId   = "20112022_1545"
stateToAnalyze = "smile"
RmsWindowSizeInMs = 30
# -------------------------------------------------- #

readCsv(sessionId)
paths = getPaths(sessionId, 0.75, 0.25)
A=0
B=1

for participant in [A,B]:
    path, correction = paths[participant]
    cleanSpace()
    # ------------------- READ FILE ------------------- #
    log(f"reading {path}, correction: {correction}")
    f = pyedflib.EdfReader(path)
    Y, freq = EdfAnalyzer.readEdf(f, doButter=True, rmsWindowSizeInMs=RmsWindowSizeInMs)
    anotations_chunks = EdfAnalyzer.getAnnotationChunks(f, correction)
    f.close()
    Y = EdfAnalyzer.RemoveDataUntilStart(anotations_chunks, Y, freq)
    cleanSpace()

    # ------------------- INIT ------------------- #
    w = []
    x = []

    training_set_x = None
    test_set_x = None
    training_set_y = xp.array([])
    test_set_y = xp.array([])
    
    # ------------------- Components ------------------- #
    if ( isICA ):
        log("ICA")
        w, x = EdfAnalyzer.ICA(np.array(Y), num_components)
        log("End ICA")
    else:
        log("PCA")
        w,x = EdfAnalyzer.PCA(xp.array(Y), num_components)
        x = x.get() # convert to numpy for zscore
        log("End PCA")
    
    Y = []
    cleanSpace()

    # ------------------- zscore ------------------- #
    log(f"size x is : {x.shape}")
    log("zscore")

    x = xp.asarray(zscore(x,1))

    # ------------------- divide to sets ------------------- #
    for state in states:
        log(f"start state {state}")
        ticks = EdfAnalyzer.getCallibrationTicks(anotations_chunks, freq, state)
        for tickIndex in range(int(len(ticks)/2)):
            isTest = tickIndex == (int(len(ticks)/2) - 1)
            start = ticks[2*tickIndex]
            end =   ticks[2*tickIndex+1] #could be more efficient
            
            m = xp.array([x[i][start:end] for i in range(num_channels)])
            if ( isTest ):
                if( type(test_set_x) == type(None) ):
                    test_set_x = m
                else:
                    test_set_x = xp.c_[test_set_x, m]
            else:
                if ( type(training_set_x) == type(None)):
                    training_set_x = m
                else:
                    training_set_x = xp.c_[training_set_x, m]
            
            if ( state == stateToAnalyze):
                if ( isTest ):
                    test_set_y = xp.append(test_set_y, xp.ones((1, m.shape[1])))
                else:    
                    training_set_y = xp.append(training_set_y, xp.ones((1, m.shape[1])))
            else:
                if ( isTest ):
                    test_set_y = xp.append(test_set_y, xp.zeros((1, m.shape[1])))
                else:
                    training_set_y = xp.append(training_set_y, xp.zeros((1, m.shape[1])))

    training_set_x = xp.asarray(xp.transpose(training_set_x))
    test_set_x = xp.asarray(xp.transpose(test_set_x))
    cleanSpace()

    # ------------------- SVM - train ------------------- #
    log("start train SVM")
    svclassifier = None
    if ( isSVC ):
        svclassifier = SVC(kernel= 'rbf')
    else:
        svclassifier = SVR(kernel= 'rbf')
    
    svclassifier.fit(training_set_x, training_set_y)
    
    log("end svm training")

    if ( isSVC ):
        y_pred = svclassifier.predict(test_set_x)
        print(confusion_matrix(test_set_y.get(),y_pred.get()))
        print(classification_report(test_set_y.get(),y_pred.get()))
        y_pred = []
    
    test_set_y = []
    training_set_x = []
    training_set_y = []
    cleanSpace()

    # ------------------- SVM - predict ------------------- #
    x = xp.transpose(x)
    cleanSpace()
    log("compute SVM over all the record")

    stateAccrossEdf = svclassifier.predict(x)
    x=[]
    cleanSpace()

    # ------------------- Print results ------------------- #
    printed_choices = []
    summaries = []
    print(anotations_chunks.keys())
    log(path)
    for chunk_id in anotations_chunks:
        chunk_summary = dict()
        chunk_summary["key"] = chunk_id
        if ( "." in chunk_id ):
            ucIdx = chunk_id.split('.')[1]
            if ( ucIdx in userChoiceDict):
                uc = userChoiceDict[ucIdx]
                chunk_summary.update(uc.__dict__)
                if ( ucIdx not in printed_choices):
                    print(uc)
                    printed_choices.append(ucIdx)

        chunk = anotations_chunks[chunk_id]
        start = int(chunk.Start.Time * freq)
        end = int(chunk.End.Time * freq)
        mean_occurences = xp.mean(stateAccrossEdf[start:end])
        chunk_summary["start"] = start
        chunk_summary["end"] = end
        log(f"Id : {chunk_id} ; {stateToAnalyze}s : {mean_occurences}")
        # listOfIndexes.append(start)
        # listOfIndexes.append(end)
        summaries.append(chunk_summary)
    
    # ------------------- Save Files ------------------- #
    resultsPath = os.path.dirname(path)
    resultsPath = os.path.join(resultsPath, stateToAnalyze)
    resultsPath = os.path.join(resultsPath, getSvmTypeString(isSVC))
    if ( not os.path.exists(resultsPath)):
        os.path.makedirs(resultsPath)
    
    savetxt(os.path.join(resultsPath, f"{getParticiapntString(participant)}.csv"), X = stateAccrossEdf, delimiter=",")
    with(open(os.path.join(resultsPath, f"{getParticiapntString(participant)}.list"), 'w') as jsonfile):
        json.dump(summaries, jsonfile)
        
    stateAccrossEdf=[]
