import pyedflib
import numpy as np
from scipy.stats import zscore
#from cuml.svm import LinearSVC
from cuml.svm import SVC, SVR
from cuml.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cupy as xp
import os
from cupy import savetxt
from EDF.EdfAnalyzer import EdfAnalyzer
import gc
import pandas as pd
import json
from General.HelperFunctions import toFilePath, getPathsFromSessionFolder, log

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
        
def GenerateUserChoice(df, i):
    s1 = str(df["StoryOrder1"][i])
    s2 = str(df["StoryOrder2"][i])
    choiceA = int(df["UserANumberChoice"][i])
    choiceB = int(df["UserBNumberChoice"][i])
    rtA = str(df["UserA_choice.rt"][i])
    rtB = str(df["UserB_choice.rt"][i])
    isOther = "other" in str(df["AudioInstruction"][i]).lower()
    return UserChoice(s1, s2, choiceA, choiceB, rtA, rtB,isOther)

def readCsv(sessionsId):
    userChoiceDict = dict()
    fname = [x for x in os.listdir(sessionsId) if x.endswith(".csv")][0]
    csvPath = toFilePath(sessionsId, fname)
    df = pd.read_csv(csvPath)
    storyIndexes = np.where(df["trialId"].notnull())[0]
    for i in storyIndexes:
        uc = GenerateUserChoice(df, i)
        key = f"{uc.story1.split('.')[1].lower()}_{int(df['trialId'][i])}"
        userChoiceDict[key] = uc
    return userChoiceDict

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

def getPathsFromSessionFolder(sessionId):
    files = [x for x in os.listdir(sessionId) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    p = os.path.join(".", sessionId, "corrections.txt")
    with open(p, "r") as f:
        corrections = json.load(f)
    
    return [(toFilePath(sessionId, fileA), corrections["A"]),(toFilePath(sessionId, fileB), corrections["B"])]


num_components = 16
num_channels = 16
states = ["smile", "angry", "blink"]
A=0
B=1


def PredictSmiles(sessionId, workingDir = ".", isICA = True, RmsWindowSizeInMs = 50, stateToAnalyze="smile"):
    if ( workingDir!= "."):
        os.chdir(workingDir)
    
    userChoiceDict = readCsv(sessionId)
    paths = getPathsFromSessionFolder(sessionId)

    for participant in [A,B]:
        path, correction = paths[participant]
        cleanSpace()
        # ------------------- READ FILE ------------------- #
        log(f"reading {path}, correction: {correction}")
        f = pyedflib.EdfReader(path)
        Y, freq = EdfAnalyzer.readEdf(f, doButter=True)
        anotations_chunks = EdfAnalyzer.getAnnotationChunks(f, correction, RmsWindowSizeInMs)
        f.close()
        Y = EdfAnalyzer.RemoveDataUntilStart(anotations_chunks, Y, freq)
        cleanSpace()

        # ------------------- INIT ------------------- #
        w = []
        x = []

        tagged_x = None
        tagged_y = xp.array([])
        
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
        
        del Y
        cleanSpace()

        # ------------------- RMS ------------------- #

        #x = EdfAnalyzer.window_rms(x, window_size = RmsWindowSizeInMs)
        x = EdfAnalyzer.window_rms_downsample_no_cut(x, RmsWindowSizeInMs)
        cleanSpace()
        
        # ------------------- zscore ------------------- #
        #log("zscore")
        #x = xp.asarray(zscore(x,1))
        x = xp.asarray(x)
        print(x.shape)

        # ------------------- divide to sets ------------------- #
        for state in states:
            log(f"start state {state}")
            ticks = EdfAnalyzer.getCallibrationTicks(anotations_chunks, freq, state)
            for tickIndex in range(int(len(ticks)/2)):
                start = int(ticks[2*tickIndex])
                end =   int(ticks[2*tickIndex+1])

                m = x[:, start:end]
                
                if ( type(tagged_x) == type(None)):
                    tagged_x = m
                else:
                    tagged_x = xp.append(tagged_x, m, axis=1)
                
                if ( state == stateToAnalyze): 
                    tagged_y = xp.append(tagged_y, xp.ones((1, m.shape[1])))
                else:
                    tagged_y = xp.append(tagged_y, xp.zeros((1, m.shape[1])))

        tagged_x = xp.transpose(tagged_x)
        cleanSpace()
        x = xp.transpose(x)
        cleanSpace()

        X_train, X_test, y_train, y_test = train_test_split(tagged_x, tagged_y, test_size = 0.2, random_state=42)

        # ------------------- SVM - train ------------------- #
        log("start train SVM")
        svm_classifier = None
        svmText = ""

        for isSVC in [True, False]:
            if ( isSVC ):
                svm_classifier = SVC(kernel= 'rbf')
            else:
                svm_classifier = SVR(kernel= 'rbf')
            
            svm_classifier.fit(X_train, y_train)
            
            log("end svm training")

            if ( isSVC ):
                y_pred = svm_classifier.predict(X_test)
                svmText = str(confusion_matrix(y_test.get(),y_pred.get()))
                svmText += "\r\n"
                svmText += classification_report(y_test.get(),y_pred.get())
                print(svmText)
                y_pred = []
            cleanSpace()

            # ------------------- SVM - predict ------------------- #
            log("compute SVM over all the record")

            stateAccrossEdf = svm_classifier.predict(x)
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
                chunk_summary["mean_occurences"] = float(mean_occurences)
                log(f"Id : {chunk_id} ; {stateToAnalyze}s : {mean_occurences}")
                summaries.append(chunk_summary)
            
            # ------------------- Save Files ------------------- #
            resultsPath = os.path.dirname(path)
            resultsPath = os.path.join(resultsPath, stateToAnalyze)
            resultsPath = os.path.join(resultsPath, getSvmTypeString(isSVC))
            if ( not os.path.exists(resultsPath)):
                os.makedirs(resultsPath)
            
            
            savetxt(os.path.join(resultsPath, f"{getParticiapntString(participant)}.csv"), X = stateAccrossEdf, delimiter=",")
            with(open(os.path.join(resultsPath, f"{getParticiapntString(participant)}.list"), 'w') as jsonfile):
                json.dump(summaries, jsonfile)
            if ( isSVC ):
                with (open(os.path.join(resultsPath, f"SVC_Classification_{getParticiapntString(participant)}.txt"), 'w') as f):
                    f.write(svmText)
            
            del stateAccrossEdf
            cleanSpace()
        del x
        del X_train
        del y_train
        del X_test 
        del y_test
        cleanSpace()

