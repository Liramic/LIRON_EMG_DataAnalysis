import os
from datetime import datetime
import gc
import cupy as xp
import json
import numpy as np
import pyedflib


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


def getICACsvFileName():
    return "ICafterSharedIca.npz"

def getICACsvFileNameWithPath(data_path, session):
    return os.path.join(os.path.join(data_path, session) ,getICACsvFileName())

def getWeightsFileNameWithPath(data_path, session):
    pathOfSession = os.path.join(data_path, session)
    return os.path.join(pathOfSession, "weightsAfterSharedICA.csv")

def isSession(s):
    return "202" in s

def GetSessions(path):
    return [ s for s in os.listdir(path) if isSession(s) ]

def getPathsFromSessionFolder(fullFolder):
    files = [x for x in os.listdir(fullFolder) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    p = os.path.join(fullFolder, "corrections.txt")
    with open(p, "r") as f:
        corrections = json.load(f)
    
    return [(toFilePath(fullFolder, fileA), corrections["A"]),(toFilePath(fullFolder, fileB), corrections["B"])]
