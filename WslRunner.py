from SmileSvmGpu import PredictSmiles
import os
from HelperFunctions import cleanSpace, log

data_path = r'/mnt/c/Liron/DataEmg'


def isSession(s):
    return "202" in s

def GetSessions(path = data_path):
    return [ s for s in os.listdir(path) if isSession(s) ]

def runSingleSession(session, dataPath = data_path):
    # in the paper - The sound of smile: Auditory biofeedback of facial EMG activity
    # https://doi.org/10.1016/j.displa.2016.09.002
    # They used window of 50MS for sonification and 150MS for the SVM classification.
    PredictSmiles(session, dataPath, True, 50, "smile" )

sessions = GetSessions()
for session in sessions:
    try:
        runSingleSession(session)
        cleanSpace()
    except Exception as e:
        log(str(e))