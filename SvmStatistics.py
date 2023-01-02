from typing import Any
from dataclasses import dataclass
import json
import os
import cupy as xp
from HelperFunctions import cleanSpace

def getInt(value):
    if(value==None):
        return None
    return int(value)

def getStr(value):
    if(value == None):
        return None
    return str(value)

def getFloat(value):
    if ( value == None):
        return None
    return float(value)

def getBool(value):
    if ( value == None):
        return None
    return bool(value)

@dataclass
class Event:
    key: str
    story1: str
    story2: str
    choiceA: int
    choiceB: int
    rtA: str
    rtB: str
    isOther: bool
    start: int
    end: int
    mean : float
    std : float

    @staticmethod
    def from_dict(obj: Any) -> 'Event':
        _key = getStr(obj.get("key"))
        _story1 = getStr(obj.get("story1"))
        _story2 = getStr(obj.get("story2"))
        _choiceA = getInt(obj.get("choiceA"))
        _choiceB = getInt(obj.get("choiceB"))
        _rtA = getFloat(obj.get("rtA"))
        _rtB = getFloat(obj.get("rtB"))
        _isOther = getBool(obj.get("isOther"))
        _start = getInt(obj.get("start"))
        _end = getInt(obj.get("end"))
        return Event(_key, _story1, _story2, _choiceA, _choiceB, _rtA, _rtB, _isOther, _start, _end,0,0)

def GetMeasurmentsFromFile(filePath, eventsList : list):
    arr = xp.loadtxt(filePath, delimiter=",")
    for event in eventsList:
        eventData = arr[event.start : event.end]
        event.mean = float(xp.mean(eventData))
        event.std  = float(xp.std(eventData))
        del eventData
        cleanSpace()
    del arr
    cleanSpace()

def getShortName(name):
    threadId = name.split("_")[-1]
    if ".ogg" in name:
        return f"listen_{threadId}"
    elif ".png" in name:
        return f"read_{threadId}"
    else:
        return name.split("_")[0]


def flattenDict(d):
    result = []
    values = d.values()
    for listEvent in values:
        for ev in listEvent:
            result.append(ev)
    return result

def GetAllSessions(working_dir, svmType, state="smile" ):
    sessions = dict()
    for sessionName in [x for x in os.listdir(working_dir) if "202" in x]:
        print(f"started working on session : {sessionName}")
        eventListsForSession = []
        listsPath = os.path.join(working_dir, sessionName,state, svmType)
        for participant in ["A", "B"]:
            eventList = dict()
            with open(os.path.join(listsPath, f"{participant}.list"), "r") as f:
                jarr = json.load(f)
                for item in jarr:
                    #eventList.append(Event.from_dict(item))
                    event = Event.from_dict(item)
                    threadId = getShortName(event.key)
                    if ( threadId in eventList ):
                        eventList[threadId].append(event)
                    else:
                        eventList[threadId] = [event]
            GetMeasurmentsFromFile(os.path.join(listsPath, f"{participant}.csv"), flattenDict(eventList))
            eventListsForSession.append(eventList)
        sessions[sessionName] = eventListsForSession
    return sessions


svmType = "svr" # svc or svr
sessions = GetAllSessions(r"C:\Liron\DataEmg\Done", svmType, "smile")

csvDictionary = dict()
csvDictionary["meanSmiles"] = list()
csvDictionary["stdSmiles"] = list()
csvDictionary["rt"] = list()
csvDictionary["choice"] = list()
csvDictionary["isOther"] = list()
csvDictionary["diffMean"] = list()
A=0
B=1

for key in sessions:
    for participant in [A,B]:
        for eventDictKey in sessions[key][participant]:
            #for event in sessions[key][participant]:
            if ( "listen" in eventDictKey):
                eventlist = sessions[key][participant][eventDictKey]
                for i in range(0,2):
                    event = eventlist[i]
                    if ( "Listening" in event.key ):
                        csvDictionary["meanSmiles"].append(event.mean)
                        csvDictionary["stdSmiles"].append(event.std)
                        csvDictionary["isOther"].append(int(event.isOther))
                        csvDictionary["rt"].append([event.rtA, event.rtB][participant])
                        csvDictionary["diffMean"].append(event.mean - eventlist[(i+1)%2].mean)
                        choice = [event.choiceA, event.choiceB ][participant]
                        if ( choice == 1):
                            if( event.story1.lower() in event.key.lower() ):
                                csvDictionary["choice"].append(1)
                            else:
                                csvDictionary["choice"].append(0)
                        else:
                            if( event.story2.lower() in event.key.lower() ):
                                csvDictionary["choice"].append(1)
                            else:
                                csvDictionary["choice"].append(0)

import pandas as pd
df = pd.DataFrame(csvDictionary)
df.to_csv(os.path.join(r"C:\Liron\DataEmg\Done", f"smiles_mean_over_listening_{svmType}_with_diff_1.csv"))
