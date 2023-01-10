import pyedflib
from scipy import signal
import numpy as np
#from scipy.signal import butter, lfilter
from sklearn.decomposition import FastICA
#from cuml.decomposition import PCA
from sklearn.decomposition import PCA
import cupy as xp
from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt
from General.HelperFunctions import isMyAnnotation, cleanSpace

#from mne.preprocessing import ICA
#from picard import picard

# def reduceLastComponent(Y):
#     last = Y[8, :]
#     for i in range(0, Y.shape[0]):
#         Y[i, : ] = Y[i, : ] - last

# def butter_bandpass(lowcut, highcut, fs, order=5):
#     return butter(order, [lowcut, highcut], fs=fs, btype='band')

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype='band')
        #sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        #return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        # sos = butter_bandpass(lowcut, highcut, fs, order=order)
        # y = sosfiltfilt(sos, data)
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        y = filtfilt(b,a,data)
        return y

def window_rms_single(a, window_size = 800):
    a2 = np.power(np.array(a),2)
    window = np.ones(window_size)/float(window_size)
    result =  np.sqrt(np.convolve(a2, window, 'valid'))
    cleanSpace()
    return result

def NotchFilterSignal(noisySignal, sampling_rate, removed_frequency=50.0, Q_Factor=30.0):
    # Design notch filter
    b_notch, a_notch = signal.iirnotch(removed_frequency, Q_Factor, sampling_rate)
    return signal.filtfilt(b_notch, a_notch, noisySignal)

def smooth_single(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth(y_arr, box_pts):
    return np.array([smooth_single(y,box_pts) for y in y_arr])

def ExtractIdAndPos(s):
    splitted = s.split("_")
    return ( int(splitted[1]), splitted[2])

def extractAnnotation(s : str, timing):
    s = s.lower()
    splitted = s.split("_")
    a = Annotation()
    a.Time = timing
    if ( splitted[0] in ["smile", "angry", "blink"]):
        a.Type = splitted[0]
        a.TrialId = int(splitted[1])
        a.order = int(splitted[1])
        a.isStart = splitted[2] == "start"
        return a

    if (splitted[0][0] == "a"):
        a.Type = "Listening"
    elif ( splitted[0][0] == "r"):
        a.Type = "Reading"
    
    if(splitted[0][1] == "s"):
        a.isStart = True
    else:
        a.isStart = False
    
    a.order = int(splitted[1])
    a.TrialId = int(splitted[3])
    a.Story = splitted[5]
    return a

def GetCallibrationsTicks(chunks, freq, calibration_state_to_analyze, num_repeats = 3):
    ticks = []
    for i in range(0,num_repeats):
        key = '%s__%d' % (calibration_state_to_analyze, i)
        chunk = chunks[key]
        ticks.append(int(chunk.Start.Time * freq))
        ticks.append(int(chunk.End.Time * freq))
    
    return ticks

class EdfAnalyzer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def readEdf(f : pyedflib.EdfReader, doButter = True):
        sampling_rate = int(f.getSampleFrequency(0))
        n = 16 #f.signals_in_file
        num_sets = 1
        size_limit = int(f.getNSamples()[0] / num_sets)
        #read EDF
        sigbufs = np.zeros((n, size_limit))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i, start=0, n=size_limit)
        
        #Filter before ICA
        sigbufs = NotchFilterSignal(sigbufs, sampling_rate)
        if (doButter):
            sigbufs = butter_bandpass_filter(sigbufs, 20.0, 400.0, sampling_rate)
        
        return sigbufs, sampling_rate
    
    @staticmethod
    def ICA(matrix, n_components):
        transformer = FastICA(n_components=n_components, random_state=0 )
        x= np.transpose(transformer.fit_transform(np.transpose(matrix))) #FAST ICA
        w = transformer.components_ #np.linalg.inv(transformer.mixing_) #transformer.components_ # np.dot(unmixing_matrix, self.whitening_)
        #K, W, X = picard(matrix, n_components=n_components, ortho=True, extended=True, max_iter=200)
        return w, x #np.dot(W,K) ,X
    
    @staticmethod
    def PCA(matrix, n_components):
        transformer = PCA(n_components=n_components, random_state=0)
        x = xp.transpose(transformer.fit_transform(xp.transpose(matrix)))
        w = transformer.components_
        return w,x

    @staticmethod
    def reduceMeanFromEachCol(Y):
        colMeans = np.mean(Y, axis=0)
        # for i in range(0, Y.shape[1]):
        #     Y[:,i] = Y[:,i] - colMeans[i]
        return Y - colMeans
    
    @staticmethod
    def getIntervalWithWindowReduction(Y, start, end, window=2000):
        meanVec = np.mean(Y[:, start-window : start], axis=1)
        return Y[:, start:end] - np.matrix(meanVec).T
    
    def getIntervalWithWindowDivision(Y, start, end, window=2000):
        meanVec = np.mean(Y[:, start-window : start], axis=1)
        return Y[:, start:end] / np.matrix(meanVec).T

    @staticmethod
    def window_rms(a, window_size=200):
        window_size = window_size*4 # divide by 1000, and multiply by 4000
        return np.array([window_rms_single(x,window_size) for x in a])

    @staticmethod
    def window_rms_downsample(a, window_size_in_ms):
        window_size = window_size_in_ms*4 # divide by 1000, and multiply by 4000
        windows = np.array_split(a, a.shape[1] // (window_size/2), axis=1)
        rms_values = None
        for i in range(len(windows)-1):
            window = np.append(windows[i], windows[i+1], axis=1)
            rms = np.sqrt(np.mean(window**2, axis=1))
            if i == 0:
                rms_values = rms
            else:
                rms_values = np.append(rms_values, rms)
        return np.reshape(rms_values, (16, len(rms_values)//16))

    def window_rms_downsample_no_cut(a, window_size_in_ms):
        window_size = window_size_in_ms*4 # divide by 1000, and multiply by 4000
        windows = np.array_split(a, a.shape[1] // (window_size), axis=1)
        rms_values = np.array([])
        for window in windows:
            rms_values = np.append(rms_values, np.sqrt(np.mean(window**2, axis=1)))
        return np.reshape(rms_values, (16, len(rms_values)//16))
    
    
    def window_rms_downsample_no_cut_faster(a, window_size_in_ms):
        window_size = window_size_in_ms*4 # divide by 1000, and multiply by 4000
        rms_values = np.array([])
        squared = a**2
        for i in range(0, a.shape[1]-window_size , window_size):
            rms_values = np.append(rms_values, np.sqrt(np.mean(squared[:, i:i+window_size], axis=1)))
        return np.reshape(rms_values, (16, len(rms_values)//16))

    @staticmethod
    def getCallibrationTicks(chunks, freq, calibration_state_to_analyze = "smile"):
        return GetCallibrationsTicks(chunks, freq, calibration_state_to_analyze)

    startCorrectionTime = 0

    @staticmethod
    def getAnnotationChunks(f : pyedflib.EdfReader, correctBy, RmsDownsampleWindow = 0):
        annotations = f.readAnnotations()
        file_sampling_rate = int(f.getSampleFrequency(0))
        startIndex = np.where(annotations[2] == "StartExperiment")[0][-1]
        #correctByIdx = np.where(annotations[2] == "Recording Started")[0][0]
        endIndex = len(annotations[2])
        startCorrectIdx = np.where(annotations[2] == "Smile_0_start")[0][0]
        __class__.startCorrectionTime = annotations[0][startCorrectIdx] - 60
        #correctBy = annotations[0][correctByIdx]
        chunks = dict()
        #chunks["ExperimentStartTime"] = startExperimentTime - correctBy

        #Should I notice that recording startes is not exactly at 0.00

        for i in range(startIndex+1, endIndex):
            annotation = str(annotations[2][i])
            if ( not isMyAnnotation(annotation)):
                continue
            timing = annotations[0][i] - correctBy - __class__.startCorrectionTime # fix error by xTrodes.
            # Todo - return anyway a timing*sr - this makes much more sense...
            if ( RmsDownsampleWindow != 0 ):
                timing = timing*file_sampling_rate/(int(4*RmsDownsampleWindow)) # 4*RmsDownsampleWindow // 2 if cut, otherwise - without the division -- 4*RmsDownSampleWindow
            extractedAnnotation = extractAnnotation(annotation, timing)
            index = "%s_%s_%d" % (extractedAnnotation.Type, extractedAnnotation.Story, extractedAnnotation.TrialId)
            if ( index in chunks):
                if (extractedAnnotation.isStart):
                    chunks[index].Start = extractedAnnotation
                else:
                    chunks[index].End = extractedAnnotation
            else:
                chunks[index] = Chunk()
                if(extractedAnnotation.isStart):
                    chunks[index].Start = extractedAnnotation
                else:
                    chunks[index].End = extractedAnnotation
        
        return chunks

    @staticmethod
    def RemoveDataUntilStart(chunks, x, freq):
        #start = chunks.pop("ExperimentStartTime")
        start = __class__.startCorrectionTime
        print(f"start is : {start}")
        return x[:, int(start*freq):]


class Chunk:
    def __init__(self) -> None:
        self.Start = None
        self.End = None
        self.Data = None

class Annotation:
    def __init__(self) -> None:
        self.Type = ""
        self.Story = ""
        self.Reader = ""
        self.order = None
        self.TrialId = None
        self.isStart = False
        self.Time = None
