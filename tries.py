import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions import xwt_coherence

#mat = np.random.randn(10,1000)
#plt.imshow(mat, cmap='hot', aspect='auto', interpolation='gaussian')
#plt.show()
if False:
    y = np.random.randn(5,5) #np.array([[1,2],[3,4]])
    m = np.mean(y, axis=1)

    print(y)
    print(m)
    print( y - np.matrix(m).T)

# a = np.random.randn(1,4000*60).flatten()
# b = np.random.randn(1,4000*60).flatten()

# print ( xwt_coherence(a,b,4000) )

from os import path
import json
p = path.join(r"C:\Liron\neuroscience\PHD\First year experiment - EMG\DataAnalysis\WSL", "20112022_1545", "correct.txt")
with open(p, "r") as f:
    corrections = json.load(f)
print(type(corrections["A"]))
print(corrections["B"])


def GetTrapezoid(start, end, windowInMs = 50):
    d = windowInMs*4
    y = np.array([])
    slope = np.arange(0,d)/d
    y = np.append(y, slope)
    y = np.append(y, np.ones((1, end-start-2*d)))
    y = np.append(y, np.flip(slope))

    return y


start = 3000
end = 3000 + 3*4000

xrange = np.arange(start,end)
yrange = GetTrapezoid(start,end,500)

import matplotlib.pyplot as plt
plt.ylim(top = 1.1)
plt.plot(xrange, yrange)
plt.show()