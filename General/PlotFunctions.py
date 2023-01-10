import matplotlib.pyplot as plt

num_components = 16

def fixTicks(ticks):
    s = ticks[0]
    return [ x - s for x in ticks]

def plot(y_val, X, totalSize, index, title, color, ticks): #, maximun_val):
    plot = plt.subplot2grid(totalSize, index, rowspan=2, colspan=2)
    plot.plot(X,y_val, color = color)
    plot.set_title(title)
    if ( len(ticks) > 0 ):
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


def plotDataFromBothChunks(X1, X2, ticks1=[], ticks2=[], title="smiles"):
    total_graph_size = (num_components*2,4)
    #ticks1 = fixTicks(ticks1)
    #ticks2 = fixTicks(ticks2)

    for index in range(num_components):
        y1 = X1[index]
        y2 = X2[index]
        plot(y1, range(0,y1.shape[0]), total_graph_size,(index*2, 0), "component %d" % index, 'r', ticks1 )
        plot(y2, range(0,y2.shape[0]), total_graph_size,(index*2, 2), "component %d" % index, 'b', ticks2 )
        index = index+1
        
    plt.suptitle(title)
    plt.show()
