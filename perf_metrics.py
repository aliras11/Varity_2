'''experiments and tests using different classifier scoring metrics as well as visualizations'''
from sklearn.dummy import DummyClassifier
import sklearn.metrics as skk
from statistics import mean
import numpy as np
import numpy.random as npr
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def create_data(miu1=0,miu2=1,std1=1,std2=1,size=100,prior=0.5):
    '''create a binary classification problem with one feature that is drawn from two normal distributions
    prior -> ratio of total data that is labelled positive'''
    norm1 = npr.normal(miu1,std1,round(size*prior)) #feature for 0 class
    norm2 = npr.normal(miu2,std2,size-round(size*(prior))) #feature for 1 class
    features = np.r_[norm1,norm2]
    features = features[:,np.newaxis]
    labels = np.r_[np.repeat(1,round(size*prior)),np.repeat(0,size-round(size*(prior)))]
    labels = labels[:,np.newaxis]
    data = np.concatenate([features,labels],axis=1)
    return data

def hist_plot(data: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlim(np.min(data[:,0])-0.5,np.max(data[:,0])+0.5)
    ax1.hist(data[(data[:,1]==1)][:,0],label='positive')
    ax1.hist(data[(data[:,1]==0)][:,0],label='negative')
    ax1.legend(loc='upper right')
    ax2.set_xlim(0,1)
    
    vline = ax1.axvline(-2.5, ls='-', color='r', lw=1, zorder=10)
    precision, recall, thresholds = skk.precision_recall_curve((data[:,1]==1),data[:,0])
    precision
    line, = ax2.plot(recall[0],precision[0])

    def animate(i):
        line.set_xdata(recall[:i])
        line.set_ydata(precision[:i])  # update the data.
        vline.set_xdata(thresholds[i])
        return vline, line,

    
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=len(thresholds), interval=500,blit=False)
    plt.show()




data = create_data(0,5,1,1,1000,0.2)
precision, recall, thresholds = skk.precision_recall_curve(data[:,1],data[:,0])
print(thresholds)


