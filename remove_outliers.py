import numpy as np
from sklearn.decomposition import PCA

def remove_outliers(x,y,outmethod):
    
    if outmethod == 'std':
        K = 4
        badidx = ((y > np.nanmedian(y) + K*np.nanstd(y)) | (y < np.nanmedian(y) - K*np.nanstd(y)))
        return (x[~badidx],y[~badidx])
    
    elif outmethod == 'pca':
        X = np.vstack((x,y)).T
        X = (X - np.nanmean(X,axis = 0))/np.nanstd(X,axis = 0)

        pca = PCA(n_components = 2)
        pca.fit(X)
        
        S = pca.transform(X)
        ci95 = 1.96*np.std(S[:,1]) #Can fiddle with this to do more or less than 1.96
        indremove = ((S[:,1] > ci95) | (S[:,1] < -ci95))
        return (x[~indremove],y[~indremove])
    
    else:
        return (x,y)