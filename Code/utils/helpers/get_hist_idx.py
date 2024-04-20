import numpy as np


__all__ = ['get_hist_idx']


def get_hist_idx(descriptors, kmeans, k):
    # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []

    for des in descriptors:
        hist = np.zeros(k)

        idx = kmeans.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)

    return hist_list, idx_list
