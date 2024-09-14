import numpy as np


def smoothing(kernel_size, data):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode="same")


def make_next_data(time_series, n, ndata, dn=1):
    """times_series is a np 1D array, n-1 number of input points, dn space between them? Produces dataset with length ndata.
    For each window of input+output, substract the value before to normalise the data. first returned array is to get back the original normalisation later.
    """
    l = len(time_series) - dn * n - dn
    return np.array([[time_series[dn + i * (l // ndata) + j * dn - dn, :] for j in range(n)] for i in range(ndata)]), np.array(
        [
            [time_series[dn + i * (l // ndata) + j * dn, :] - time_series[dn + i * (l // ndata) + j * dn - dn, :] for j in range(n)]
            for i in range(ndata)
        ]
    )
