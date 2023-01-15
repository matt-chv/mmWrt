from numpy import array, sqrt
from scipy.fft import fft
from numpy import complex_ as complex


def error(targets_i, targets_f):
    """ Computes the error in the targets position estimation

    Parameters
    ----------
    targets_i: list[Targets]
        list of targets as defined intially
    targets_f: list[Targets]
        list of targets as computed by rt and rsp

    Returns
    -------
    total_error: float
        sum of distances between each closest targets
    """
    total_error = 0
    if len(targets_f) > 0:
        for t in targets_f:
            err0 = t.distance(targets_i[0])
            idx0 = 0
            for idx, ti in enumerate(targets_i):
                err = t.distance(ti)
                if err < err0:
                    err0 = err
                    idx0 = idx
            total_error += err0
            targets_i.pop(idx0)
            if len(targets_i) == 0:
                break

    # if less targets found than inserted
    # add the remaining ones to the error
    for t in targets_i:
        total_error += t.distance()

    # FIXME: add here code in case missing targets or
    # excessive targets in the found target list

    return total_error


def cfar_ca_1d(X, count_train_cells=10, count_guard_cells=2,
               Pfa=1e-2):
    """ Retuns indexs of peaks found via CA-CFAR
    i.e Cell Averaging Constant False Alarm Rate algorithm

    Parameters
    ----------
    X: numpy ndarray
        signal whose peaks have to be detected and reported
    count_train_cells : int
        number of cells used to train CA-CFAR
    count_guard_cells : int
        number of cells guarding CUT against noise power calculation
    Pfa : float
        Probability of false alert, used to compute the variable threshold

    Returns
    --------
    cfar_th : numpy array
        CFAR threshold values
    """
    signal_length = X.size
    M = count_train_cells
    half_M = round(M / 2)
    count_leading_guard_cells = round(count_guard_cells / 2)
    half_window_size = half_M + count_leading_guard_cells
    # compute power of signal
    P = [abs(x)**2 for x in X]

    # T scaling factor for threshold
    # from Eq 6, Eq 7 from [1]
    # T = M*(Pfa**(-1/M) - 1)**M
    T = M*(Pfa**(-1/M) - 1)

    peak_locations = []
    thresholds = [0]*(half_window_size)
    for i in range(half_window_size, signal_length - half_window_size):
        p_noise = sum(P[i - half_M: i + half_M + 1])
        p_noise -= sum(P[i - count_leading_guard_cells:
                       i + count_leading_guard_cells + 1])
        p_noise = p_noise / M
        threshold = T * p_noise
        thresholds.append(sqrt(threshold))
        if P[i] > threshold:
            peak_locations.append(i)
    peak_locations = array(peak_locations, dtype=int)

    cfar_th = array(thresholds + [0]*(half_window_size))
    return cfar_th


def cfar_1d(cfar_type, FT):
    """ CFAR for 1D FFT values

    Parameters
    ----------
    cfar_type: str
        valid value CA, OS, GO
    FT: ndarray
        signal whose peaks have to be detected and reported

    Returns
    -------
    cfar_th : numpy array
        CFAR threshold values

    Raises
    ------
    ValueError
        if cfar type is not supported
    """
    # TBD
    if cfar_type == "CA":
        cfar_th = cfar_ca_1d(FT)
    else:
        raise ValueError(f"Unsupported CFAR type: {cfar_type}")

    return cfar_th


def peak_grouping_1d(cfar_idx):
    # groups adjacent idx from cfar
    new_idx = [cfar_idx[0]]
    for i in range(1, cfar_idx.shape[0]):
        if cfar_idx[i] == cfar_idx[i-1]+1:
            pass
        else:
            new_idx.append(cfar_idx[i])
    return new_idx


def range_fft(baseband, chirp_index=0, fft_window=None):
    if chirp_index == 0:
        adc = baseband['adc_cube'][0][0][0]
    else:
        raise ValueError("chirp index value not supported yet")
    FT = fft(adc)
    if baseband["datatype"] == complex:
        pass
    else:
        # return half of FFT for real bb signal
        FT = FT[:len(FT)//2]
    Distances = [i * baseband["fs"] / adc.shape[0] *
                 baseband["v"]/2/baseband["slope"]
                 for i in range(len(FT))]
    Range_FFT = (Distances, abs(FT))
    return Range_FFT
