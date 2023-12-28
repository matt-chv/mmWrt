from numpy import array, sqrt, log2, log, pi
from scipy.fft import fft
from numpy import complex_ as complex
from numpy import angle


def error(targets_synthetics, targets_f):
    """ Computes the error in the targets position estimation

    Parameters
    ----------
    targets_synthetics: list[Targets]
        list of synthetic targets (as defined intially)
    targets_f: list[Targets]
        list of targets as computed by rt and rsp

    Returns
    -------
    total_error: float
        sum of distances between each closest targets
    """
    total_error = 0
    # create a local copy to avoid modifying the initial list
    targets_i = targets_synthetics.copy()
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
        if CFAR type is not supported
    """
    # TBD
    if cfar_type == "CA":
        cfar_th = cfar_ca_1d(FT)
    else:
        raise ValueError(f"Unsupported CFAR type: {cfar_type}")

    return cfar_th


def peak_grouping_1d(cfar_idx, mag_r):
    """groups adjacent idx from cfar by first putting adjacent one in clusters
    then finding the index with the highest magnitude in FFT and returning
    this one as peak

    Parameters
    ----------
    cfar_idx: numpy array
        vector of index where fft magnitude is higher than CFAR threshold
    mag_r: numpy array
        abs(FFT)

    Returns
    -------
    idx_peaks: numpy array
        grouped peaks
    """

    cluster = [cfar_idx[0]]
    if cfar_idx.shape[0] > 1:
        idx_peaks = []
    else:
        idx_peaks = [cfar_idx[0]]

    for i in range(1, cfar_idx.shape[0]):
        # iterate to build cluster
        if cfar_idx[i] == cfar_idx[i-1]+1:
            cluster.append(cfar_idx[i])
            if i < cfar_idx.shape[0]-1:
                continue
        # here process cluster to find highest peak
        mag_max = 0
        idx_max = 0
        for idx in cluster:
            if mag_r[idx] > mag_max:
                mag_max = mag_r[idx]
                idx_max = idx
        idx_peaks.append(idx_max)
        cluster = []
    return idx_peaks


def range_fft(baseband, chirp_index=0, fft_window=None, fft_padding=0):
    """ scipy FFT wrapper with windowing and padding options

    Parameters
    ----------
    baseband: numpy array
        the IF ADC signals data matrix
    chirp_index: int
        index of the chirp in the data matrix
    fft_window: str
        FFT windowing names supported by scipy get_window
    fft_padding: int
        if 0 - no padding
        if -1: padding to next level of power of 2
        other values: padding to those values

    Returns
    -------
    Range_FFT: tuple
        Distances: np array
        abs_FT: np array

    Raises
    ------
    ValueError
        when fft_padding has a value < -1
    """
    if chirp_index == 0:
        # v0.1.1: adc = baseband['adc_cube'][0][0][0]
        adc = baseband['adc_cube'][0, chirp_index, 0, 0, :]
    else:
        raise ValueError("chirp index value not supported yet")

    if fft_padding == -1:
        fft_length = 2**int(log2(len(adc)) + 1)
    elif fft_padding == 0:
        fft_length = len(adc)
    elif fft_padding < 0:
        raise ValueError(f"Unsupported fft padding value with : {fft_padding}")
    else:
        fft_length = fft_padding

    if fft_window is None:
        FT = fft(adc, n=fft_length)
    else:
        from scipy.signal import get_window
        w = get_window(fft_window, len(adc))
        FT = fft(adc * w, n=fft_length)
    if baseband["datatype"] == complex:
        pass
    else:
        # return half of FFT for real bb signal
        FT = FT[:len(FT)//2]
    Distances = [i * baseband["fs"] / adc.shape[0] *
                 baseband["v"]/2/baseband["slope"]
                 for i in range(len(FT))]
    Range_FFT = (Distances, FT)
    return Range_FFT

def plot_range_doppler():
    pass


def __quinnsecond__(FT, k):
    """ Provide frequency estimator via Quinn's second estimate

    Parameters
    ----------
      FT: numpy array
        Fourier Transform with complex values
      k: int
        the index of the range bin where
        the frequency estimator needs to be applied
    Returns
    --------
      d: float
        offset from k for more accurate frequency estimate

    Details:
    --------
      C code source from
       https://gist.github.com/hiromorozumi/f74fd4d5592a7f79028560cb2922d05f
       out[k][0]  ... real part of FFT output at bin k
       out[k][1]  ... imaginary part of FFT output at bin k
    c++ code:
    divider = pow(out[k][0], 2.0) + pow(out[k][1], 2.0);
    ap = (out[k+1][0] * out[k][0] + out[k+1][1] * out[k][1]) / divider;
    dp = -ap  / (1.0 - ap);
    am = (out[k-1][0] * out[k][0] + out[k-1][1] * out[k][1]) / divider;

    dm = am / (1.0 - am);
    d = (dp + dm) / 2 + tau(dp * dp) - tau(dm * dm);
    """
    out = [[z.real, z.imag] for z in FT]

    def tau(x):
        return 1 / 4 * log(3 * x ** 2 + 6 * x + 1) - sqrt(6) / 24 * log((x + 1 - sqrt(2 / 3)) / (x + 1 + sqrt(2 / 3)))  # noqa 501

    divider = out[k][0] ** 2.0 + out[k][1] ** 2
    ap = (out[k + 1][0] * out[k][0] + out[k + 1][1] * out[k][1]) / divider
    dp = -ap / (1.0 - ap)
    am = (out[k - 1][0] * out[k][0] + out[k - 1][1] * out[k][1]) / divider

    dm = am / (1.0 - am)
    d = (dp + dm) / 2 + tau(dp * dp) - tau(dm * dm)
    return d


def __phase_estimator__(FT, k):
    """ Provide frequency estimator via phase method - DOES NOT WORK

    Parameters
    ----------
      FT: numpy array
        Fourier Transform with complex values
      k: int
        the index of the range bin where
        the frequency estimator needs to be applied
    Returns
    --------
      d: float
        offset from k for more accurate frequency estimate
    """
    d = angle(FT[k]) / pi
    # n_samples = len(FT)
    # d = (phi) * (n_samples) / (n_samples - 1)
    return d


def frequency_estimator(FFT, idxs, estimator_name="fft"):
    """ Wrapper around the different frequency estimator possible

    Parameters
    ----------
    FFT: numpy array
        Fourier Transform with complex values
    idxs: List[int]
        list of indexes where peaks in FFT are found and where the
        frequency estimator `estimator_name` needs to be applied
    estimator_name: str
        fft
        phase
        quinn_second

    Returns
    -------
    i_peaks: numpy array
        array of estimated float index from the int idxs

    Raises
    ------
    ValueError  # noqa: DAR402
        when invalid estimator_name value is passed as parameter
    """
    def __estimator(estimator_name, FFT, idx):
        if estimator_name == "fft":
            return 0
        elif estimator_name == "quinn2":
            return __quinnsecond__(FFT, idx)
        else:
            log_msg = f"Unsupported  estimator named: {estimator_name}"
            raise ValueError(log_msg)
    i_peaks = []
    for idx in idxs:
        d = __estimator(estimator_name, FFT, idx)
        idx_est = (idx + d)
        i_peaks.append(idx_est)
    i_peaks = array(i_peaks)
    return i_peaks
