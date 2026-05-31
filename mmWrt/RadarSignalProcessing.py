from numpy import append, array, concatenate, log2, log, pi, sqrt, where, zeros
from numpy import abs as np_abs
import numpy as np
from numpy import sum as np_sum
from numpy.typing import NDArray
from scipy.fft import fft, fft2
from scipy.signal import find_peaks
from numpy import angle

from .Scene import Radar


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
        # d = t.distance()
        total_error += t.distance()

    # FIXME: add here code in case missing targets or
    # excessive targets in the found target list

    return total_error

def cfar_1D_convolve(X, num_training_cells = 10,
                     num_guard_cells=2, Pfa=0.1,
                     mode="same", debug=False):
    """ CFAR implementation via convolution the idea is to see 
    CFAR as convolution with a kernel of 0 for guard an CuT cells and 1 for train cells, 
    then scaling the output of the convolution by T/M for Pfa"""
    from numpy import ones, convolve, pad
    test = ones(len(X))
    n = 2*num_training_cells+num_guard_cells
    M = n
    T = M*(Pfa**(-1/M) - 1)
    P = [x**2 for x in X]
    if n>len(P):
        n=len(P)
    exclude = ones(n)
    extra_left = (len(P) - n )//2
    extra_right = extra_left
    
    if extra_right+extra_left+n<len(P):
        extra_right +=1
    exclude2 = pad(exclude, ((extra_left),(extra_right)), mode='constant', constant_values=0)
    test = test - exclude2
    th = convolve(P, test, mode)
    th = sqrt(th) * T/M
    if debug:
        import matplotlib.pyplot as plt
        pass
    return th


def cfar_ca_1d(X, num_training_cells=10,
               num_guard_cells=2,
               Pfa=1e-2,
               mode="same",
               debug=True):
    """ Retuns indexs of peaks found via CA-CFAR
    i.e Cell Averaging Constant False Alarm Rate algorithm

    Parameters
    ----------
    X: numpy ndarray
        signal whose peaks have to be detected and reported
    num_training_cells : int
        number of cells used to train CA-CFAR
    num_guard_cells : int
        number of cells guarding CUT against noise power calculation
    Pfa : float
        Probability of false alert, used to compute the variable threshold
    mode: str
        same meaning as np.convolve
    debug: bool
        if True will output debug info

    Returns
    --------
    cfar_th : numpy array
        CFAR threshold values
    """
    signal_length = X.size
    M = num_training_cells
    half_M = round(M / 2)
    count_leading_guard_cells = round(num_guard_cells / 2)
    half_window_size = half_M + count_leading_guard_cells
    # compute power of signal
    P = np_abs(X)  # [abs(x)**2 for x in X]

    if mode == "same":
        # for same mode, we need to pad the signal at the beginning and end
        # to be able to compute the CFAR threshold for the first and last samples
        P_extended = concatenate((P[-half_window_size:], P, P[:half_window_size]))

    # T scaling factor for threshold
    # from Eq 6, Eq 7 from [1]
    # T = M*(Pfa**(-1/M) - 1)**M
    T = M*(Pfa**(-1/M) - 1)

    peak_locations = []
    # thresholds = [0]*(half_window_size)
    thresholds = zeros(signal_length)


    for idx in range(0, signal_length):
        p_noise = np_sum(P_extended[half_window_size+idx - half_M: half_window_size+idx + half_M + 1])
        p_noise -= np_sum(P_extended[half_window_size+idx - count_leading_guard_cells:
                    half_window_size+idx + count_leading_guard_cells + 1])
        if debug:
            print("MCV 105 M", M)
            print("MCV 106", p_noise)
            print("MCV 107", p_noise/M)
        p_noise = p_noise / M
        threshold = T * p_noise
        if debug:
            print("idx, th", idx, threshold)
            print("idx pnoise", p_noise)
            print("P idx", P[idx])
        # thresholds.append(sqrt(threshold))
        thresholds[idx] = sqrt(threshold)
        if P[idx] > threshold:
            peak_locations.append(idx)
    peak_locations = array(peak_locations, dtype=int)
    if debug:
        import matplotlib.pyplot as plt
        plt.plot(X, label="signal")
        plt.plot(thresholds, label="CFAR threshold")
        plt.show()
        print("CFAR CA debug---")
        print("signal length", X.shape)
        print("debug cfar CA mag", abs(X))
        print("tresholds", thresholds)
        print("max(thresholds)", max(thresholds))
        print("T", T)
        print("E", sum(P)/M)
        print("th=sqrt(T*sum(P)/M)",sqrt(T*sum(P)/M))
        print("peaks", peak_locations)
        print("CFAR CA TH shape", thresholds.shape)
        print("---CFAR CA EOM")
    cfar_th = thresholds
    return cfar_th


def cfar_1d(FT,
            num_training_cells=10,
            num_guard_cells=2, mode="same",
            cfar_type="CA",
            debug=False):
    """ CFAR for 1D FFT values

    Parameters
    ----------
    FT: ndarray
        signal whose peaks have to be detected and reported
    num_training_cells: int
        sum of left and right train cells count
    num_guard_cells: int
        sum of left and right guard cells count
    mode: str
        same as np.convolve
    cfar_type: str
        valid value CA, OS, GO
    debug: bool
        if True outputs debug info

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
        if mode == "full":
            zero_train = zeros(num_training_cells+num_guard_cells+FT.shape[0])\
                .astype(complex)
            zero_train[(num_training_cells+num_guard_cells)//2:
                       -(num_training_cells+num_guard_cells)//2] = FT
            FT = zero_train
        cfar_th = cfar_ca_1d(FT, num_training_cells=num_training_cells,
                             num_guard_cells=num_guard_cells,
                             debug=debug)
        if mode == "full":
            cfar_th = cfar_th[(num_training_cells+num_guard_cells)//2:
                              -(num_training_cells+num_guard_cells)//2]
    else:
        raise ValueError(f"Unsupported CFAR type: {cfar_type}")

    if debug:
        print("CFAR 1D debug---")
        print("CFAR_TH shape", cfar_th.shape)
    return cfar_th


def cfar_ca(fft_values: np.ndarray,
            guard_cell_count: int,
            train_cell_count: int,
            pfa: float) -> np.ndarray:
    """
    Constant False Alarm Rate (CFAR) detector.
    Computes CFAR threshold for each range bin by averaging surrounding training
    cells (excluding guard cells) to estimate background noise level. The threshold
    is set based on the desired probability of false alarm (PFA).

    Parameters
    ----------
    fft_values
        Complex FFT values (1D array).
    guard_cell_count
        Number of guard cells on each side of the cell under test.
    train_cell_count
        Number of training cells on each side (excluding guard cells).
    pfa:
        Probability of False Alarm (0 < pfa < 1).

    Returns
    -------
    cfar_thresholds
        CFAR threshold for each range bin (same shape as fft_values).

    Notes
    -----
    The CFAR algorithm:
    1. For each cell under test, identify guard cells (immediately adjacent) and
       training cells (surrounding the guard cells).
    2. Compute mean power from training cells to estimate background noise.
    3. Set threshold = noise_mean * threshold_factor, where threshold_factor is
       derived from the desired PFA using chi-square statistics.

    Examples
    --------
    >>> import numpy as np
    >>> # Example 1: Peak at first range bin
    >>> fft = np.array([100.0+0j, 1.0+0j, 1.0+0j, 1.0+0j, 1.0+0j, 1.0+0j])
    >>> threshold = cfar(fft, guard_cell_count=1, train_cell_count=2, pfa=0.01)
    >>> magnitude = np.abs(fft)
    >>> peak_idx = np.argmax(magnitude)
    >>> peak_idx
    0
    >>> np.where(magnitude > threshold)[0][0]
    0
    >>> # Example 2: Peak in the middle
    >>> fft = np.array([1.0+0j, 1.0+0j, 100.0+0j, 1.0+0j, 1.0+0j, 1.0+0j])
    >>> threshold = cfar(fft, guard_cell_count=1, train_cell_count=2, pfa=0.01)
    >>> magnitude = np.abs(fft)
    >>> peak_idx = np.argmax(magnitude)
    >>> peak_idx
    2
    >>> np.where(magnitude > threshold)[0][0]
    2
    """
    # Convert complex FFT values to magnitude
    magnitude = np.abs(fft_values)
    n_bins = len(magnitude)
    # Compute threshold factor from PFA
    # Using chi-square approximation: for single cell test in noise,
    # threshold_factor ≈ -ln(pfa)
    num_train_cells = 2 * train_cell_count
    threshold_factor = -np.log(pfa) / num_train_cells

    # Initialize threshold array
    cfar_threshold = np.zeros(n_bins, dtype=float)

    # Compute threshold for each bin
    for i in range(n_bins):
        # Define indices for training cells on the left
        left_start = i - train_cell_count - guard_cell_count
        left_end = i - guard_cell_count
        # Define indices for training cells on the right
        right_start = i + guard_cell_count + 1
        right_end = i + guard_cell_count + train_cell_count + 1
        # Collect valid training cell indices
        train_indices = []
        # FIXME: need to handle teh case when left_end or right_end out of range
        if left_start >= 0:
            train_indices.extend(range(left_start, left_end))
        else:
            train_indices.extend(range(0, left_end))
            train_indices.extend(range(n_bins + left_start, n_bins))
        if right_end <= n_bins:
            train_indices.extend(range(right_start, right_end))
        else:
            train_indices.extend(range(right_start, n_bins))
            train_indices.extend(range(0, right_end - n_bins))

        # Compute mean noise power from available training cells
        if train_indices:
            mean_noise = np.mean(magnitude[train_indices])
        else:
            # Fallback if no training cells available
            # mean_noise = np.mean(magnitude)
            raise ValueError("No Training cells for CFAR")
        # Compute CFAR threshold
        cfar_threshold[i] = mean_noise * threshold_factor
    return cfar_threshold


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
    # else:
    #    idx_peaks = [cfar_idx[0]]

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


def range_resolution(v, B):
    """ Range resolution is c/2B

    Parameters
    ----------
    v: float
        celerity of light in medium
    B: float
        Bandwidth of signal sampled (often simplified as chirped)

    Returns
    -------
    delta_R: float
        Range Resolution
    """
    delta_R = v/2/B
    return delta_R


def if2d(radar):
    """ ratio from IF frequency to distance
    !!! important

        the ratio is 1/2 of the d2f as the IF frequency results from the wave
        traveling to the target and back. Whereas if2d gives the distance
        between the radar and the scatterer which is 1/2 the distance
        travelled by the radar EM wave.

    Parameters
    ----------
    radar: object
        a radar object

    Returns
    --------
    f2d: float
        ratio between frequency and distance for given radar
        settings

    Usage
    -----
    f2d = if2d(radar)
    # assuming f_if is an IF frequency
    # then d will be the distsance to the target
    d = f2d * f_if
    """

    f2d = radar.v/2/radar.slope
    return f2d


def range_fft(adc_values: NDArray,
              baseband: dict,
              chirp_index=0,
              fft_window=None, fft_padding=0,
              full_FFT=False,
              debug=False):
    """ scipy FFT wrapper with windowing and padding options

    Parameters
    ----------
    adc_values: NDArray
        the IF ADC signals of shape(N,) - i.e. 1D array
    baseband: dict
    chirp_index: int
        (obsolete) index of the chirp in the data matrix
    fft_window: str
        FFT windowing names supported by scipy get_window
    fft_padding: int
        if 0 - no padding
        if -1: padding to next level of power of 2
        other values: padding to those values
    full_FFT: bool
        if True returns the full FFT, else only 0..d_max_unambiguous
    debug: bool
        if True logs debug information on console

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
    """if chirp_index == 0:
        # v0.1.1: adc = baseband['adc_cube'][0][0][0]
        frame_idx = 0
        rx_idx = 0
        tx_idx = 0
        adc = baseband['adc_cube'][frame_idx, chirp_index, tx_idx, rx_idx, :]
    else:
        raise ValueError("chirp index value not supported yet")"""
    adc = adc_values

    if fft_padding == -1:
        if debug:  # pragma: no cover
            print("padding FFT to next **2")
        fft_length = 2**int(log2(len(adc)) + 1)
    elif fft_padding == 0:
        if debug:  # pragma: no cover
            print(f"no FFT padding, using len: {len(adc)}")
        fft_length = len(adc)
    elif fft_padding < -1:
        raise ValueError(f"Unsupported fft padding value with : {fft_padding}")
    else:
        if debug:  # pragma: no cover
            print(f"padding up to len: {fft_padding} as opposed " +
                  f"to adc len of: {len(adc)}")
        fft_length = fft_padding

    if fft_window is None:
        if debug:  # pragma: no cover
            print("FFT without windowing")
            print("fft_leng", fft_length)
        FT = fft(adc, n=fft_length)
    else:
        if debug:  # pragma: no cover
            print(f"FFT windowing, using: {fft_window}")
        from scipy.signal import get_window
        w = get_window(fft_window, len(adc))
        FT = fft(adc * w, n=fft_length)

    chirp_bandwidth = baseband["chirp_slope"] * baseband["chirp_end_time"]

    # baseband = {"fs":adc_sampling_rate}
    delta_R = range_resolution(baseband["medium_velocity"], chirp_bandwidth)
    # D_max = c*f_if_max/(2*S)
    # if complex FFT, f_if_max = fs
    # if real FFT, f_if_max = fs/2 (for non-ambiguous)
    delta_R_FFT = baseband["adc_sample_rate"] * baseband["medium_velocity"] \
        / (2 * len(FT) * baseband["chirp_slope"])
    Distances = [i * delta_R_FFT for i in range(len(FT))]

    if debug:  # pragma: no cover
        print(f"Range Resolution: {delta_R:.2g}, based on chirping")
        print(f"Range resolution based on sampling:{delta_R_FFT:.2g}")

    if full_FFT:
        if debug:  # pragma: no cover
            print("FULL FFT")
    else:
        # return half of FFT for real bb signal
        if debug:  # pragma: no cover
            print("returning only half of FFT (non ambiguous ranges/volicity)")
        FT = FT[:len(FT)//2]
        Distances = Distances[:len(Distances)//2]

    Range_FFT = (Distances, FT)
    return Range_FFT


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


def ranges_from_fft_threshold(adc_values:NDArray, chirp_slope:float,
          adc_sample_rate:float,fft_threshold:float) -> NDArray:
    """ returns a NDArray of ranges using a simple fft threshold for target 
    detection, used for simple examples.
    Not recommended in most cases, cfar peak detection recommended 

    Parameters
    ----------
    adc_values: NDArray
        the ADC values for a given chirp. assumed to be a 1D (adc_samples_per_chirp,) shape
    chirp_slope: float
        the chirp slope in Hz/s
    adc_sample_rate: float
        the ADC sampling rate in Hz
    fft_threshold: float
        threshold used by find peaks for peak detection

    Returns
    -------
    ranges: numpy array
        the ranges corresponding to each ADC sample
    Example
    -------

    """
    c = 3e8  # speed of light in m/s
    adc_samples_per_chirp = adc_values.shape[0]
    range_fft = fft(adc_values)

    peaks = find_peaks(np_abs(range_fft[:len(range_fft)//2]),
                       height=fft_threshold)[0]
    # d = i * fs*c/2/k/NA
    i2r = lambda idx: idx*adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    ranges = array([i2r(peak_idx) for peak_idx in peaks])
    return ranges


def dft_cfr_idx(fft_mag:NDArray,
                train_cell_count:int,
                pfa:float) -> NDArray:
    """ returns indexes where cfar finds peaks
    """

    # fft_values = np.fft.fft(adc_values)
    cfar_thresholds = cfar_ca(fft_mag,
                              guard_cell_count=1,
                              train_cell_count=train_cell_count,
                              pfa=1e-6)  # pfa)

    # cfar_th = cfar_ca_1d(range_mag, num_training_cells=10,
    #                num_guard_cells=1,
    #                           debug=False)
    peak_idxs = where(fft_mag > cfar_thresholds + 1e-10)[0]
    # peaks = range_mag > cfar_thresholds
    import matplotlib.pyplot as plt
    if pfa==1e-6:
        print("fft mag", fft_mag[:5])
        print("cfar", cfar_thresholds[:5])
        print("peak_idxs", peak_idxs)
        print("fft_mag0", fft_mag[0] > cfar_thresholds[0])
        print("fft_mag1", fft_mag[1] > cfar_thresholds[1])
        print("fft_mag2", fft_mag[2] > cfar_thresholds[2])

        plt.plot(fft_mag, 'r-')
        plt.plot(cfar_thresholds, 'b*')
        plt.title("doppler dimension")
        plt.show()
        
        
    return peak_idxs


def ranges_dft_cfar(adc_values:NDArray, chirp_slope:float,
                    adc_sample_rate:float,pfa:float) -> NDArray:
    """ returns a NDArray of ranges using a simple fft threshold for target """
    adc_samples_per_chirp = adc_values.shape[0]
    c = 3e8  # speed of light in m/s
    adc_samples_per_chirp = adc_values.shape[0]
    fft_mag = np_abs(fft(adc_values))
    peak_idxs = dft_cfr_idx(fft_mag,train_cell_count=20,
                            pfa=pfa)
    # d = i * fs*c/2/k/NA
    i2r = lambda idx: idx*adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    ranges = array([i2r(peak_idx) for peak_idx in peak_idxs])
    return ranges


def range_doppler(adc_values: NDArray,
                  adc_sample_rate:float,
                  chirp_slope: float,
                  wavelength:float,
                  chirp_period: float):
    """
    Parameters
    ----------
    adc_values
        (chirp_count, adc_samples count)
    """
    Z_fft2 = fft2(adc_values)
    print("adc_values.shape", adc_values.shape)
    print("adc_values.shape[1]", adc_values.shape[1] // 2)
    Z_fft2 = Z_fft2[:, :adc_values.shape[1] // 2]
    
    import matplotlib.pyplot as plt
    # plt.imshow(np_abs(Z_fft2))
    # plt.show()
    # exit()
    fft_1d_mag = np_abs(np_sum(Z_fft2, axis=0))

    range_idxes = dft_cfr_idx(fft_1d_mag[:-3],
                              train_cell_count=10,
                              pfa=0.001)
    print("range_idxes", range_idxes)
    range_idxes_grouped = peak_grouping_1d(range_idxes, fft_1d_mag)
    print("range_idxes_grouped", range_idxes_grouped)
    # FIXME: make this vectored
    range_dopplers_idxes = []
    # range_idxes_grouped = [range_idxes_grouped[0]]
    """import matplotlib.pyplot as plt
    plt.plot(np_abs(np_sum(Z_fft2, axis=1)))
    plt.title("Range------Doppler")
    plt.show()"""
    # range_idxes_grouped = [13]
    for range_idx in range_idxes_grouped:
        doppler_idxes = dft_cfr_idx(np_abs(Z_fft2[:, range_idx]), 
                                    train_cell_count=10,
                                    pfa=0.0001)
        doppler_idxes_grouped = peak_grouping_1d(doppler_idxes,
                                               np_abs(Z_fft2[:, range_idx]))
        range_dopplers_idxes += [(range_idx, doppler_idx) for doppler_idx in doppler_idxes_grouped]
        range_to_meters = lambda idx: float(idx*adc_sample_rate * 3e8 / \
            (2*chirp_slope*adc_values.shape[1]))
        doppler_to_mps = lambda idx: float(idx*wavelength/4/chirp_period)
        detections = [(range_to_meters(r), doppler_to_mps(d)) for r, d in range_dopplers_idxes]
    return detections


def range_aoa(adc_values: NDArray, radar: Radar):
    import numpy as np
    from scipy.fft import fft, fftshift

    range_axis = 1
    aoa_axis = 0

    range_window = np.kaiser(adc_values.shape[range_axis], beta=6)
    adc_windowed = adc_values * range_window[np.newaxis, :]

    # --- Range FFT ---
    range_fft = fft(adc_windowed, axis=range_axis)

    aoa_window = np.kaiser(adc_values.shape[aoa_axis], beta=10)
    range_fft_windowed = range_fft * aoa_window[:, np.newaxis]

    range_aoa = fftshift(fft(range_fft_windowed, axis=aoa_axis), axes=aoa_axis)
    range_aoa = fft(range_fft_windowed, axis=aoa_axis)
    import matplotlib.pyplot as plt
    plt.plot(np.abs(np.abs(range_aoa[0, :])))
    plt.plot(np.abs(np.abs(range_aoa[1, :])))
    plt.show()
    range_peak_idxs = dft_cfr_idx(np.abs(range_aoa[0, :]),
                            train_cell_count=20,
                            pfa=1e-6)
    print(range_peak_idxs)
    range_idxes_grouped = peak_grouping_1d(range_peak_idxs, np.abs(range_aoa[0, :]))
    i2r = lambda idx: idx*radar.adc_sample_rate * 3e8 / \
        (2*radar.chirp_slope*radar.adc_sample_count)
    ranges = array([i2r(peak_idx) for peak_idx in range_idxes_grouped])
    print(ranges)
    for idx in range_idxes_grouped:
        aoa_peak_idxs = dft_cfr_idx(np.abs(range_aoa[:, idx]),
                                    train_cell_count=32,
                                    pfa=1e-6)
        print("aoa_peak_idxs", aoa_peak_idxs)


def pcl(cube):
    """ returns array of 3D pcl

    Parameters
    ----------
    cube: numpy array
        cube is [chirp][elevation][azimuth][adc]

    Returns
    -------
    pcl: numpy array
        pcl is a 1D array of point
        each point is defined by (x,y,z,vr,mag)
    """
    pass
    points = []
    return points