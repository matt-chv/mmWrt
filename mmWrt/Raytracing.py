from numpy import arange, cos, exp, pi, sqrt, zeros
from numpy import float32  # alternatives: float16, float64
from numpy import complex_ as complex


def BB_IF(f0_min, slope, T, antenna_tx, antenna_rx, target,
          v=3e8, datatype=float32):
    """ This function implements the mathematical IF defined in latex as
    y_{IF} = cos(2 \\pi [f_0\\delta + s * \\delta * t - s* \\delta^2])
    into following python code
    y_IF = cos (2*pi*(f_0 * delta + slope * delta * T + slope * delta**2))

    Parameters
    ----------
    f0_min: float
        the frequency at the begining of the chirp
    slope: float
        the slope with which the chirp frequency inceases over time
    T: ndarray
        the 1D vector containing time values
    antenna_tx: tuple of floats
        x, y, z coordinates
    antenna_rx: tuple of floats
        x, y, z coordinates
    target: Target
        instance of Target()
    v : float
        speed of light in considered medium
    datatype: type
        either float16, 32, 64 or complex128

    Returns
    -------
    YIF : ndarray
        vector containing the IF values
    """
    tx_x, tx_y, tx_z = antenna_tx
    rx_x, rx_y, rx_z = antenna_rx
    t_x, t_y, t_z = target.pos(T[0])
    distance = sqrt((tx_x - t_x)**2 + (tx_y - t_y)**2 + (tx_z - t_z)**2)
    distance += sqrt((rx_x - t_x)**2 + (rx_y - t_y)**2 + (rx_z - t_z)**2)
    delta = distance / v
    fif_max = 2*slope*distance/v
    if datatype == complex:
        YIF = exp(2 * pi * 1j *
                  (f0_min * delta + slope * delta * T + slope * delta**2))
    else:
        YIF = cos(2 * pi *
                  (f0_min * delta + slope * delta * T + slope * delta**2))
    IF = (YIF, fif_max)
    return IF


def rt_points(radar, targets, datatype=float32, debug=False):
    """ raytracing with points

    Parameters
    ----------
    radar: Radar
        instance of Radar
    targets: List[Target]
        list of targets in the Scene
    datatype: Type
        type of data to be generate by rt: float16, float32, ... or complex
    debug: bool
        if True increases level of print messages seen

    Returns
    -------
    baseband: dict
        dictonnary with adc values and other parameters used later in analysis

    Raises
    ------
    ValueError
        if Nyquist rule is not upheld
    """
    n_frames = radar.frames_count
    n_tx = len(radar.tx_antennas)
    n_rx = len(radar.rx_antennas)
    n_adc = radar.n_adc
    ts = 1/radar.fs
    adc_cube = zeros((n_frames, n_tx, n_rx, n_adc)).astype(datatype)
    f0_min = radar.f0_min
    slope = radar.slope
    T = arange(0, n_adc*ts, ts, dtype=datatype)

    for frame_i in range(n_frames):
        T[:] += radar.t_interchirp
        for tx_i in range(n_tx):
            for rx_i in range(n_rx):
                YIF = zeros(n_adc).astype(datatype)
                for target in targets:
                    YIFi, fif_max = BB_IF(f0_min, slope, T,
                                          radar.tx_antennas[tx_i],
                                          radar.rx_antennas[rx_i],
                                          target,
                                          datatype=datatype)
                    # ensure Nyquist is respected
                    try:
                        assert fif_max * 2 <= radar.fs
                    except AssertionError:
                        if debug:
                            print(f"failed Nyquist for target: {tx_i}" +
                                  f"fif_max is: {fif_max} " +
                                  f"radar ADC fs is: {radar.fs}")
                        raise ValueError("Nyquist will always prevail")
                    YIF += YIFi
                adc_cube[frame_i, tx_i, rx_i, :] = YIF

    baseband = {"adc_cube": adc_cube, "frames_count": n_frames,
                "t_interchirp": radar.t_interchirp, "n_tx": n_tx,
                "n_rx": n_rx, "datatype": datatype,
                "f0_min": f0_min, "slope": slope, "T": T,
                "fs": radar.fs, "v": radar.v}
    return baseband
