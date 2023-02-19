from numpy import arctan2, arange, cos, exp, pi, sqrt, zeros
from numpy import float32  # alternatives: float16, float64
from numpy import complex_ as complex


def BB_IF(f0_min, slope, T, antenna_tx, antenna_rx, target,
          medium, datatype=float32, radar_equation=False):
    """ This function implements the mathematical IF defined in latex as
    y_{IF} = cos(2 \\pi [f_0\\delta + s * \\delta * t - s* \\delta^2])
    into following python code
    y_IF = cos (2*pi*(f_0 * delta + slope * delta * T + slope * delta**2))

    Parameters
    ----------
    f0_min: float
        the frequency at the beginning of the chirp
    slope: float
        the slope with which the chirp frequency increases over time
    T: ndarray
        the 1D vector containing time values'
    antenna_tx: Antenna
        x, y, z coordinates
    antenna_rx: Antenna
        x, y, z coordinates
    target: Target
        instance of Target()
    medium : Medium
        instance of Medium
    datatype: type
        either float16, 32, 64 or complex128
    radar_equation: bool
        if True adds Radar Equation contribution to IF values
    Returns
    -------
    YIF : ndarray
        vector containing the IF values
    """
    tx_x, tx_y, tx_z = antenna_tx.xyz
    rx_x, rx_y, rx_z = antenna_rx.xyz
    t_x, t_y, t_z = target.pos(T[0])
    v = medium.v
    L = medium.L
    distance = sqrt((tx_x - t_x)**2 + (tx_y - t_y)**2 + (tx_z - t_z)**2)
    distance += sqrt((rx_x - t_x)**2 + (rx_y - t_y)**2 + (rx_z - t_z)**2)
    azimuth_rx = arctan2(rx_x-t_x, rx_y-t_y)
    azimuth_tx = arctan2(tx_x-t_x, tx_y-t_y)
    elevation_rx = arctan2(rx_y-t_y, rx_z-t_z)
    elevation_tx = arctan2(tx_y-t_y, tx_z-t_z)
    delta = distance / v
    fif_max = 2*slope*distance/v
    if datatype == complex:
        YIF = exp(2 * pi * 1j *
                  (f0_min * delta + slope * delta * T + slope * delta**2))
    else:
        YIF = cos(2 * pi *
                  (f0_min * delta + slope * delta * T + slope * delta**2))
    # here bring in the radar equation
    # target_type and RCS
    # most targets will have 1/R*4, corner reflector as 1/R**2
    # and antenna radiation patterns in azimuth, elevation
    # and frequency response
    # f0 being the center frequency of the chirp
    # f0 = f0_min + slope*(T[-1]-T[0])/2
    # Ptc = conducted Power in W
    # Ptr = Ptc * Gt(azimuth, elevation, f0)
    # Ptarget = Ptr * 1/(4*pi*distance**2) * RCS
    # if target is `corner reflector
    # Prx = Ptarget * L
    # else
    # Prx = Ptarget * 1/(4*pi*distance**2) * L
    # Where L = Medium Losses during propagation *
    #       fluctuation Losses (often modeled w/ Swerling models)
    # Prx_e = Prx * AW (where AW is effective area RX antenna)
    # Prx_c = Prx_c * Gr(azimuth, elevation, f0)
    f0 = f0_min + slope*(T[-1]-T[0])/2
    YIF = YIF * antenna_tx.gain(azimuth_tx, elevation_tx, f0) \
        * antenna_rx.gain(azimuth_rx, elevation_rx, f0)
    if radar_equation:
        YIF = YIF * target.rcs(f0)
        if target.target_type == "corner_reflector":
            YIF = YIF / distance**2
        else:
            YIF = YIF / distance**4
        YIF = YIF * 10**(L*distance)
    IF = (YIF, fif_max)
    return IF


def rt_points(radar, targets, radar_equation=False,
              datatype=float32, debug=False):
    """ raytracing with points

    Parameters
    ----------
    radar: Radar
        instance of Radar
    targets: List[Target]
        list of targets in the Scene
    radar_equation: bool
        if True includes the radar equation when computing the IF signal
        else ignores radar equation
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
    # n_chirps is the # chirps each TX antenna sends per frame
    n_chirps = radar.chirps_count
    n_tx = len(radar.tx_antennas)
    n_rx = len(radar.rx_antennas)
    n_adc = radar.n_adc
    ts = 1/radar.fs
    adc_cube = zeros((n_frames, n_chirps, n_tx, n_rx, n_adc)).astype(datatype)
    f0_min = radar.f0_min
    slope = radar.slope
    T = arange(0, n_adc*ts, ts)

    for chirp_i in range(n_chirps):
        for tx_i in range(n_tx):
            T[:] += radar.t_interchirp
            for rx_i in range(n_rx):
                YIF = zeros(n_adc).astype(datatype)
                for target in targets:
                    YIFi, fif_max = BB_IF(f0_min, slope, T,
                                          radar.tx_antennas[tx_i],
                                          radar.rx_antennas[rx_i],
                                          target,
                                          radar.medium,
                                          radar_equation=radar_equation,
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
                adc_cube[0, chirp_i, tx_i, rx_i, :] = YIF

    baseband = {"adc_cube": adc_cube, "frames_count": n_frames,
                "chirps_count": radar.chirps_count,
                "t_interchirp": radar.t_interchirp, "n_tx": n_tx,
                "n_rx": n_rx, "datatype": datatype,
                "f0_min": f0_min, "slope": slope, "T": T,
                "fs": radar.fs, "v": radar.v}
    return baseband
