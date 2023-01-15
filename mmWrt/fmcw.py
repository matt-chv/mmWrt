from numpy import cos, sqrt, pi


def BB_IF(f0_min, slope, T, antenna_tx, antenna_rx, target, v=3e8):
    """ This function implements the mathematical IF defined in latex as
    y_{IF} = cos(2 \\pi [f_0\\delta + s * \\delta * t - s* \\delta^2])
    into following python code
    y_IF = cos (2*pi*(f_0 * delta + slope * delta * T + slope * delta**2))

    Parameters
    ----------
    f0_min : float
        the frequency at the begining of the chirp
    slope : float
        the slope with which the chirp frequency inceases over time
    T : ndarray
        the 1D vector containing time values
    antenna_tx : tuple of floats
        x, y, z coordinates
    antenna_rx : tuple of floats
        x, y, z coordinates
    target : tuple of floats
        x, y, z coordinates
    v : float
        speed of light in considered medium

    Returns
    -------
    YIF : ndarray
        vector containing the IF values
    """
    tx_x, tx_y, tx_z = antenna_tx
    rx_x, rx_y, rx_z = antenna_rx
    t_x, t_y, t_z = target
    distance = sqrt((tx_x - t_x)**2 + (tx_y - t_y)**2 + (tx_z - t_z)**2)
    distance += sqrt((rx_x - t_x)**2 + (rx_y - t_y)**2 + (rx_z - t_z)**2)
    delta = distance / v
    YIF = cos(2 * pi * (f0_min * delta + slope * delta * T + slope * delta**2))
    return YIF
