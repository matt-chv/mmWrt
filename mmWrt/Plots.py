from numpy import arange
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt


def plot_range_doppler(cube, radar, _d0=None, _v0=None, no_speed_shift=True):
    """ Plots Range Doppler with axis labeled in SI

    Parameters
    -----------
    cube: numpy array
        contains a 2D array with slow time and fast time samples
    radar: Radar instance
        contains configuration values for displaying the range doppler
    _d0: optional
        if is not None, will be used in title for target pos
    _v0: optional
        if is not None, will be used in title as target speed
    no_speed_shift: bool
        if True does not do fftshift on the speeds

    Returns
    -------
    plot_details: tuple
        (fig, ranges, speeds)
    """
    # FIXME: here call rsp.range_fft
    """
    c = 3e8
    _fs = cfg["fs"]
    _k = cfg["k"]
    _NA = cfg["NA"]
    _NC = cfg["NC"]
    _TIC = cfg["TIC"]
    _f0_min = cfg["f0_min"]
    """
    c = radar.v
    _fs = radar.fs
    _k = radar.slope
    _NA = radar.n_adc
    _NC = radar.chirps_count
    _TIC = radar.t_inter_chirp
    _f0_min = radar.f0_min
    _L0M = c/_f0_min

    if _d0 is None or _v0 is None:
        title = "Range Doppler"
    else:
        title = f"Range-Doppler  (d0, v0)={_d0}, {_v0:.2g}"

    if no_speed_shift:
        ranges = arange(0, _fs*c/4/_k, _fs*c/4/_k/_NA)
        speeds = arange(0, _L0M/4/_TIC,
                        _L0M/4/_TIC/_NC)
        rdop = abs(fft2(cube))
    else:
        # if ranges could be negative
        # ranges = arange(-_fs*c/8/_k, _fs*c/8/_k, _fs*c/4/_k/_NA)
        # but physics must prevail ...
        ranges = arange(0, _fs*c/4/_k, _fs*c/4/_k/_NA)
        speeds = arange(-_L0M/4/_TIC, _L0M/4/_TIC,
                        _L0M/2/_TIC/_NC)
        # frequency shift the doppler axes as distances cannot be negative
        rdop = abs(fftshift(fft2(cube), axes=0))

    fig = plt.figure(figsize=(10, 6))
    no_labels = 10  # how many labels to see on axis x
    step_x = int(_NA / (no_labels - 1))  # step between consecutive labels
    x_positions = arange(0, _NA, step_x)  # pixel count at label position
    x_labels = ranges[::step_x]  # labels you want to see
    x_labels = [f"{d:.2g}" for d in x_labels]
    plt.xticks(x_positions, x_labels)

    no_labels_y = 10  # how many labels to see on axis x
    step_y = int(_NC / (no_labels_y - 1))  # step between consecutive labels
    y_positions = arange(0, _NC, step_y)  # pixel count at label position
    y_labels = speeds[::step_y]  # labels you want to see
    y_labels = [f"{v:.2g}" for v in y_labels]  # rounding up for easier dispaly
    plt.yticks(y_positions, y_labels)

    plt.xlabel("Range (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title(title)
    plt.imshow(rdop, aspect='auto')
    plot_details = (fig, ranges, speeds)
    return plot_details
