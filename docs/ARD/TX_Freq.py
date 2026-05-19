""" Standalone implementation of the TX_Freq method of the Transmitter class
"""
import numpy as np
from numpy.typing import NDArray
from typing import Literal


def TX_Freq(times: NDArray, multiplexing: Literal["TDM", "DDM"] = "TDM") -> NDArray:
    """ Returns the frequencies for each TX antenna at given timestamps
    Parameters
    ----------
    times:
        [timestamp_count] the timestamps at which TX frequency
        for all TX channels should be reported
    multiplexing:
        a switch for how the TX frequency are computed

    Returns
    -------
    tx_frequencies:
        [timestamp_count, tx_antenna_count] the tx frequency for each
        tx antenna at each timestamp. when a timestamp is out of a chirp
        for the given antenna the value returned is 0
    """
    chirp_start_freq = 60
    chirp_slope = 1
    chirp_end_time = 0.5
    t_inter_chirp = 1
    chirp_count = 10  # number chirps per antenna
    antenna_count = 3  # number antennas
    chirp_indexes = np.arange(chirp_count)
    antenna_indexes = np.arange(antenna_count)

    # Absolute chirp index for transmitter k on its nth chirp
    # is a function of the multiplexing
    if multiplexing == "TDM":
        """
        [[ 0  3  6  9 12 15 18 21 24 27]
         [ 1  4  7 10 13 16 19 22 25 28]
         [ 2  5  8 11 14 17 20 23 26 29]]
        """
        # [antenna_count, chirp_count]
        chirp_index = antenna_indexes[:, None] + chirp_indexes[None, :] * antenna_count

    elif multiplexing == "DDM":
        """ DDM allows 10 chirps per antenna to be sent over only 10 chirps 
        in total being in effect antenna_count faster than TDM
        (at some compromises)
        [[0 1 2 3 4 5 6 7 8 9]
         [0 1 2 3 4 5 6 7 8 9]
         [0 1 2 3 4 5 6 7 8 9]]
        """
        # [antenna_count, chirp_count]
        chirp_index = antenna_indexes[:, None]*0 + chirp_indexes[None, :]

    # [1, antenna_count, chirp_count]
    chirp_start = chirp_index[None, :, :] * t_inter_chirp
    chirp_end = chirp_start + chirp_end_time

    times = times[:, None, None]  # [timestamp_count, 1, 1]
    # Broadcasting: NumPy aligns axes from the right and expands size-1 axes to match.
    #
    # times       : (timestamp_count, 1,             1          )  < 1s expand rightward
    # chirp_start : (1,               antenna_count, chirp_count)  < 1 expands leftward
    # result      : (timestamp_count, antenna_count, chirp_count)
    #
    # Each timestamp is compared against every (antenna, chirp) pair — no loops needed.
    # [timestamp_count, antenna_count, chirp_count]
    active = (times >= chirp_start) & (times <= chirp_end)

    # now compute the frequency as a function of chirp_index
    # [timestamp_count, antenna_count, chirp_count]
    freq = chirp_start_freq + chirp_slope * (times - chirp_start)

    """NOTE: Boolean-float multiplication as a zero-mask
    # ──────────────────────────────────────────────────
    # NumPy handles booleans as integers (True=1, False=0).
    # Multiplying a boolean array by a float array promotes bool→float,
    # making this equivalent to np.where(active, freq, 0.0) but without
    # an extra temporary array allocation.
    #
    #   active * freq  →  1.0 * freq  (active chirp   — keeps frequency)
    #                     0.0 * freq  (inactive chirp  — zeroes out)
    #
    # Safe to sum over the chirp axis because TDM guarantees at most one
    # True entry per timestamp for all chirps per antenna."""
    # [timestamp_count, antenna_count]
    tx_frequencies = (active * freq).sum(axis=2)
    return tx_frequencies


if __name__ == "__main__":
    times = np.array([-0.5, 0.25, 1.2, 1.6])
    res_tdm = TX_F_TDM(times, multiplexing="TDM")
    expected_tdm = np.array([
        [0.0,          0.0,          0.0],   # t=-0.5  no antenna active
        [60.0 + 0.25,  0.0,          0.0],   # t=0.25  TX0 chirp0 active
        [0.0,          60.0 + 0.2,   0.0],   # t=1.2   TX1 chirp0 active
        [0.0,          0.0,          0.0],   # t=1.6   None active
    ])

    np.testing.assert_allclose(res_tdm, expected_tdm)
    res_ddm = TX_F_TDM(times, multiplexing="DDM")
    expected_ddm = np.array([
        [ 0.  ,  0.  ,  0.  ],   # t=-0.5  no antenna active
        [60.25, 60.25, 60.25],   # t=0.25  TX0 chirp0 active
        [60.2 , 60.2 , 60.2 ],   # t=1.2   TX1 chirp0 active
        [ 0.  ,  0.  ,  0.  ],   # t=1.6   None active
    ])
    np.testing.assert_allclose(res_ddm, expected_ddm)