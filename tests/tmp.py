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
        (timestamp_count, TX, S, RX) the timestamps at which TX frequency
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
    chirp_count = 3  # number chirps per antenna
    antenna_count = 2  # number antennas
    chirp_indexes = np.arange(chirp_count)
    antenna_indexes = np.arange(antenna_count)
    assert antenna_count == timestamps.shape[1]

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
        print("chirp index", chirp_index)
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

    # [1, antenna_count, 1, 1, chirp_count]
    chirp_start = chirp_index[None, :, None, None, :] * t_inter_chirp
    print("chirp_start", chirp_start)
    chirp_end = chirp_start + chirp_end_time

    timestamps = timestamps[...,None]  #[:, None, None, None]  # [timestamp_count, 1, 1] T/TX/S/RX
    # Broadcasting: NumPy aligns axes from the right and expands size-1 axes too match.
    #
    # timestamps       : (timestamp_count, TX, S, RX,     1          )  < 1s expand rightward
    # chirp_start : (1, antenna_count, 1, 1, chirp_count        )  < 1 expands leftward
    # result      : (timestamp_count, TX, S, RX,     chirp_count)
    #
    # Each timestamp is compared against every (antenna, chirp) pair — no loops needed.
    # [timestamp_count, antenna_count, chirp_count]
    print(69, timestamps.shape, chirp_start.shape)
    active = (timestamps >= chirp_start) & (timestamps <= chirp_end)
    print("start", chirp_start[0,0,0,0,0])
    print("active ts0", active[0,0,0,0,0])
    print("active ts1", active[1,0,0,0,0])


    # now compute the frequency as a function of chirp_index
    # [timestamp_count, antenna_count, chirp_count]
    freq = chirp_start_freq + chirp_slope * (timestamps - chirp_start)

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
    tx_frequencies = (active * freq).sum(axis=4)
    return tx_frequencies


if __name__ == "__main__":
    times = np.array([[[[-0.5, 0.25, 1.2, 1.6, 2],
                      [-0.5, 0.25, 1.2, 1.6, 2]],
                     [[-0.5, 0.25, 1.2, 1.6, 2],
                      [-0.5, 0.25, 1.2, 1.6, 2]]]]).transpose(3, 2, 0, 1)
    times = np.array([[[[-0.5,  -0.5 ]], [[-0.5 , -0.5 ]]],
                      [[[ 0.25,  0.25]], [[ 0.25  ,0.25]]],
                      [[[ 1.2 ,  1.2 ]], [[ 1.2,   1.2 ]]],
                      [[[ 1.6 ,  1.6 ]], [[ 1.6,   1.6 ]]],
                      [[[ 2.,    2.  ]], [[ 2.,    2.  ]]]])

    print("times.shape", times.shape)
    print(101, times[1,0,:,:])
    # times(timestamps, tx, scatter, rx)
    print(times[0,0,0,0])
    print(times[3,0,0,0])
    res_tdm = TX_Freq(times, multiplexing="TDM")
    print(res_tdm)  # should be all 60s at time 0
    print("res_tdm", res_tdm.shape)
    print("--------")
    print(res_tdm[1,0,:,:]) # 60.25 on TX0

    print(res_tdm[2,1,:,:]) # 60.2 on TX1
    # if will be sum over axis 1, 2, 3 ?
    print(res_tdm[:,:,:,0]) # axis=0 is time, then TX timeshifted -> signal - RX, then SC-> drop, then RX -> which ADC channels sees this
    # emulate 2 timestamps
    # 2 TX
    # on first timestamp, 0 s when TX0 transmits 
    # second timestampe, t=1.2 one when TX1 transmits
    # 1 scatterer, 1 RX
    # results is a (2,2,1,1) shape
    times = np.concatenate([np.zeros((1, 2, 1, 1)),
                            np.ones((1,2,1,1))*1.2], axis=0)
    print("123, times.shape", times.shape)
    print(times)
    res_tdm = TX_Freq(times, multiplexing="TDM")
    print("res_tdm.shape", res_tdm.shape)  # should be all 60s at time 0
    print("antenna 0, then 1")
    print(res_tdm[:,0,:, :])
    print("-"*10)
    print(res_tdm[:,1,:,:])
    # emulate 3 timestamps
    # 2 TX
    # on first timestamp, 0 s when TX0 transmits 
    # second timestampe, t=1.2 one when TX1 transmits
    # third timestmp t=1.6 when no one trasmits
    # 1 scatterer, 1 RX
    # results is a (2,2,1,1) shape
    print("# 3. 3,2,1,1")
    print("------------")
    times = np.concatenate([np.zeros((1, 2, 1, 1)),
                            np.ones((1,2,1,1))*1.2,
                            np.ones((1,2,1,1))*1.6], axis=0)
    res_tdm = TX_Freq(times, multiplexing="TDM")
    print("res_tdm.shape", res_tdm.shape)
    print(res_tdm)
    print("# 4. 2,2,2,1")
    print("------------")  # should be all 60s at time 0
    ts0_tx0_scatterer_0 = np.zeros((1, 1, 1, 1))
    ts0_tx0_scatterer_1 = np.zeros((1, 1, 1, 1)) - 0.1
    ts0_tx0_scatterers = np.concatenate([ts0_tx0_scatterer_0,
                                         ts0_tx0_scatterer_1], axis=2)
    ts0_tx1_scatteers = np.zeros((1, 1, 2, 1))
    ts0 = np.concatenate([ts0_tx0_scatterers, ts0_tx1_scatteers], axis=1)
    times = np.concatenate([ts0,
                            np.ones((1,2,2,1))*1.2], axis=0)
    res_tdm = TX_Freq(times, multiplexing="TDM")
    print("res_tdm.shape", res_tdm.shape)
    print("res_tdm", res_tdm)
    """
    [[[[60. ] - ts0, tx0, s0 - 60
   [ 0. ]]    - ts0, tx0, s1 - 0

  [[ 0. ]     - ts0, tx1, x - 0
   [ 0. ]]]


 [[[ 0. ]      - ts1 , tx0, x - 0
   [ 0. ]]

  [[60.2]       - ts1, tx1, x - 60.2
   [60.2]]]]
    """