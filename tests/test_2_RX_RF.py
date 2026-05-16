""" Place holder for testing the RX side of the system.
at the moment it is mostly done in the Raytracing module
will need to be moved into the scene modeling, place holder for later
"""

from os.path import abspath, join, pardir
import sys
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks


dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)


def test_no_mixing():
    """ to simulate no mixing, we define a radar without transmission
    receiving a frequency and without filters, we get the same frequency
    at the output of the mixer as at the input
    """
    from tests.test_assets import radar_tx_off, radar_tdm_1_chirp_8_adc
    from numpy import abs as np_abs

    adc_times = np.array([1]*7)
    f_rx = np.arange(57e9, 64e9, 1e9)

    if_frequencies = radar_tx_off.BB_IF(adc_times,
                                        f_rx=f_rx)

    assert np.allclose(if_frequencies, -f_rx, atol=1e-3)


def test_interferer_no_mixing():
    """ to simulate no mixing, we define a radar without transmission
    receiving a frequency and change LPF filter, to get IF = RF RX
    """
    from tests.test_assets import radar_tx_off, radar_tdm_1_chirp_8_adc
    from numpy import abs as np_abs

    radar = radar_tdm_1_chirp_8_adc
    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = np.arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)

    f_rx = np.array([radar.TX_freqs(adc_times) for radar in [radar_tdm_1_chirp_8_adc]])

    if_frequencies = radar_tx_off.BB_IF(adc_times,
                                        f_rx=f_rx)
    assert np.allclose(if_frequencies, -f_rx, atol=1e-3)


def test_interferer_mixing():
    """ to simulate mixing, we define a radar with CW transmission
    receiving a frequency and change LPF filter,
    to get IF = RF RX - CW
    """
    from tests.test_assets import radar_tx_cw
    from numpy import abs as np_abs

    adc_times = np.array([1]*7)
    f_rx = np.arange(57e9, 64e9, 1e9)

    if_frequencies = radar_tx_cw.BB_IF(adc_times,
                                        f_rx=f_rx)
    assert np.allclose(if_frequencies, -(f_rx-60e9), atol=1e-3)