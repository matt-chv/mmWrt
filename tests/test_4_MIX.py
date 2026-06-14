""" testing the mixer logic
v 0.0.11 - 5 passed
"""
import logging
from os.path import abspath, join, pardir
import sys
from test_assets import radar_tx_off, radar_tdm_1_chirp_8_adc, d_5p1m, \
    scatterer_static_10p1m, scatterer_static_5p1m, fif00, fif01, \
    chirp_slope_tdm0, adc_8_values_complex_fif00, \
    adc_sampling_times_8_samples, tof_5p1m, f0_60G
import numpy as np
import pytest
from numpy import allclose, array, arange

from os.path import abspath, join, pardir
import pytest
import sys
dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

# from mmWrt.Raytracing import BB_IF
# from mmWrt.Scene import scene_distance


def test_RX_eq_TX():
    # test that F_IF all 0 when f_rx == f_tx
    # 1 TX same as RX and 1 scatterer
    radar = radar_tdm_1_chirp_8_adc
    chirp_start_freq = radar.transmitter.chirp_start_freq
    adc_sample_rate = radar.receiver.adc_sample_rate
    timestamps = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    number_adc_samples = radar.receiver.number_adc_samples
    # f_rx = np.empty((number_adc_samples, 1))
    f_rx = array([chirp_start_freq + radar.transmitter.slope*t for t in timestamps])
    f_rx = f_rx[:, None, None, None]

    # inject f_rx which is the same as f_tx
    # f_if will be zeros after mixer
    f_if = radar.mixer(timestamps=timestamps,
                       f_rx=f_rx)
    f_if_expected = np.zeros(number_adc_samples)
    # since we are modelling 1TX, 1 Scatterer, 1 RX
    f_if_expected = f_if_expected[:, None, None, None]

    assert allclose(f_if, f_if_expected, atol=1e-2)
    assert f_if.shape == f_if_expected.shape


def test_RX_freq_2():
    # test that adding an offset yield the correct if

    radar = radar_tdm_1_chirp_8_adc
    chirp_start_freq = radar.transmitter.chirp_start_freq
    adc_sample_rate = radar.receiver.adc_sample_rate
    timestamps = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Radar").setLevel(logging.DEBUG)

    f_rx = array([chirp_start_freq + radar.transmitter.slope*t for t in timestamps])
    f_rx = f_rx[:, None, None, None]

    # remove the if for scatterer at 5.1 m
    # since the tx will be if ahead than rx
    f_rx -= fif00

    f_if = radar.mixer(timestamps=timestamps,
                                         f_rx=f_rx)

    f_if_expected = np.ones((8,1,1,1))*fif00

    assert f_if.shape == f_if_expected.shape
    assert allclose(f_if, f_if_expected, atol=1e-8)


def test_interferer_mixing():
    # to simulate mixing, we define a radar with CW transmission
    # receiving a frequency and change LPF filter,
    # to get IF = RF RX - CW
    from tests.test_assets import radar_tx_cw

    adc_times = np.array([1]*7)  # [:, None, None, None]
    f_rx = np.arange(57e9, 64e9, 1e9)
    f_rx = f_rx[:, None, None, None]

    if_frequencies = radar_tx_cw.mixer(adc_times,
                                       f_rx=f_rx)
    assert np.allclose(if_frequencies, -(f_rx-60e9), atol=1e-3)


def test_no_mixing():
    # to simulate no mixing, we define a radar without transmission
    # receiving a frequency and without filters, we get the same frequency
    # at the output of the mixer as at the input
    from tests.test_assets import radar_tx_off, radar_tdm_1_chirp_8_adc
    from numpy import abs as np_abs

    adc_times = np.array([1]*7)
    f_rx = np.arange(57e9, 64e9, 1e9)[:,None, None, None]

    if_frequencies = radar_tx_off.mixer(adc_times,
                                        f_rx=f_rx)
    assert np.allclose(if_frequencies, -f_rx, atol=1e-3)


def test_interferer_no_mixing():
    # to simulate no mixing, we define a radar without transmission
    # receiving a frequency and change LPF filter, to get IF = RF RX

    radar = radar_tdm_1_chirp_8_adc
    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = np.arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)

    # broadcast adc_times to match (T,TX,S,RX)
    f_rx = radar_tdm_1_chirp_8_adc.TX_freq(adc_times[:, None, None, None])

    if_frequencies = radar_tx_off.mixer(adc_times,
                                        f_rx=f_rx)
    assert np.allclose(if_frequencies, -f_rx, atol=1e-3)
