""" This is mostly to test the RX_freq and IF signal
"""

from os.path import abspath, join, pardir
import sys
from test_assets import radar_tdm_1_chirp_8_adc, target_static_10p1m, target_static_5p1m, fif00, fif01, chirp_slope_tdm0  # noqa E402
import numpy as np
import pytest
from numpy import allclose, array, arange

from os.path import abspath, join, pardir
import pytest
import sys
dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

# from mmWrt.Raytracing import BB_IF
from mmWrt.Scene import scene_distance


def test_RX_TX_DC_mix():
    """ test that ADC are all 0 when f_rx == f_tx
    """
    # boiler plate
    radar = radar_tdm_1_chirp_8_adc
    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    number_adc_samples = radar.receiver.number_adc_samples
    f_rx = array([radar.transmitter.slope*i for i in range(number_adc_samples)])
    ph_rx = np.zeros(number_adc_samples)

    # inject f_rx which is the same as f_tx
    f_if = radar.BB_IF(adc_times=adc_times,
                       f_rx=f_rx)
    yif = radar.adc_sampling(f_if,
                             ph_rx=ph_rx,
                             adc_times=adc_times)

    assert allclose(yif, np.zeros(number_adc_samples), atol=1e-2)

def test_RX_freq_2():
    """ test that ADC values match expected value from the receiving
    radar having transmitted with TX which off by offset equivalent to 
    TOF for 5.1 given slope"""
    # boiler plate
    radar = radar_tdm_1_chirp_8_adc

    adc_times = np.array([0.00000000e+00, 9.90099010e-07, 1.98019802e-06, 2.97029703e-06,
                          3.96039604e-06, 4.95049505e-06, 5.94059406e-06, 6.93069307e-06])
    # 60e9+chirp_slope_tdm0*adc_times 
    f_rx = np.array([[[6.00000000e+10, 6.00049505e+10, 6.00099010e+10, 6.00148515e+10,
                     6.00198020e+10, 6.00247525e+10, 6.00297030e+10, 6.00346535e+10]]])
    # add the if for target at 5.1 m
    f_rx += fif00

    f_if = radar_tdm_1_chirp_8_adc.BB_IF(adc_times=adc_times,
                                         f_rx=f_rx)
    yif = radar.adc_sampling(f_if,
                             ph_rx=np.zeros(8),
                             adc_times=adc_times)

    # yif_expected = np.exp(2*1j*np.pi*f_if*adc_times
    yif_expected = np.array([ 1.+0.j, 0.49096725+0.87117803j, -0.517955  +0.85540786j,
                            -0.99950762-0.0313769j,  -0.46322689-0.88623972j,  0.54484995-0.83853356j,
                            0.99799606+0.06327614j,  0.43452578+0.9006594j ])
    assert allclose(yif, yif_expected, atol=1e-8)

@pytest.mark.parametrize("target, radar, frequency_if", [
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, fif00),
    (target_static_10p1m, radar_tdm_1_chirp_8_adc, fif01),
])
def tbd_if_freq0(target, radar, frequency_if):
    """ check that the frequency of the
    IF signal (after mixing) is the good one
    """

    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)

    time_of_flight = 2*target.distance()/radar.medium.v
    f_rx = radar.TX_freqs(adc_times-time_of_flight)
    f_tx = radar.TX_freqs(adc_times)
    f_mix = f_tx - f_rx
    f_if = f_mix.copy()
    f_if[f_if > 1e8] = 0

    f_tone = np.array([frequency_if]*radar.number_adc_samples)
    f_tone[0] = 0
    assert np.allclose(f_if, f_tone)

def tbdest_1():
    targets = [target_static_5p1m]
    radars = [radar_tdm_1_chirp_8_adc]
    receiver_radar = radar_tdm_1_chirp_8_adc
    adc_sample_rate = receiver_radar.receiver.adc_sample_rate
    adc_times = arange(0, receiver_radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    number_adc_samples = adc_times.shape[0]

    # adc_samples = zeros((n_rx, number_adc_samples)).astype(datatype)
    # rx_antennas_pos = zeros((n_rx, number_adc_samples))
    # tx_antennas_count = sum([len(radar.tx_antennas) for radar in radars])
    # tx_antennas_pos = zeros((n_rx, number_adc_samples))
    # targets_pos = zeros((len(targets), 3, numer_adc_samples))


    rx_antennas_pos = array([rx_antenna.xyz for rx_antenna in receiver_radar.rx_antennas])
    # rx_antennas_pos = tile(rx_antennas_pos, (len(targets), 1, number_adc_samples))

    tx_antennas_pos = array([tx_antenna.xyz for tx_antenna in receiver_radar.tx_antennas])

    targets_positions = np.empty((len(targets), adc_times.shape[0], 3))
    for i, target in enumerate(targets):
        targets_positions[i,:,:] = target.pos_t1(adc_times)
    # diff = targets_positions - tx_antennas_pos # 2000 targets * 1024 samples  operations
    # distance_tx_target = sqrt(sum(diff * diff, axis=-1))
    distance_tx_target = scene_distance(targets_positions, tx_antennas_pos)
    # Compute the distance from target to rx for each time point
    # distance_target_rx = euclidian_distance(targets_positions - rx_antennas_pos, axis=1)
    # diff = targets_positions - rx_antennas_pos
    # distance_target_rx = sqrt(sum(diff * diff, axis=-1))
    distance_target_rx = scene_distance(targets_positions, rx_antennas_pos)

    # Total distance is the sum of both distances for each time point
    total_distance = distance_tx_target + distance_target_rx
    time_of_flight = total_distance/receiver_radar.v

    # for radar in radars:
    f_rx = array([radar.TX_freqs(adc_times-time_of_flight) for radar in radars])
    print("f_rx ---------", f_rx)
    assert False
    ph_rx = array([radar.TX_phases(adc_times-time_of_flight) for radar in radars])
    f_if = receiver_radar.BB_IF(adc_times, f_rx, ph_rx)
    YIF = receiver_radar.adc_sampling(f_if, total_distance, ph_rx, adc_times)