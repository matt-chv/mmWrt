""" This is mostly to test the ADC part of the Raytracing
Covers:
 - TDM mode
(not written DDM, SFMCW)
Does not cover:
- non point targets
- attenuation
"""

from os.path import abspath, join, pardir
import sys
from numpy import arange, where, float32
from numpy.fft import fft
from scipy.signal import find_peaks
from time import perf_counter

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import adc_samples
from mmWrt.Raytracing_old import rt_points
from mmWrt.RadarSignalProcessing import ranges_from_fft_threshold
from mmWrt.Scene import Radar, Transmitter, Receiver, Target

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"

def test_if_error_radar_tdm_1_chirp_8_adc_target_static_5p1m():
    """
    Test that in TDM mode the range estimation is within one range bin width
    Given a 8 ADC samples per chirp
    """

    from test_assets import radar_tdm_1_chirp_8_adc, target_static_5p1m, d_5p1m
    c = 3e8

    adc_sample_rate = radar_tdm_1_chirp_8_adc.receiver.adc_sample_rate
    chirp_slope = radar_tdm_1_chirp_8_adc.transmitter.chirp_slope
    adc_samples_per_chirp = radar_tdm_1_chirp_8_adc.receiver.adc_samples_per_chirp
    adc_times = arange(0, radar_tdm_1_chirp_8_adc.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar_tdm_1_chirp_8_adc,
                             [target_static_5p1m],
                             radars=[radar_tdm_1_chirp_8_adc])
    ranges = ranges_from_fft_threshold(adc_values[0, :],
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=2)
    
    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    try:
        assert len(ranges) == 1
        assert abs(ranges[0]-d_5p1m) < range_bin_width
    except:
        raise
    else:
        print("test_if_error_radar_tdm_1_chirp_8_adc_target_static_5p1m:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar_tdm_1_chirp_1024_adc_target_static_5p1m():
    """
    Test that in TDM mode the range estimation is within one range bin width
    Given a 1024 ADC samples per chirp
    """
    from test_assets import radar_tdm_1_chirp_1024_adc, target_static_5p1m, d_5p1m
    c=3e8

    radar = radar_tdm_1_chirp_1024_adc

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp

    adc_times = arange(0, adc_samples_per_chirp, 1)*(1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target_static_5p1m],
                             radars=[radar])

    ranges = ranges_from_fft_threshold(adc_values[0, :],
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=400)

    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    try:
        assert len(ranges) == 1
        assert abs(ranges[0]-d_5p1m) < range_bin_width
    except:
        raise
    else:
        print("test_if_error_radar_tdm_1_chirp_1024_adc_target_static_5p1m:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps():
    """
    Test that in TDM mode the range estimation is within one range bin width
    even with a linear moving target
    Given a 1024 ADC samples per chirp
    """
    from test_assets import radar_tdm_1_chirp_1024_adc, target_linear_speed_5p1m_1mps, d_5p1m
    c=3e8

    radar = radar_tdm_1_chirp_1024_adc

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp

    adc_times = arange(0, adc_samples_per_chirp, 1)*(1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target_linear_speed_5p1m_1mps],
                             radars=[radar])

    ranges = ranges_from_fft_threshold(adc_values[0, :],
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=100)

    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    try:
        assert len(ranges) == 1
        assert abs(ranges[0]-d_5p1m) < range_bin_width
    except:
        raise
    else:
        print("test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps_complex64():
    """
    Test that in TDM mode the range estimation is within one range bin width
    even with a linear moving target
    Given a 1024 ADC samples per chirp
    with datatype being complex64 (instead of float32)
    """
    from test_assets import radar_tdm_1_chirp_1024_adc, target_linear_speed_5p1m_1mps, d_5p1m
    from numpy import complex64
    c=3e8

    radar = radar_tdm_1_chirp_1024_adc
    target1 = target_linear_speed_5p1m_1mps

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp

    adc_times = arange(0, adc_samples_per_chirp, 1)*(1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target1],
                             datatype=complex64,
                             radars=[radar])

    ranges = ranges_from_fft_threshold(adc_values[0, :],
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=100)

    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    try:
        assert len(ranges) == 1
        assert abs(ranges[0]-d_5p1m) < range_bin_width
    except:
        raise
    else:
        print("test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps_complex128():
    """
    Test that in TDM mode the range estimation is within one range bin width
    even with a linear moving target
    Given a 1024 ADC samples per chirp
    with datatype being complex64 (instead of float32)
    """
    from test_assets import radar_tdm_1_chirp_1024_adc, target_linear_speed_5p1m_1mps, d_5p1m
    from numpy import complex128
    c=3e8

    radar = radar_tdm_1_chirp_1024_adc

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp

    adc_times = arange(0, adc_samples_per_chirp, 1)*(1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target_linear_speed_5p1m_1mps],
                             datatype=complex128,
                             radars=[radar])

    ranges = ranges_from_fft_threshold(adc_values[0, :],
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=100)

    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    try:
        assert len(ranges) == 1
        assert abs(ranges[0]-d_5p1m) < range_bin_width
    except:
        raise
    else:
        print("test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps:"+GREEN+"OK"+DEFAULT)


def test_if_radar1_2_targets():
    """
    Test that in TDM mode the range estimation is within one range bin width
    even with a linear moving target
    Given a 1024 ADC samples per chirp
    with datatype being complex128
    with 2 targets
    """
    from test_assets import radar_tdm_1_chirp_1024_adc, target_linear_speed_5p1m_1mps, \
        target_linear_speed_10p1m_1mps, d_5p1m, d_10p1m
    from numpy import complex128
    c=3e8

    radar = radar_tdm_1_chirp_1024_adc

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp

    adc_times = arange(0, adc_samples_per_chirp, 1)*(1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target_linear_speed_5p1m_1mps, target_linear_speed_10p1m_1mps],
                             datatype=complex128,
                             radars=[radar])

    ranges = ranges_from_fft_threshold(adc_values[0, :],
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=100)

    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_samples_per_chirp)
    try:
        assert len(ranges) == 2
        assert abs(ranges[0]-d_5p1m) < range_bin_width
        assert abs(ranges[1]-d_10p1m) < range_bin_width
    except:
        raise
    else:
        print("test_if_error_radar_tdm_1_chirp_1024_adc_target_linear_speed_1mps_complex128:"+GREEN+"OK"+DEFAULT)