""" This is mostly to test the ADC part of the Raytracing
"""

from os.path import abspath, join, pardir
import sys
from numpy import arange
from numpy.fft import fft
from scipy.signal import find_peaks

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import adc_samples
from mmWrt.RadarSignalProcessing import ranges_from_fft_threshold

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"

def test_if_error_radar00_target0_8_adc_samples():

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
        print("test_if_error_radar00_target0_8_adc_samples:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar00_target0_1024_adc_samples():

    from test_assets import radar_tdm_1_chirp_1024_adc, target_static_5p1m, d_5p1m
    c=3e8

    adc_sample_rate = radar_tdm_1_chirp_1024_adc.receiver.adc_sample_rate
    chirp_slope = radar_tdm_1_chirp_1024_adc.transmitter.chirp_slope
    adc_samples_per_chirp = radar_tdm_1_chirp_1024_adc.receiver.adc_samples_per_chirp

    adc_times = arange(0, adc_samples_per_chirp, 1)*(1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar_tdm_1_chirp_1024_adc,
                             [target_static_5p1m],
                             radars=[radar_tdm_1_chirp_1024_adc])

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
        print("test_if_error_radar00_target0_8_adc_samples:"+GREEN+"OK"+DEFAULT)

if __name__ == "__main__":
    test_if_error_radar00_target0_8_adc_samples()
    test_if_error_radar00_target0_1024_adc_samples()