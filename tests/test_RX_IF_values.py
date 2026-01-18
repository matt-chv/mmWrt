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

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"

def test_if_error_radar00_target0_8_adc_samples():

    from test_assets import radar00, fif00, target0

    adc_times = arange(0, radar00.number_adc_samples, 1)*(1/radar00.receiver.fs)
    adc_values = adc_samples(adc_times, radar00, [target0],
                             radars=[radar00])
    r_fft = abs(fft(adc_values[0, :]))
    pks = find_peaks(r_fft, height=1)
    pk0 = pks[0][0]
    result = radar00.receiver.fs*pk0/radar00.number_adc_samples
    max_expected_error = radar00.receiver.fs/radar00.number_adc_samples
    try:
        assert abs(result-fif00) < max_expected_error
    except Exception as ex:
        print(RED+f"NOK: expected{max_expected_error}"+DEFAULT)
        print("computed", result)
        print("delta", result-fif00)
        print("bin size", max_expected_error)
        raise
    else:
        print("test_if_1_target_1_chirp:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar11_target0_1024_adc_samples():

    from test_assets import radar11, target0, fif00

    adc_times = arange(0, radar11.number_adc_samples, 1)*(1/radar11.receiver.fs)
    adc_values = adc_samples(adc_times, radar11, [target0],
                             radars=[radar11])
    r_fft = abs(fft(adc_values[0, :]))
    pk0 = find_peaks(r_fft, height=1)[0][0]

    result = radar11.receiver.fs*pk0/radar11.number_adc_samples
    max_expected_error = radar11.receiver.fs/radar11.number_adc_samples

    try:
        assert abs(result-fif00) < max_expected_error
    except Exception as ex:
        print(RED+f"NOK: expected{max_expected_error}"+DEFAULT)
        print("computed", result)
        print("delta", result-fif00)
        print("bin size", max_expected_error)
        raise
    else:
        print("test_if_1_target_1_and_2_chirp_1024samples:"+GREEN+"OK"+DEFAULT)


def test_if_error_radar11_target0_target1_1024_adc_samples():

    from test_assets import radar11, target0, target1, fif00, fif01

    adc_times = arange(0, radar11.number_adc_samples, 1)*(1/radar11.receiver.fs)
    adc_values = adc_samples(adc_times, radar11, [target0, target1],
                             radars=[radar11])

    r_fft = abs(fft(adc_values[0, :]))
    if pk0>pk1:
        pk1_old = pk1
        pk1=pk0
        pk0=pk1_old
    result0 = radar11.receiver.fs*pk0/radar11.number_adc_samples
    result1 = radar11.receiver.fs*pk1/radar11.number_adc_samples
    max_expected_error = radar11.receiver.fs/radar11.number_adc_samples

    try:
        assert abs(result0-fif00) < max_expected_error
        assert abs(result1-fif01) < max_expected_error
    except Exception as ex:
        print(RED+f"NOK: expected{max_expected_error}"+DEFAULT)
        print("pk0,pk1",pk0,pk1)
        print("computed0", result0)
        print("fif00",fif00)
        print("delta0", abs(result0-fif00))
        print("computed1", result1)
        print("fif01",fif01)
        print("delta1", abs(result1-fif01))
        raise
    else:
        print("test_if_1_target_1_and_2_chirp_1024samples:"+GREEN+"OK"+DEFAULT)

if __name__ == "__main__":
    test_if_error_radar00_target0_8_adc_samples()
    test_if_error_radar11_target0_1024_adc_samples()
    test_if_error_radar11_target0_target1_1024_adc_samples()
