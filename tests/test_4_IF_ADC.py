""" This is mostly to test the ADC values
"""
from numpy import arange, pi, real
import numpy as np
import pytest

import sys
from os.path import abspath, join, pardir
dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import adc_samples
from mmWrt.Scene import Target
from tests.test_assets import target_static_5p1m, radar_tdm_1_chirp_8_adc, \
    fif00, target_static_10p1m, fif01
from numpy import complex64

USE_RT = True
if USE_RT:
    from mmWrt.Raytracing import rt_points
    get_adc = rt_points
else:
    from mmWrt.Raytracing import adc_chirp
    get_adc = adc_chirp


@pytest.mark.parametrize("target, radar, datatype, index", [
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, np.complex64, 1),
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, np.float64, 1),
    (target_static_10p1m, radar_tdm_1_chirp_8_adc, np.complex64, 3),
    (target_static_10p1m, radar_tdm_1_chirp_8_adc, np.float64, 3),
])
def tok_fft_peak_index(target, radar, datatype, index):
    """ check that adc values yield a peak in the expect bin
    with radar_tdm_1_chirp_8_adc target_static_5p1m in 1st bin
    and target_static_10p1m in 3rd bin
    """

    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times=adc_times,
                             receiver_radar=radar,
                             targets=[target],
                             radars=[radar],
                             datatype=datatype,
                             debug=True)
    fft_values = np.abs(np.fft.fft(adc_values[0, :]))
    peak_index = np.argmax(fft_values)

    assert peak_index == index


@pytest.mark.parametrize("target, radar, frequency_if, datatype", [
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, fif00, np.complex64),
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, fif00, np.float64),
    (target_static_10p1m, radar_tdm_1_chirp_8_adc, fif01, np.complex64),
    (target_static_10p1m, radar_tdm_1_chirp_8_adc, fif01, np.float64),
])
def tok_adc_values(target, radar, frequency_if, datatype):
    """ check that adc values yield a peak in the expect bin
    and with the expected magnitude by comparing with
    a tone at the same frequency
    """

    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times=adc_times,
                             receiver_radar=radar,
                             targets=[target],
                             radars=[radar],
                             datatype=datatype,
                             debug=True)
    fft_values = np.abs(np.fft.fft(adc_values[0, :]))
    peak_index = np.argmax(fft_values)
    peak_energy = fft_values[peak_index].sum()

    f_tone = np.array([frequency_if]*radar.number_adc_samples)
    tone_values = np.exp(2j*pi*f_tone*adc_times)

    if datatype in [np.float64, np.float32, np.float16]:
        tone_values = np.real(tone_values)
    else:
        pass
    tone_values[0] = 0
    print("adc times", adc_times)
    print("f_tone", frequency_if)
    print("51 tones values", tone_values)
    fft_tone = np.abs(np.fft.fft(tone_values))
    tone_energy = fft_tone[peak_index].sum()

    ratio = abs(peak_energy/tone_energy-1)
    print("ratio", ratio)
    from matplotlib import pyplot as plt
    plt.plot(fft_values, 'b-')
    plt.plot(fft_tone, 'r*')
    plt.show()

    assert  ratio< 1e-6

def tbd_if_dc(targets, radars, datatype):
    """ check that if target in in range bin 0 we have the right DC component
    """
    from numpy import assert_allclose
    c = 3e8

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_samples_per_chirp = radar.receiver.adc_samples_per_chirp
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [targets],
                             radars)
    # assert_allclose

@pytest.mark.parametrize("target, radar, adc_skip, fault_injection", [
    (Target(xt=lambda t: 148 + 0.0*t), radar_tdm_1_chirp_8_adc, 1, "ok"),
    (Target(xt=lambda t: 148 + 0.0*t), radar_tdm_1_chirp_8_adc, 2, "nok"),
    (Target(xt=lambda t: 149 + 0.0*t), radar_tdm_1_chirp_8_adc, 2, "ok"),
    (Target(xt=lambda t: 149 + 0.0*t), radar_tdm_1_chirp_8_adc, 3, "nok"),
])
def tok_tof(target, radar, adc_skip, fault_injection):
    """ check that if target far away,
    first values of adc values are 0 for sampling time
    for samples happening before full time of flight
    """
    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target],
                             [radar])
    print("ADC values", adc_values)
    try:
        assert adc_values[0, :adc_skip].sum() == 0
    except:
        if fault_injection == "ok":
            raise
    else:
        if fault_injection == "nok":
            raise


def tbd_BBIF2():
    """ TX-RX < LPF so returns sinewave
    res = BB_IF(array([0, 1.3e-7, 2.6e-7, 4e-7,
                    5.3e-7, 6.6e-7, 8e-7, 9.3e-7,
                    1e-6, 1.2e-6, 1.3e-6, 1.46e-6,
                    1.6e-6, 1.73e-6, 1.86e-6, 2e-6]),
            array([6.001e9, 6.007e9,6.014e9,6.021e9,
                    6.027e9,6.034e9,6.041e9,6.047e9,
                    6.054e9,6.061e9,6.067e9,6.074e9,
                    6.081e9,6.087e9,6.094e9,6.101e9]),
            array([6e9,6.006e9,6.013e9,6.02e9,
                    6.026e9,6.033e9,6.04e9,6.046e9,
                    6.053e9,6.06e9,6.066e9,6.073e9,
                    6.08e9,6.086e9,6.093e9,6.1e9]))
    print("RES", res)"""

    from numpy import exp, pi
    f1 = 1e6
    NA = 16
    times = linspace(0, 2/f1, NA)
    f_rx = linspace(6e9, 6.1e9, NA)
    f_tx = f_rx + f1
    adcs = BB_IF(times, f_rx, f_tx, rx_hpf=1e3, rx_lpf=1e8, tx_phase_offset=0)
    expected = exp(2*pi*1j*times*f1)
    try:
        assert allclose(adcs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", adcs)
        raise Exception(ex)
    else:
        print("passed test_BBIF2")


def tbd_adc_frame_v2():
    """ Check ADC values for multiple chirps and multiple frames
    !!!!!!!!!!!!

    a faire
        verifier phase pour TDM et pour DDM dans chirp (FFT?)
        par chirp idx
        par frame idx
    """


def tbd_BBIF1():
    # TX is too high, IF should be zeros
    times = linspace(0, 1e-6, 4)
    f_rx = array([6.00000000e+10,6.00033333e+10,6.00066667e+10,6.00100000e+10])
    f_tx = f_rx + 1e9
    adcs = BB_IF(times, f_rx, f_tx, rx_hpf=1e3, rx_lpf=1e8, tx_phase_offset=0)
    expected = zeros(4)
    try:
        assert allclose(adcs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", adcs)
        raise
    else:
        print("passed test_BBIF1")

def tbd_phase_delta_chirp_to_chirp():
    """ test the phase change in the target range bin change from one
    chirp to another chirp is within the expected error range

    for DFT, the d(phase)/dt = [phase(DFT[idx_peak_1st_chirp])-phase(DFT[idx_peak_2nd_chirp]) ] / (t_inter_chirp)
    the expected change of phase from chirp to chirp is for a target moving at velocity v (assuming no range bin change)
    dphase_dt = 4*pi*v/lambda_60G
    given the number of DFT bins in the velocity dimension (doppler dimension), which is the number of chirps
    the maximum error is 1 range bin:
    2*pi/t_inter_chirp/number_adc_samples
    """  # noqa E501

    from test_assets import radar_tdm_2_chirp_8adc, \
        target_linear_speed_1mps, \
        dphase_dt_1mps

    number_samples = radar_tdm_2_chirp_8adc.receiver.number_adc_samples
    t_inter_chirp = radar_tdm_2_chirp_8adc.t_inter_chirp

    chirp_idx = tile(arange(0,
                            radar_tdm_2_chirp_8adc.transmitter.chirps_count),
                     radar_tdm_2_chirp_8adc.transmitter.frames_count)
    frame_idx = repeat(arange(0,
                              radar_tdm_2_chirp_8adc.transmitter.frames_count),
                       radar_tdm_2_chirp_8adc.transmitter.chirps_count)
    start_time = radar_tdm_2_chirp_8adc.t_inter_frame*frame_idx + \
        t_inter_chirp*chirp_idx + \
        radar_tdm_2_chirp_8adc.transmitter.tx_start_time

    end_time = start_time + radar_tdm_2_chirp_8adc.transmitter.ramp_end_time
    adc_times_2d = linspace(start_time, end_time,
                            num=number_samples,
                            axis=1)
    adc_times = adc_times_2d.flatten()

    adc_values = adc_samples(adc_times, radar_tdm_2_chirp_8adc,
                             [target_linear_speed_1mps],
                             radars=[radar_tdm_2_chirp_8adc],
                             debug=True)

    r_fft_1st_chirp = fft(adc_values[0, :number_samples])
    r_fft_2nd_chirp = fft(adc_values[0, number_samples:])

    peak_1st_chirp = find_peaks(abs(r_fft_1st_chirp), height=1)[0][0]
    peak_2nd_chirp = find_peaks(abs(r_fft_2nd_chirp), height=1)[0][0]
    phase_peak_1st_chirp = angle(r_fft_1st_chirp[peak_1st_chirp])
    phase_peak_2nd_chirp = angle(r_fft_2nd_chirp[peak_2nd_chirp])

    dphase_dt = (phase_peak_1st_chirp-phase_peak_2nd_chirp)/t_inter_chirp

    max_expected_dphase_dt_error = 2*pi/t_inter_chirp/number_samples

    try:
        assert abs(dphase_dt-dphase_dt_1mps) < max_expected_dphase_dt_error
    except Exception as ex:
        print(RED+f"NOK: expected{dphase_dt_1mps}"+DEFAULT)
        print("computed", dphase_dt)
        print("delta", dphase_dt_1mps-dphase_dt)
        print("bin size", max_expected_dphase_dt_error)
        raise Exception(ex)
    else:
        print("test_if_1_target_1_chirp:"+GREEN+"OK"+DEFAULT)

def tbd_if_error_radar00_target0_8_adc_samples():

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


def tbd_if_error_radar11_target0_1024_adc_samples():

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


def tbd_if_error_radar11_target0_target1_1024_adc_samples():

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