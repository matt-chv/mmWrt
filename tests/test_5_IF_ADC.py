""" testing sample_all_rays to ensure ADC values are correct
v0.0.11: 1
"""
from numpy import arange, pi, real
import numpy as np
from scipy.signal import find_peaks
import pytest

import sys
from os.path import abspath, join, pardir
dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import sample_all_rays
from mmWrt.Scene import Scatterer
from tests.test_assets import scatterer_static_0, scatterer_static_5p1m, radar_tdm_1_chirp_8_adc, \
    fif00, scatterer_static_10p1m, fif01, adc_sampling_frequency_0, adc_8_values_complex_fif00, \
    adc_sampling_times_8_samples, scatterer_5p1m_radar1_bin_1, scatterer_10p1m_radar1_bin_3
from numpy import complex64, allclose


def test_adc_simple():
    # test that given a given scatterer, we get expected adc value
    from test_assets import adc_sampling_times_8_samples, adc_8_values_complex_fif00
    # timesamples = adc_sampling_times_8_samples  # [:, None, None, None]

    adc_values = sample_all_rays(adc_sampling_times_8_samples,
                                 [radar_tdm_1_chirp_8_adc],
                                 [scatterer_static_5p1m],
                                 radar_tdm_1_chirp_8_adc)
    adc0 = adc_values[:, 0]
    expected = np.real(adc_8_values_complex_fif00)

    assert adc0.shape == adc_8_values_complex_fif00.shape, "shapes don't match"
    assert np.allclose(adc0, expected, atol=1e-2), f"adc0:{adc0} vs expected: {expected}"

"""
@pytest.mark.parametrize("scatterer, radar, datatype, index", [
    (scatterer_static_5p1m, tof_5p1m, radar_tdm_1_chirp_8_adc, np.complex64, scatterer_5p1m_radar1_bin_1),
    # (scatterer_static_5p1m, tof_5p1m, radar_tdm_1_chirp_8_adc, np.float64, scatterer_5p1m_radar1_bin_1),
    # (scatterer_static_10p1m, tof_10p1m, radar_tdm_1_chirp_8_adc, np.complex64, 3),
    # (scatterer_static_10p1m, tof_10p1m, radar_tdm_1_chirp_8_adc, np.float64, 3),
])
def test_fft_peak_index(scatterer, tof, radar, datatype, index):
    # check that adc values yield a peak in the expect bin
    #with radar_tdm_1_chirp_8_adc scatterer_static_5p1m in 1st bin
    #and scatterer_static_10p1m in 3rd bin

    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_times = adc_times[:, None, None, None]

    #YIF += receiver_radar.adc_sampling(f_if=f_if,
    #                                       ph_rx=ph_rx,
    #                                       adc_times=adc_times,
    #                                       time_of_flight=time_of_flight,

    # FIXME: change the ph_rx to be including the time of flight
    adc_values = radar.adc_sampling(f_if=f_if,
                                    ph_rx=ph_rx,
                                    adc_times=adc_times,
                                    time_of_flight=tof)
    print("adc_values.shape", adc_values.shape)
    fft_values = np.abs(np.fft.fft(adc_values[0, :]))
    peak_index = np.argmax(fft_values)

    print(45, peak_index, index)
    assert peak_index == index


@pytest.mark.parametrize("scatterer, radar, frequency_if, datatype", [
    (scatterer_static_5p1m, radar_tdm_1_chirp_8_adc, fif00, np.complex64),
    (scatterer_static_5p1m, radar_tdm_1_chirp_8_adc, fif00, np.float64),
    (scatterer_static_10p1m, radar_tdm_1_chirp_8_adc, fif01, np.complex64),
    (scatterer_static_10p1m, radar_tdm_1_chirp_8_adc, fif01, np.float64),
])
def t0est0_adc_values(scatterer, radar, frequency_if, datatype):
    # check that adc values yield a peak in the expect bin
    # and with the expected magnitude by comparing with
    # a tone at the same frequency

    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times=adc_times,
                             receiver_radar=radar,
                             scatterers=[scatterer],
                             radars=[radar],
                             datatype=datatype,
                             debug=True)
    fft_values = np.abs(np.fft.fft(adc_values[0, :]))
    peak_index = np.argmax(fft_values)
    if frequency_if == fif00:
        assert peak_index == scatterer_5p1m_radar1_bin_1
    else:
        assert peak_index == scatterer_10p1m_radar1_bin_3
    peak_energy = fft_values[peak_index].sum()

    f_tone = np.array([frequency_if]*radar.number_adc_samples)
    tone_values = np.exp(2j*pi*f_tone*adc_times)

    if datatype in [np.float64, np.float32, np.float16]:
        tone_values = np.real(tone_values)
    else:
        pass
    tone_values[0] = 0
    fft_tone = np.abs(np.fft.fft(tone_values))
    tone_energy = fft_tone[peak_index].sum()

    ratio = abs(peak_energy/tone_energy-1)
    assert  ratio < 1e-2


def fixme_if_dc():
    # check that if scatterer in in range bin 0 we have the right DC component
    # FIXME:does not work yet as returned values are [0,1,1,1,1,1] with current code
    radar = radar_tdm_1_chirp_8_adc
    scatterer = scatterer_static_0
    adc_times = arange(0, 8/adc_sampling_frequency_0,
                       1/adc_sampling_frequency_0)
    adc_values = adc_samples(adc_times, radar,
                             [scatterer],
                             [radar])
    assert allclose(adc_values, np.zeros(8))


@pytest.mark.parametrize("scatterer, radar, adc_skip, fault_injection", [
    (Scatterer(xt=lambda t: 148 + 0.0*t), radar_tdm_1_chirp_8_adc, 1, "ok"),
    (Scatterer(xt=lambda t: 148 + 0.0*t), radar_tdm_1_chirp_8_adc, 2, "nok"),
    (Scatterer(xt=lambda t: 149 + 0.0*t), radar_tdm_1_chirp_8_adc, 2, "ok"),
    (Scatterer(xt=lambda t: 149 + 0.0*t), radar_tdm_1_chirp_8_adc, 3, "nok"),
])
def t0est_tof(scatterer, radar, adc_skip, fault_injection):
    # check that if scatterer far away,
    # first values of adc values are 0 for sampling time
    # for samples happening before full time of flight
    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [scatterer],
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
    # TX-RX < LPF so returns sinewave
    #res = BB_IF(array([0, 1.3e-7, 2.6e-7, 4e-7,
    #                5.3e-7, 6.6e-7, 8e-7, 9.3e-7,
    #                1e-6, 1.2e-6, 1.3e-6, 1.46e-6,
    #                1.6e-6, 1.73e-6, 1.86e-6, 2e-6]),
    #        array([6.001e9, 6.007e9,6.014e9,6.021e9,
    #                6.027e9,6.034e9,6.041e9,6.047e9,
    #                6.054e9,6.061e9,6.067e9,6.074e9,
    #                6.081e9,6.087e9,6.094e9,6.101e9]),
    #        array([6e9,6.006e9,6.013e9,6.02e9,
    #                6.026e9,6.033e9,6.04e9,6.046e9,
    #                6.053e9,6.06e9,6.066e9,6.073e9,
    #                6.08e9,6.086e9,6.093e9,6.1e9]))
    #print("RES", res)

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


@pytest.mark.parametrize("chirp_idx, frame_idx, adc_values_expected", [
    (0, 0, adc_8_values_complex_fif00),
    (1, 0, adc_8_values_complex_fif00),
    (2, 0, np.zeros(8)),
    (0, 1, adc_8_values_complex_fif00),
    (1, 1, adc_8_values_complex_fif00),
    (2, 1, np.zeros(8)),
    (0, 2, np.zeros(8)),
    (1, 2, np.zeros(8)),
    (2, 2, np.zeros(8)),
])
def t0est0_adc_frame(chirp_idx, frame_idx, adc_values_expected):
    # Check ADC values for multiple chirps yield the right values
    # radar_tdm_2_chirp_8adc only transmits 2 chirps
    # check that the first 2 chirps yield adc_8_values_complex_fif00
    # and taht third chirp yields 0 since nothing transmitted
    # same for 2nd frame
    # 3rd frame is only 0 since not transmitted
    #
    # TODO
    #    verifier phase pour TDM et pour DDM dans chirp (FFT?)
    #    par chirp idx
    #    par frame idx
    #
    from test_assets import radar_tdm_2_frames_2_chirps_8adc, chirp_period_vmax_3mps, \
        frame_period_50ms
    radar = radar_tdm_2_frames_2_chirps_8adc
    scatterer = scatterer_static_5p1m
    adc_times = arange(0, 8/adc_sampling_frequency_0,
                       1/adc_sampling_frequency_0) + \
        chirp_period_vmax_3mps*chirp_idx + \
        frame_period_50ms * frame_idx
    adc_values = adc_samples(adc_times, radar,
                             [scatterer],
                             [radar],
                             datatype=complex64)
    print("adc values",chirp_idx, frame_idx, adc_values, adc_values_expected)
    assert allclose(adc_values, adc_values_expected)


def t0est0_MRE():
    from test_assets import lambda_60G
    from mmWrt.Scene import Radar, Transmitter, Receiver
    from numpy.fft import fft
    from numpy import angle
    from numpy import float32
    f0_min = 60e9
    lambda0_max = 3e8/f0_min
    adc_sampling_frequency = 50e6
    adc_samples_count = 512
    v=100
    print("-"*20)
    radar = Radar(transmitter=Transmitter(chirp_start_freq=60e9,
                                          chirp_slope=10e12,
                                          chirp_end_time=adc_samples_count*1/adc_sampling_frequency),
                  receiver=Receiver(adc_sample_rate=adc_sampling_frequency,
                                    max_fs=100e6,
                                    adc_sample_count=adc_samples_count))
    n_samples = adc_samples_count
    ts = 1/adc_sampling_frequency
    T = arange(0, n_samples*ts+ts, ts)
    print("T", T[:10])
    adc_times = arange(0, adc_samples_count/adc_sampling_frequency,
                       1/adc_sampling_frequency)
    f_if = 666666.6666666666
    val = np.real(np.exp(2 * pi * 1j * (f_if) * T[:10] ))
    print("vals", val)
    # print(adc_times)
    t_chirp_to_chirp = 1.2e-6
    scatterer0 = Scatterer(xt= lambda t: 10+0*t)
    adc0 = adc_samples(adc_times,
                       radar,
                       [scatterer0],
                       radars=[radar],
                       datatype=float32)
    print(adc0[0, :10])
    scatterer1 = Scatterer(xt=lambda t: 10 + v*t_chirp_to_chirp+0*t)
    adc1 = adc_samples(adc_times,
                       radar,
                       [scatterer1],
                       radars=[radar],
                       datatype=float32)
    print("adc1", adc1[0, :10])
    FT0 = fft(adc0[0,:adc_samples_count//2])
    FT1 = fft(adc1[0, :adc_samples_count//2])
    MAG0 = np.abs(FT0)
    amplitude_peak0 = sorted(MAG0, reverse = True)[0]
    i_peak0 = list(MAG0).index(amplitude_peak0)

    MAG1 = np.abs(FT1)
    amplitude_peak1 = sorted(MAG1, reverse = True)[0]
    i_peak1 = list(MAG1).index(amplitude_peak1)
    assert i_peak0 == i_peak1
    ANG0 = angle(FT0)
    ANG1 = angle(FT1)

    ph0 = ANG0[i_peak0]
    ph1 = ANG1[i_peak1]
    # d, v, ph0 10 100.0 -1.5646604036459237 -1.2627114022166515
    v_est = lambda0_max*(ph1-ph0)/(4*pi*t_chirp_to_chirp)
    print(f"speed is:{v:.2g} is computed to be at {v_est:.2g}")
    assert v_est == 109.72949873542714


def t0est0_phase_delta_chirp_to_chirp():
    # test the phase change in the scatterer range bin change from one
    # chirp to another chirp is within the expected error range
    # 
    # for DFT, the d(phase)/dt = [phase(DFT[idx_peak_1st_chirp])-phase(DFT[idx_peak_2nd_chirp]) ] / (chirp_period)
    #the expected change of phase from chirp to chirp is for a scatterer moving at velocity v (assuming no range bin change)
    #dphase_dt = 4*pi*v/lambda_60G
    #given the number of DFT bins in the velocity dimension (doppler dimension), which is the number of chirps
    #the maximum error is 1 range bin:
    #2*pi/chirp_period/number_adc_samples
    #

    from test_assets import radar_tdm_2_chirp_8adc, \
        scatterer_linear_speed_5p1m_1mps, \
        dphase_dt_1mps, chirp_period_vmax_3mps

    radar = radar_tdm_2_chirp_8adc
    number_chirps = radar.transmitter.chirp_count

    adc_times0 = adc_sampling_times_8_samples

    adc_values_first_chirp = adc_samples(adc_times0,
                             radar,
                             [scatterer_linear_speed_5p1m_1mps],
                             radars=[radar],
                            datatype=complex64,
                             debug=True)

    adc_times1 = adc_sampling_times_8_samples + chirp_period_vmax_3mps
    expected_adc_values = np.array([[ 0.        +        0.j,  0.52017745 + 0.85405821j,
                                     -0.4907899 +0.87127795j,
                                     -0.99999746-0.00225504j, -0.48685503 - 0.87348279j,
                                      0.52402542-0.85170262j,
                                      0.99916655+0.04081925j,  0.45280501 + 0.89160957j]])
    assert allclose(adc_values_first_chirp, expected_adc_values)

    adc_values_second_chirp = adc_samples(adc_times1,
                                          radar,
                                          [scatterer_linear_speed_5p1m_1mps],
                                          radars=[radar],
                                          datatype=complex64,
                                          debug=True)
    expected_adc_values = np.array([[ 0.        +0.j,         -0.47961799+0.8774774j,  -0.99994555+0.01043559j,
                                     -0.49782616-0.86727684j,  0.51332231-0.8581959j,   0.99959681+0.02839393j,
                                      0.4637788 +0.88595103j, -0.54625576+0.83761843j]])
    assert allclose(adc_values_second_chirp, expected_adc_values)

    r_fft_1st_chirp = np.fft.fft(adc_values_first_chirp[0, :])
    r_fft_2nd_chirp = np.fft.fft(adc_values_second_chirp[0, :])

    peak_1st_chirp = find_peaks(abs(r_fft_1st_chirp), height=1)[0][0]
    peak_2nd_chirp = find_peaks(abs(r_fft_2nd_chirp), height=1)[0][0]
    assert peak_1st_chirp == scatterer_5p1m_radar1_bin_1
    assert peak_2nd_chirp == scatterer_5p1m_radar1_bin_1
    phase_peak_1st_chirp = np.angle(r_fft_1st_chirp[peak_1st_chirp])
    phase_peak_2nd_chirp = np.angle(r_fft_2nd_chirp[peak_2nd_chirp])
    dphase_dt = (phase_peak_1st_chirp-phase_peak_2nd_chirp)/chirp_period_vmax_3mps
    assert dphase_dt == -2514.0893436956576

    max_expected_dphase_dt_error = 2*pi/chirp_period_vmax_3mps/number_chirps
    assert abs(dphase_dt-dphase_dt_1mps) < max_expected_dphase_dt_error


def t0est0_if_error_radar00_scatterer0_8_adc_samples():

    from test_assets import radar_tdm_1_chirp_8_adc, fif00, scatterer_static_5p1m
    radar00 = radar_tdm_1_chirp_8_adc
    scatterer0 = scatterer_static_5p1m

    adc_times = arange(0, radar00.number_adc_samples, 1)*(1/radar00.receiver.adc_sample_rate)
    adc_values = adc_samples(adc_times, radar00, [scatterer0],
                             radars=[radar00])
    r_fft = abs(np.fft.fft(adc_values[0, :]))
    pks = find_peaks(r_fft, height=1)
    pk0 = pks[0][0]
    result = radar00.receiver.fs*pk0/radar00.number_adc_samples
    max_expected_error = radar00.receiver.fs/radar00.number_adc_samples
    assert abs(result-fif00) < max_expected_error

def t0est0_if_error_radar11_scatterer0_1024_adc_samples():

    from test_assets import radar_tdm_1_chirp_1024_adc, scatterer_static_5p1m, fif00
    radar11 = radar_tdm_1_chirp_1024_adc
    scatterer0 = scatterer_static_5p1m

    adc_times = arange(0, radar11.number_adc_samples, 1)*(1/radar11.receiver.fs)
    adc_values = adc_samples(adc_times, radar11, [scatterer0],
                             radars=[radar11])
    r_fft = abs(np.fft.fft(adc_values[0, :]))
    pk0 = find_peaks(r_fft, height=1)[0][0]

    result = radar11.receiver.fs*pk0/radar11.number_adc_samples
    max_expected_error = radar11.receiver.fs/radar11.number_adc_samples

    assert abs(result-fif00) < max_expected_error


def t0est0_if_error_radar11_scatterer0_scatterer1_1024_adc_samples():
    # Test the presence of 2 tones at the right positions
    # when two scatterers are introduced
    # 
    # in case of algorithmic changes, first checks for the
    # tone to be within the range resolution then checks
    # the indexes against known good values
    #

    from test_assets import radar_tdm_1_chirp_1024_adc, scatterer_static_5p1m, scatterer_static_10p1m, fif00, fif01
    radar11 = radar_tdm_1_chirp_1024_adc
    scatterer0 = scatterer_static_5p1m
    scatterer1 = scatterer_static_10p1m

    adc_times = arange(0, radar11.number_adc_samples, 1)*(1/radar11.receiver.fs)
    adc_values = adc_samples(adc_times, radar11, [scatterer0, scatterer1],
                             radars=[radar11])

    r_fft = abs(np.fft.fft(adc_values[0, :]))
    pk0 = find_peaks(r_fft, height=1)[0][0]
    pk1 = find_peaks(r_fft, height=1)[0][1]
    if pk0 > pk1:
        pk1_old = pk1
        pk1 = pk0
        pk0 = pk1_old
    result0 = radar11.receiver.fs*pk0/radar11.number_adc_samples
    result1 = radar11.receiver.fs*pk1/radar11.number_adc_samples
    max_expected_error = radar11.receiver.fs/radar11.number_adc_samples

    assert abs(result0-fif00) < max_expected_error
    assert abs(result1-fif01) < max_expected_error

    assert pk0 == 172
    assert pk1 == 341
"""