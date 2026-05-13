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

@pytest.mark.parametrize("target, radar, frequency_if", [
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, fif00),
    (target_static_10p1m, radar_tdm_1_chirp_8_adc, fif01),
])
def tok_if_freq0(target, radar, frequency_if):
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


@pytest.mark.parametrize("target, radar, frequency_if, datatype", [
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, fif00, np.complex64),
    (target_static_5p1m, radar_tdm_1_chirp_8_adc, fif00, np.float64),
])
def tok_adc_values(target, radar, frequency_if, datatype):
    """ check that adc values are those of a tone
    whose frequency is close to theoretical one
    """

    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [target],
                             [radar],
                             debug=True)
    fft_values = np.abs(np.fft.fft(adc_values[0, :]))
    peak_index = np.argmax(fft_values)
    peak_energy = fft_values[peak_index].sum()

    f_tone = np.array([frequency_if]*radar.number_adc_samples)
    tone_values = np.real(np.exp(2j*pi*f_tone*adc_times))
    if datatype in [np.float64, np.float32, np.float16]:
        tone_values = np.real(tone_values)
    else:
        pass
    tone_values[0] = 0
    fft_tone = np.abs(np.fft.fft(tone_values))
    tone_energy = fft_tone[peak_index].sum()

    assert abs(peak_energy/tone_energy-1) < 1e-6

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