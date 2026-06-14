""" Tests for CFAR and 1D peak grouping
v0.0.11: 26
"""

from os.path import abspath, join, pardir
import re
from semver import VersionInfo
import sys
import pytest

import numpy as np
from numpy import where
from numpy import complex128 as complex
from numpy import float32  # alternatives: float16, float64

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Scatterer  # noqa: E402
from mmWrt.Scene import ERR_TARGET_T0, ERR_TFFT_lte_TC  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402
from mmWrt.RadarSignalProcessing import cfar_ca, cfar_alpha, ERR_CFAR_CELL_COUNT

# from test_1_range_point import __range__wrapper  # noqa: E402
from test_assets import scatterer_static_5p1m, radar_tdm_1_chirp_8_adc, d_5p1m

def __range__wrapper2(scatterers, radars,
                      distances,
                      cfar_peak_detect=False,
                      data_type=float32,
                      peak_threshold=2):
    from test_1_range_point import adc_samples
    from numpy import arange
    c = 3e8

    radar = radars[0]

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_sample_count = radar.receiver.adc_sample_count

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_sample_count = radar.receiver.adc_sample_count
    adc_times = arange(0, radar.number_adc_samples, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             scatterers,
                             radars=radars)
    if cfar_peak_detect:
        ranges = rsp.ranges_dft_cfar(adc_values[0, :],
                                    chirp_slope=chirp_slope,
                                    adc_sample_rate=adc_sample_rate,
                                    fft_threshold=peak_threshold)

    else:
        ranges = rsp.ranges_from_fft_threshold(adc_values[0, :],
                                        chirp_slope=chirp_slope,
                                        adc_sample_rate=adc_sample_rate,
                                        fft_threshold=peak_threshold)
    
    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_sample_count)

    for idx, _ in enumerate(distances):
        error = abs(ranges[idx]-distances[idx])
        try:
            assert  error < range_bin_width
        except Exception as ex:
            raise ValueError("Error too large")


def tbd_tdm_8adc_scatterer0():
    __range__wrapper(scatterer_idxes=[0], radars_idxes=[0],
                     distance_idxes=[0],
                     cfar_peak_detect=True)

@pytest.mark.parametrize("scatterers, radars, distances, cfar_peak_detect", [
    ([scatterer_static_5p1m], [radar_tdm_1_chirp_8_adc], [d_5p1m], True),
])
def tbd_tdm_8adc_range0m(scatterers, radars,
                     distances,
                     cfar_peak_detect):
    """ Test CFAR on range bin 0 as np.find_peaks does not work on range bin 0, 
    currently also broken with CFAR needs debug
    so we need to ensure CFAR can handle it. """
    __range__wrapper2(scatterers, radars,
                      distances,
                      cfar_peak_detect)

def test_cfar_length():
    # if the fake_fft is too short we expect
    # a ValueError exception
    mag_length=1
    # ensures cfar shape is the same as input
    fake_fft = np.ones(mag_length)
    try:
        threshold = cfar_ca(fake_fft, guard_cell_count=1,
                            train_cell_count=2, pfa=0.01)
    except Exception as ex:
        assert str(ex) == ERR_CFAR_CELL_COUNT
    else:
        assert False


@pytest.mark.parametrize("mag_length", [
    (2**idx) for idx in range(3, 10)])
def test_rsp_cfar_length(mag_length):
    # ensures cfar shape is the same as input
    fake_fft = np.ones(mag_length)
    threshold = cfar_ca(fake_fft, guard_cell_count=1,
                        train_cell_count=2, pfa=0.01)
    assert threshold.shape == fake_fft.shape, "CFAR threshold should have same length as input"


def test_rsp__cfar_ca__threshold():
    """ bypassing pfa, to check the thresholds values are as expected"""
    from mmWrt.RadarSignalProcessing import _cfar_ca
    fake_fft = np.ones(16)
    threshold = _cfar_ca(fake_fft, guard_cell_count=1,
                         train_cell_count=2, threshold_factor=1)
    assert threshold[10] == 1


def test_rsp_cfar_ex1():
    # Example 1: Peak at first range bin
    fft = np.array([100.0+0j, 1.0+0j, 1.0+0j, 1.0+0j, 1.0+0j, 1.0+0j, 0, 0])
    threshold = cfar_ca(fft, guard_cell_count=1, train_cell_count=2, pfa=0.01)
    magnitude = np.abs(fft)
    peak_idx = np.argmax(magnitude)
    assert peak_idx == np.where(magnitude > threshold)[0][0]


def test_rsp_cfar_ex2():
    from mmWrt.RadarSignalProcessing import cfar_ca
    # Example 2: Peak in the middle
    fft = np.array([1.0+0j, 1.0+0j, 100.0+0j, 1.0+0j, 1.0+0j, 1.0+0j, 0, 0])
    threshold = cfar_ca(fft, guard_cell_count=1, train_cell_count=2, pfa=0.01)
    magnitude = np.abs(fft)
    peak_idx = np.argmax(magnitude)
    assert peak_idx == np.where(magnitude > threshold)[0][0]


def test_noise_floor_calibration():
    # Threshold at an interior bin equals alpha * mean(training cells).
    n_train, n_guard, pfa = 4, 1, 1e-3
    noise_floor = 3.7  # arbitrary non-unit value to catch scaling bugs

    x = np.full(30, noise_floor)
    threshold = cfar_ca(x, guard_cell_count=n_guard,
                        train_cell_count=n_train, pfa=pfa)

    alpha = cfar_alpha(n_train, pfa)
    cut = 15  # arbitrary interior bin, well clear of edges

    assert threshold[cut] == pytest.approx(alpha * noise_floor)


@pytest.mark.parametrize("peak_idx", [
    (idx) for idx in range(8)])
def test_rsp_cfar_ex3(peak_idx: int):
    from mmWrt.RadarSignalProcessing import cfar_ca
    # Example 3: Peak in the middle
    adc_count = 16
    time_values = np.arange(0, 1, 1/adc_count)
    adc_values = np.exp(2 * 1j * np.pi * peak_idx * time_values)
    fft_values = np.fft.fft(adc_values)
    threshold = cfar_ca(fft_values, guard_cell_count=1, train_cell_count=3,
                        pfa=1e-6)
    magnitude = np.abs(fft_values)

    assert np.where((magnitude > threshold + 1e-10))[0].size > 0, f"no peaks detected but should have been at idx: {peak_idx}"
    if np.where((magnitude > threshold + 1e-10))[0].size > 0:
        assert peak_idx == np.where((magnitude > threshold + 1e-10))[0][0], f"peaks detected at {np.where((magnitude > threshold + 1e-10))[0][0]} but should have been at idx: {peak_idx}"


def test_rsp_grouping_ex1():
    mag = np.array([0, 0, 10, 9, 0, 0, 8, 7, 0, 0])
    idx = np.array([2, 3, 6, 7])
    grouped_idx = rsp.peak_grouping_1d(idx, mag)
    assert np.allclose(grouped_idx, np.array([2, 6]))


def test_rsp_grouping_ex2():
    mag = np.array([10, 5, 0, 0, 0, 0, 8, 7, 0, 0])
    idx = np.array([0, 1, 6, 7])
    grouped_idx = rsp.peak_grouping_1d(idx, mag)
    assert np.allclose(grouped_idx, np.array([0, 6]))


def test_rsp_grouping_ex3():
    mag = np.array([5, 15, 0, 0, 0, 0, 8, 7, 0, 0])
    idx = np.array([0, 1, 6, 7])
    grouped_idx = rsp.peak_grouping_1d(idx, mag)
    assert np.allclose(grouped_idx, np.array([1, 6]))

def test_rsp_grouping_ex4():
    mag = np.array([5, 15, 0, 0, 0, 0, 0, 0, 8, 7])
    idx = np.array([0, 1, 8, 9])
    grouped_idx = rsp.peak_grouping_1d(idx, mag)
    assert np.allclose(grouped_idx, np.array([1, 8]))


def test_rsp_grouping_ex5():
    mag = np.array([5, 15, 0, 0, 0, 0, 0, 0, 1, 10])
    idx = np.array([0, 1, 8, 9])
    grouped_idx = rsp.peak_grouping_1d(idx, mag)
    assert np.allclose(grouped_idx, np.array([1, 9])), f"Expected peak at index 1 and 9, got {grouped_idx}"


def test_rsp_grouping_ex6():
    mag = np.array([15, 5, 0, 0, 0, 0, 0, 0, 0, 10])
    idx = np.array([0, 1, 9])
    grouped_idx = rsp.peak_grouping_1d(idx, mag)
    assert np.allclose(grouped_idx, np.array([0, 9]))


def tbd_test_FMCW_1j():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    scatterers = [scatterer1, scatterer2]
    bb = rt_points(radar, scatterers,
                   datatype=complex,
                   debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(scatterer_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_scatterers = [Scatterer(Distances[i]) for i in grouped_peaks]
    error = rsp.error(scatterers, found_scatterers)
    assert error < 3


def tbd_test_FMCW_radar_equation():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    # adding RCS to ensure scatterers are detected ...
    scatterer1 = Scatterer(5.1, rcs_f=lambda f: 10824)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10, rcs_f=lambda f: 43000)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, radar_equation=True, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(scatterer_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_scatterers = [Scatterer(Distances[i]) for i in grouped_peaks]
    error = rsp.error(scatterers, found_scatterers)
    assert error < 3


def tbd_test_FMCW_radar_equation_corner_reflector():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    # adding RCS to ensure scatterers are detected ...
    scatterer1 = Scatterer(5.1, rcs_f=lambda f: 10824,
                     scatterer_type="corner_reflector")
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10, rcs_f=lambda f: 43000)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, radar_equation=True, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(scatterer_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_scatterers = [Scatterer(Distances[i]) for i in grouped_peaks]
    error = rsp.error(scatterers, found_scatterers)
    try:
        assert error < 3
    except Exception as ex:  # pragma: no cover
        print("found scatterers", [str(t) for t in found_scatterers])
        print("expected scatterers", [str(t) for t in scatterers])
        raise Exception(str(ex))


def tbd_test_FMCW_real_adc_po2():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  adc_po2=True,
                  debug=True)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(scatterer_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_scatterers = [Scatterer(Distances[i]) for i in grouped_peaks]
    error = rsp.error(scatterers, found_scatterers)
    assert error < 3


def tbd_test_FMCW_real_error():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(scatterer_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_scatterers = [Scatterer(Distances[i]) for i in grouped_peaks]
    error = rsp.error(scatterers, found_scatterers)
    assert error < 3


def tbd_test_FMCW_no_scatterers_found_error():
    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    error = rsp.error([scatterer2, scatterer1], [])
    assert error == 15.1


def tbd_test_FMCW_ADC_fs_vs_fs_max():
    str_ex = ""
    try:
        _ = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=1e6,
                                    max_fs=1e5,
                                    max_adc_buffer_size=2048, debug=True))
    except ValueError as ex:
        str_ex = str(ex)
    assert str_ex == "ADC sampling value must stay below max_fs"


def tbd_test_TFFT_lte_TC():
    try:
        _ = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=1e4,
                                    max_fs=1e5,
                                    max_adc_buffer_size=2048*4, n_adc=2048*3,
                                    debug=True))
    except ValueError as ex:
        str_ex = str(ex)
    assert str_ex == ERR_TFFT_lte_TC


def tbd_test_Nyquist():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=False)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(100, 0, 0, xt=lambda t: 2*t+100)
    scatterers = [scatterer1, scatterer2]

    str_ex = ""
    try:
        _ = rt_points(radar, scatterers, debug=True)
    except ValueError as ex:
        str_ex = str(ex)

    exception_start = "Nyquist will always prevail: "
    len_str = len(exception_start)
    assert str_ex[:len_str] == exception_start


def tbd_test_FMCW_ADC_buffer_size():
    str_ex = ""
    try:
        _ = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=1e6, max_adc_buffer_size=2048),
                  debug=False)
    except ValueError as ex:
        str_ex = str(ex)

    assert str_ex == "ADC buffer overflow"


def tbd_test_FMCW_range_chirp_N():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=False)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, debug=False)
    str_ex = ""

    try:
        Distances, range_profile = rsp.range_fft(bb, chirp_index=1)
    except ValueError as ex:
        str_ex = str(ex)
    assert str_ex == "chirp index value not supported yet"


def tbd_test_FMCW_cfar_names_ok():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=False)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_1d(cfar_type="CA", FT=range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(scatterer_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_scatterers = [Scatterer(Distances[i]) for i in grouped_peaks]
    error = rsp.error([scatterer2, scatterer1], found_scatterers)
    assert error < 3


def tbd_test_FMCW_cfar_names_nok():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=False)

    scatterer1 = Scatterer(5.1)
    scatterer2 = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    scatterers = [scatterer1, scatterer2]

    bb = rt_points(radar, scatterers, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    str_ex = ""
    cfar_type = "OG"
    try:
        _ = rsp.cfar_1d(cfar_type=cfar_type, FT=range_profile)
    except ValueError as ex:
        str_ex = str(ex)
    assert str_ex == f"Unsupported CFAR type: {cfar_type}"


def tbd_test_if2d():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=False)
    f2d = rsp.if2d(radar)
    assert f2d == 0.02142857142857143


def tbd_test_scatterer_def():
    """ Ensures code for scatterer default values logic remains
    constant in time"""
    _ = Scatterer(10, 0, 0, xt=lambda t: 2*t+10)
    try:
        _ = Scatterer(20, xt=lambda t: 2*t+10)
    except Exception as ex:
        assert str(ex) == ERR_TARGET_T0

    _ = Scatterer(0, 10, 0, yt=lambda t: 2*t+10)
    try:
        _ = Scatterer(0, 20, 0, yt=lambda t: 2*t+10)
    except Exception as ex:
        assert str(ex) == ERR_TARGET_T0

    _ = Scatterer(0, 0, 10, zt=lambda t: 2*t+10)
    try:
        _ = Scatterer(0, 0, 20, zt=lambda t: 2*t+10)
    except Exception as ex:
        assert str(ex) == ERR_TARGET_T0

