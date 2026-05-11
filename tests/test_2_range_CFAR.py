""" Tests RANGE FFT + CFAR
Includes:
1. 1 radar 1 chirp TDM: 8 adc samples / 1024 adc samples
2. targets:
    - static at 0, 5.1 and 10.1 meters (adding 0 as np.find_peaks does not work for 1st range bin)
    - linear motion at 1 m/s starting at 5.1 and 10.1 meters
3. CFAR: handled in test_range_cfar.py
- non point targets
- attenuation
"""

from os.path import abspath, join, pardir
import re
from semver import VersionInfo
import sys

from numpy import where
from numpy import complex128 as complex
from numpy import float32  # alternatives: float16, float64

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402
from mmWrt.Scene import ERR_TARGET_T0, ERR_TFFT_lte_TC  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402

from test_1_range_point import __range__wrapper  # noqa: E402


def test_tdm_8adc_target0():
    __range__wrapper(target_idxes=[0], radars_idxes=[0],
                     distance_idxes=[0],
                     cfar_peak_detect=True)

def nok_tdm_8adc_range0m():
    """ Test CFAR on range bin 0 as np.find_peaks does not work on range bin 0, 
    currently also broken with CFAR needs debug
    so we need to ensure CFAR can handle it. """
    __range__wrapper(target_idxes=[4], radars_idxes=[1],
                     distance_idxes=[4],
                     cfar_peak_detect=True)


def tbd_test_FMCW_1j():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    targets = [target1, target2]
    bb = rt_points(radar, targets,
                   datatype=complex,
                   debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3


def tbd_test_FMCW_radar_equation():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    # adding RCS to ensure targets are detected ...
    target1 = Target(5.1, rcs_f=lambda f: 10824)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10, rcs_f=lambda f: 43000)
    targets = [target1, target2]

    bb = rt_points(radar, targets, radar_equation=True, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3


def tbd_test_FMCW_radar_equation_corner_reflector():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    # adding RCS to ensure targets are detected ...
    target1 = Target(5.1, rcs_f=lambda f: 10824,
                     target_type="corner_reflector")
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10, rcs_f=lambda f: 43000)
    targets = [target1, target2]

    bb = rt_points(radar, targets, radar_equation=True, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    try:
        assert error < 3
    except Exception as ex:  # pragma: no cover
        print("found targets", [str(t) for t in found_targets])
        print("expected targets", [str(t) for t in targets])
        raise Exception(str(ex))


def tbd_test_FMCW_real_adc_po2():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  adc_po2=True,
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3


def tbd_test_FMCW_real_error():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3


def tbd_test_FMCW_no_targets_found_error():
    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    error = rsp.error([target2, target1], [])
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

    target1 = Target(5.1)
    target2 = Target(100, 0, 0, xt=lambda t: 2*t+100)
    targets = [target1, target2]

    str_ex = ""
    try:
        _ = rt_points(radar, targets, debug=True)
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

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
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

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_1d(cfar_type="CA", FT=range_profile)

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error([target2, target1], found_targets)
    assert error < 3


def tbd_test_FMCW_cfar_names_nok():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048),
                  debug=False)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, xt=lambda t: 2*t+10)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
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


def tbd_test_target_def():
    """ Ensures code for target default values logic remains
    constant in time"""
    _ = Target(10, 0, 0, xt=lambda t: 2*t+10)
    try:
        _ = Target(20, xt=lambda t: 2*t+10)
    except Exception as ex:
        assert str(ex) == ERR_TARGET_T0

    _ = Target(0, 10, 0, yt=lambda t: 2*t+10)
    try:
        _ = Target(0, 20, 0, yt=lambda t: 2*t+10)
    except Exception as ex:
        assert str(ex) == ERR_TARGET_T0

    _ = Target(0, 0, 10, zt=lambda t: 2*t+10)
    try:
        _ = Target(0, 0, 20, zt=lambda t: 2*t+10)
    except Exception as ex:
        assert str(ex) == ERR_TARGET_T0

