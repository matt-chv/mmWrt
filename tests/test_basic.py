from os.path import abspath, join, pardir
import sys

from numpy import where
from numpy import complex_ as complex

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402


def test_FMCW_real():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    range_profile = range_profile
    ca_cfar = ca_cfar

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3


def test_FMCW_real_error():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    range_profile = range_profile
    ca_cfar = ca_cfar

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error([target2, target1], found_targets)
    assert error < 3


def test_FMCW_no_targets_found_error():
    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
    error = rsp.error([target2, target1], [])
    assert error == 18.1


def test_FMCW_ADC_fs_vs_fs_max():
    str_ex = ""
    try:
        _ = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=1e6,
                                    max_fs=1e5,
                                    max_adc_buffer_size=2048, debug=True))
    except ValueError as ex:
        str_ex = str(ex)
    assert str_ex == "ADC sampling value must stay below max_fs"


def test_Nyquist():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(100, 0, 0, vx=lambda t: 2*t+3)
    targets = [target1, target2]

    str_ex = ""
    try:
        _ = rt_points(radar, targets, debug=True)
    except ValueError as ex:
        str_ex = str(ex)

    assert str_ex == "Nyquist will always prevail"


def test_FMCW_ADC_buffer_size():
    str_ex = ""
    try:
        _ = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=1e6, max_adc_buffer_size=2048),
                  debug=True)
    except ValueError as ex:
        str_ex = str(ex)

    assert str_ex == "ADC buffer overflow"


def test_FMCW_range_chirp_N():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
    targets = [target1, target2]

    bb = rt_points(radar, targets, debug=False)
    str_ex = ""

    try:
        Distances, range_profile = rsp.range_fft(bb, chirp_index=1)
    except ValueError as ex:
        str_ex = str(ex)
    assert str_ex == "chirp index value not supported yet"


def test_FMCW_cfar_names_ok():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
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
    grouped_peaks = rsp.peak_grouping_1d(index_peaks)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error([target2, target1], found_targets)
    assert error < 3


def test_FMCW_cfar_names_nok():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
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


def test_FMCW_1j():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048),
                  debug=True)

    target1 = Target(5.1)
    target2 = Target(10, 0, 0, vx=lambda t: 2*t+3)
    targets = [target1, target2]
    bb = rt_points(radar, targets,
                   datatype=complex,
                   debug=False)
    # data_matrix = bb['adc_cube'][0][0][0]
    Distances, range_profile = rsp.range_fft(bb)
    ca_cfar = rsp.cfar_ca_1d(range_profile)

    range_profile = range_profile
    ca_cfar = ca_cfar

    mag_r = abs(range_profile)
    mag_c = abs(ca_cfar)
    # little hack to remove small FFT ripples : mag_r> 5
    target_filter = ((mag_r > mag_c) & (mag_r > 5))

    index_peaks = where(target_filter)[0]
    grouped_peaks = rsp.peak_grouping_1d(index_peaks)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3
