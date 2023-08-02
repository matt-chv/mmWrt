from os.path import abspath, join, pardir
import sys

from numpy import where, sin, pi
from numpy import complex_ as complex
from numpy import float32  # alternatives: float16, float64

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402


def test_FMCW_vibration():
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=3e3, max_adc_buffer_size=2048,
                                    t_inter_chirp=1.2e-6,
                                    chirps_count=2,
                                    t_inter_frame=0,
                                    frames_count=10),
                  debug=True)

    target1 = Target(5.1)
    freq_vibration=5e2
    target2 = Target(10, 0, 0, vx=lambda t: 0.01 * sin(2 * pi * freq_vibration * t))
    targets = [target1, target2]

    bb = rt_points(radar, targets,
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
    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

    found_targets = [Target(Distances[i]) for i in grouped_peaks]
    error = rsp.error(targets, found_targets)
    assert error < 3

