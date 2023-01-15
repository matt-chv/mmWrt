# CFAR
# This is a simple code for mmWrt for CFAR

# Needed if running from project folder
# should be commented if importing the mmWrt module from pip
from os.path import abspath, join, pardir
import sys
from numpy import arange, where
from tqdm import tqdm

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402

c = 3e8
fp_fft_cfar_2D = abspath(join(__file__, pardir, "FFT_CFAR_2D.png"))
fp_fft_1D = abspath(join(__file__, pardir, "FFT_1D.png"))

target1 = Target(1.5)
target2 = Target(2, 0, 0, vx=lambda t: 2*t+3)

min_error = target1.distance() + target2.distance()  # we start at 2.5
config = {"bw": "?", "fs": "?", "error": "?"}
# this will take minutes
bws = [0.1e9, 0.2e9, 0.3e9, 0.5e9, 1e9, 2e9, 3e9, 4e9]
slopes = range(1, 100)
fss = arange(100, 25e6, 100)
# this takes seconds to verify that min_error is 0 with those settings
bws = [3e9]
slopes = [6]
fs = range(50, 200)

debug_ON = False

with tqdm(total=len(bws) * len(slopes) * len(fss)) as pbar:
    for bw in bws:
        for slope_m in slopes:
            slope = slope_m * 1e8
            for fs in fss:
                pbar.update(1)
                try:

                    radar = Radar(transmitter=Transmitter(bw=bw, slope=slope),
                                  receiver=Receiver(fs=fs,
                                                    max_adc_buffer_size=512,
                                                    debug=debug_ON),
                                  debug=debug_ON)

                    bb = rt_points(radar, [target1, target2], debug=debug_ON)
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

                    found_targets = [Target(Distances[i])
                                     for i in grouped_peaks]
                    error = rsp.error([target1, target2], found_targets)
                    # print("error", error)
                    if error < min_error:
                        min_error = error
                        config = {"bw": bw, "fs": fs,
                                  "slope": slope,
                                  "error": error}
                except Exception:
                    pass
                    # print(str(ex))
                    # raise

# yields 0 error for bw=3e9, fs=100, slope=6e8
print(f"optimal config: {config}, yields error: {min_error}")
