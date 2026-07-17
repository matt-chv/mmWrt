# CFAR
# This is a simple code for mmWrt for CFAR

# Needed if running from project folder
# should be commented if importing the mmWrt module from pip
from os.path import abspath, join, pardir
import sys
from numpy import arange, where, abs as np_abs
from tqdm import tqdm

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Scatterer  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402

import logging
logging.getLogger().setLevel(logging.CRITICAL)

c = 3e8
debug_code = True

scatterer1 = Scatterer(5.1)

min_error = scatterer1.distance()
config = {"bw": "?", "adc_sample_rate": "?", "error": "?"}

bws = arange(30, 40)*1e8
slopes = arange(1, 10)*1e12
adc_sample_rates = arange(1, 100)*1e5  # intentionally small lower bound

with tqdm(total=len(bws) * len(slopes) * len(adc_sample_rates)) as pbar:
    for bw in bws:
        for slope in slopes:
            # slope = slope_m * 1e8
            for adc_sample_rate in adc_sample_rates:
                pbar.update(1)
                try:

                    chirp_end_time = bw/slope
                    transmitter = Transmitter(chirp_end_time=chirp_end_time,
                                              chirp_slope=slope)
                    receiver0 = Receiver(adc_sample_rate=adc_sample_rate,
                                         adc_sample_rate_max=adc_sample_rate+1,
                                         adc_sample_count=32)

                    radar = Radar(transmitter=transmitter,
                                  receiver=receiver0)

                    bb = rt_points([radar], [scatterer1],
                                   radar)
                    Distances, range_profile = rsp.range_fft(bb["adc_cube"]
                                                             [0, 0, 0, :], bb)
                    Distances = Distances/2

                    cfar_ca = rsp.cfar_ca(range_profile, train_cell_count=5,
                                          guard_cell_count=1, pfa=1e-6)

                    mag_r = np_abs(range_profile)
                    mag_c = np_abs(cfar_ca)

                    # little hack to remove small FFT ripples : mag_r> 5
                    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

                    index_peaks = where(scatterer_filter)[0]
                    if index_peaks.size > 0:
                        grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

                        found_scatterers = [Scatterer(Distances[i])
                                            for i in grouped_peaks]
                        error = rsp.error([scatterer1], found_scatterers)
                        # print("error", error)
                        if error < min_error:
                            min_error = error
                            config = {"bw": float(bw),
                                      "adc_sample_rate": float(adc_sample_rate),
                                      "slope": float(slope),
                                      "error": float(error)}
                except Exception as ex:
                    if debug_code:
                        print(str(ex))
                        raise

# yields 0 error for bw=3e9, adc_sample_rate=100, slope=6e8
print(f"optimal config: {config}, yields error: {min_error}")
