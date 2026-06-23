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

c = 3e8

scatterer1 = Scatterer(5.1)

min_error = scatterer1.distance()
config = {"bw": "?", "adc_sample_rate": "?", "error": "?"}

bws = arange(30, 40)*1e8
slopes = arange(1, 10)*1e12
adc_sample_rates = arange(1, 100)*1e6

print(len(bws))

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
                                         adc_sample_count=32)

                    radar = Radar(transmitter=transmitter,
                                  receiver=receiver0)

                    bb = rt_points([radar], [scatterer1],
                                   radar)
                    Distances, range_profile = rsp.range_fft(bb["adc_cube"][0,0,0,:], bb)
                    Distances = Distances/2
                    ca_cfar = rsp.cfar_ca_1d(range_profile)
                    mag_r = np_abs(range_profile)
                    mag_c = np_abs(ca_cfar)

                    # little hack to remove small FFT ripples : mag_r> 5
                    scatterer_filter = ((mag_r > mag_c) & (mag_r > 5))

                    index_peaks = where(scatterer_filter)[0]
                    grouped_peaks = rsp.peak_grouping_1d(index_peaks, mag_r)

                    found_scatterers = [Scatterer(Distances[i])
                                        for i in grouped_peaks]
                    error = rsp.error([scatterer1], found_scatterers)
                    # print("error", error)
                    if error < min_error:
                        min_error = error
                        config = {"bw": bw, "adc_sample_rate": adc_sample_rate,
                                  "slope": slope,
                                  "error": error}
                except Exception as ex:
                    pass
                    # print(str(ex))
                    # raise

# yields 0 error for bw=3e9, adc_sample_rate=100, slope=6e8
print(f"optimal config: {config}, yields error: {min_error}")
