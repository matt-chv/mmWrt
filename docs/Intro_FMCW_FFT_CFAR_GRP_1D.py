# FMCW-FFT -> CFAR -> GRP
# This is a simple code for mmWrt for CFAR

# Needed if running from project folder
# should be commented if importing the mmWrt module from pip
from os.path import abspath, join, pardir
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from numpy import where, expand_dims

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402
from mmWrt import RadarSignalProcessing as rsp  # noqa: E402

c = 3e8
fp_fft_cfar_2D = abspath(join(__file__, pardir, "FFT_2D.png"))
fp_fft_1D = abspath(join(__file__, pardir, "FFT_1D.png"))

debug_ON = True
test = 0
radar = Radar(transmitter=Transmitter(bw=1e9, slope=70e8),
              receiver=Receiver(fs=1e3, max_adc_buffer_size=256,
                                debug=debug_ON), debug=debug_ON)

target1 = Target(1.5)
target2 = Target(2, 0, 0, vx=lambda t: 2*t+3)

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

found_targets = [Target(Distances[i]) for i in grouped_peaks]
error = rsp.error([target1, target2], found_targets)
print("targets", [t.pos() for t in found_targets])
print("error is", error)

# 2D representation of the FFT and CFAR
# plot on X,Y axis the FFT and CFAR
plt.plot(mag_r)
plt.plot(mag_c)
plt.title("2D plots FFT w/ CFAR")
plt.savefig(fp_fft_cfar_2D)
plt.clf()

# 1D representation of the FFT
# Select the color map named CMRmap_r
cmap = cm.get_cmap(name='CMRmap_r')
# convert the 1D array in 2D array to plot using imshow
mag_r = expand_dims(mag_r, axis=0)
# set aspect ratio to auto to have high enough pixels to see them in the y_axis
# change the norm to have a log color scale
# to better see the peaks in correlation with 2D FFT plot
plt.imshow(mag_r, cmap,
           aspect='auto',
           norm=colors.LogNorm(vmin=min(mag_r[0][:]), vmax=max(mag_r[0][:])))
plt.title("1D FFT")
plt.savefig(fp_fft_1D)
