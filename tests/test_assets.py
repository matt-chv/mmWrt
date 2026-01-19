from numpy import pi
from os.path import abspath, join, pardir
import sys

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa E402

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"

f0_60G = 60e9
lambda_60G = 3e8/f0_60G
chirp_slope_tdm0 = 5e12

number_adc_samples_8 = 8
number_adc_samples_1024 = 1024
d_5p1m = 5.1
d1 = 10.1
v_1mps = 1
vmax = 2*v_1mps
dphase_dt_1mps = 4*pi*v_1mps/lambda_60G
# vmax = lambda_0/2/time_inter_chirp
# => t_inter_chirp_vmax_2mps = lambda_60G/2/vmax
t_inter_chirp_vmax_2mps = lambda_60G/2/vmax
fif00 = 2*chirp_slope_tdm0*d_5p1m/3e8
fif01 = 2*chirp_slope_tdm0*d1/3e8
adc_sampling_frequency_0 = 3*fif01
ramp_end_time_8adc = number_adc_samples_8*1/adc_sampling_frequency_0*1.5
ramp_end_time_1024adc = number_adc_samples_1024*1/adc_sampling_frequency_0*1.5

tdm_1chirp_8adc = Transmitter(ramp_end_time=ramp_end_time_8adc,
                              slope=chirp_slope_tdm0)
tdm_2chirp_8adc = Transmitter(f0_min=f0_60G,
                              ramp_end_time=ramp_end_time_8adc,
                              chirps_count=2,
                              t_inter_chirp=t_inter_chirp_vmax_2mps,
                              slope=chirp_slope_tdm0)
tdm_1chirp_1024adc = Transmitter(ramp_end_time=ramp_end_time_1024adc,
                                 slope=chirp_slope_tdm0)

receiver0 = Receiver(fs=adc_sampling_frequency_0,
                     n_adc=number_adc_samples_8)
receiver1 = Receiver(fs=adc_sampling_frequency_0,
                     max_adc_buffer_size=1025,
                     n_adc=number_adc_samples_1024)

radar_tdm_1_chirp_8_adc = Radar(transmitter=tdm_1chirp_8adc,
                                receiver=receiver0)
radar_tdm_1_chirp_1024_adc = Radar(transmitter=tdm_1chirp_1024adc, receiver=receiver1)

radar_tdm_2_chirp_8adc = Radar(transmitter=tdm_2chirp_8adc,
                               receiver=receiver0)

target_static_5p1m = Target(xt=lambda t: d_5p1m+0*t)
target_static_10p1m = Target(xt=lambda t: d1+0*t)
target_linear_speed_1mps = Target(xt=lambda t: d_5p1m+v_1mps*t)
