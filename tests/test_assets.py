from numpy import array, arange, exp, pi
from os.path import abspath, join, pardir
import sys

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Scene import Antenna, Radar, Transmitter, TransmitterDDM, Receiver, Target  # noqa E402

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"

# speed of light in void
c = 3e8
n_air = 1  # really is 1.0003 but simplified for testing
v_light = c/n_air

f0_60G = 60e9
f0_77G = 77e9
lambda_60G = v_light/f0_60G
lambda_77G = v_light/f0_77G
chirp_slope_tdm0 = 5e12
chirp_slope_5e12 = chirp_slope_tdm0

adc_samples_count_8 = 8
adc_samples_count_64 = 64
adc_samples_count_1024 = 1024
d_0m = 0.01
d_5p1m = 5.1
d_5p05m = 5.05
d_10p1m = 10.1
d_15p1m = 15.1
d_th = 100
v_1mps = 1
vmax_2mps = 2.19*v_1mps #MCV was 2
vmax_3mps = 3
dphase_dt_1mps = 4*pi*v_1mps/lambda_60G
# vmax = lambda_0/2/time_inter_chirp
# => t_inter_chirp_vmax_2mps = lambda_60G/2/vmax
t_inter_chirp_vmax_2mps = lambda_60G/4/vmax_2mps
# t_inter_chirp_vmax_3mps = \
# 0.0004166666666666667 ~ 4.16e-4 = 461us
t_inter_chirp_vmax_3mps = lambda_60G/4/vmax_3mps  # 461us
tof_5p1m = d_5p1m/3e8
tof_10p1m = d_10p1m/3e8
fif00 = 2*chirp_slope_tdm0*d_5p1m/3e8  # 170 kHz
fif01 = 2*chirp_slope_tdm0*d_10p1m/3e8
t_inter_frame_50ms = 50e-3


# for DDM, means +pi/2 at every chirp
phase_slope_half_pi = 0.5

adc_sampling_frequency_0 = 3*fif01
chirp_end_time_8adc = adc_samples_count_8*1/adc_sampling_frequency_0*1.5
chirp_end_time_64adc = adc_samples_count_64*1/adc_sampling_frequency_0*1.5
chirp_end_time_1024adc = adc_samples_count_1024*1/adc_sampling_frequency_0*1.5

adc_sampling_times_8_samples = arange(0, 8/adc_sampling_frequency_0, 1/adc_sampling_frequency_0)
adc_sampling_times_64_samples = arange(0, 64/adc_sampling_frequency_0, 1/adc_sampling_frequency_0)
adc_sampling_times_1024_samples = arange(0, 1024/adc_sampling_frequency_0, 1/adc_sampling_frequency_0)

adc_8_values_complex_fif00 = array([ 0.        +0.j,0.52230123+0.85276106j,
                                     -0.48644699+0.87371009j, -0.99998642+0.00521191j,
                                     -0.49552783-0.86859206j,  0.51338395-0.85815903j,
                                      0.9996648 +0.02589005j,  0.46827505+0.88358275j])
adc_8_values_complex_fif00 = array([ 0.        +0.j        , 0.5067312 +0.86210411j,
                                    -0.5022311 +0.86473344j,-0.9999162 -0.01294611j,
                                    -0.47967476-0.87744636j, 0.52888126-0.84869583j,
                                     0.9990299 +0.0440371j , 0.45215427+0.89193975j])
#adc_8_values_complex_fif00 = exp(2*1j*pi*fif00*adc_sampling_times_8_samples)
#adc_8_values_complex_fif00[0] = 0
adc_8_values_complex_fif01 = exp(2*1j*pi*fif01*adc_sampling_times_8_samples)
adc_8_values_complex_fif01[0] = 0

antennas_ULA_64_60G = [Antenna(x=lambda_60G/2*i) for i in range(64)]
antennas_ULA_x_64_60G = [Antenna(x=lambda_60G/2*i) for i in range(64)]
antennas_ULA_y_64_60G = [Antenna(y=lambda_60G/2*i) for i in range(64)]
antennas_ULA_z_64_60G = [Antenna(z=lambda_60G/2*i) for i in range(64)]

antenna_origin_static = Antenna()

transmitter_off = Transmitter(chirp_start_freq=0,
                              chirp_slope=0,
                              chirp_end_time=0)
transmitter_cw_60G = Transmitter(chirp_start_freq=f0_60G,
                              chirp_slope=0,
                              chirp_end_time=10)
tdm_1chirp_8adc = Transmitter(chirp_end_time=chirp_end_time_8adc,
                              chirp_slope=chirp_slope_tdm0)
tdm_2chirp_8adc = Transmitter(chirp_start_freq=f0_60G,
                              chirp_end_time=chirp_end_time_8adc,
                              chirps_count=2,
                              t_inter_chirp=t_inter_chirp_vmax_3mps,
                              chirp_slope=chirp_slope_tdm0)
tdm_2frames_2chirp_8adc = Transmitter(chirp_start_freq=f0_60G,
                                      chirp_end_time=chirp_end_time_8adc,
                                      chirps_count=2,
                                      t_inter_chirp=t_inter_chirp_vmax_3mps,
                                      chirp_slope=chirp_slope_tdm0,
                                      t_inter_frame=t_inter_frame_50ms,
                                      frames_count=2)
tdm_2chirp_8adc = Transmitter(chirp_start_freq=f0_60G,
                              chirp_end_time=chirp_end_time_8adc,
                              chirps_count=2,
                              t_inter_chirp=t_inter_chirp_vmax_3mps,
                              chirp_slope=chirp_slope_tdm0)
tdm_1chirp_1024adc = Transmitter(chirp_end_time=chirp_end_time_1024adc,
                                 chirp_slope=chirp_slope_tdm0)
tdm_2chirp_1024adc = Transmitter(chirp_end_time=chirp_end_time_1024adc,
                                 chirps_count=2,
                                 t_inter_chirp=t_inter_chirp_vmax_3mps,
                                 chirp_slope=chirp_slope_tdm0)

tdm_1chirp_64adc = Transmitter(chirp_start_freq=f0_60G,
                                  chirp_slope=chirp_slope_tdm0,
                                  chirp_end_time=chirp_end_time_64adc,
                                  chirps_count=64,
                                  t_inter_chirp=t_inter_chirp_vmax_3mps)

tdm_1chirp_1024adc_ula_z_64 = Transmitter(chirp_start_freq=f0_60G,
                                  chirp_slope=chirp_slope_tdm0,
                                  chirp_end_time=chirp_end_time_1024adc,
                                  antennas=antennas_ULA_z_64_60G,
                                  chirps_count=1,
                                  t_inter_chirp=t_inter_chirp_vmax_3mps)

tdm_vmax_2mps = Transmitter(chirp_start_freq=f0_60G,
                            chirp_end_time=t_inter_chirp_vmax_2mps/3,
                            t_inter_chirp=t_inter_chirp_vmax_2mps,
                            chirps_count=32)

ddm_4chirps_0_half_pi = TransmitterDDM(chirp_start_freq=60e9,
                                       chirp_slope=chirp_slope_tdm0,
                                       chirp_end_time=chirp_end_time_8adc,
                                       t_inter_chirp=1.1*chirp_end_time_8adc,
                                       antennas=[Antenna() for _ in range(2)],
                                       chirps_count=4,
                                       conf={"TX_phaser_slopes": [0, phase_slope_half_pi]})

"""slope=1e12
d = 150 #m
fif = 1e6 = 64 idx
nadc = 256
fs = 
tdm_1e12_77_"""


receiver0 = Receiver(adc_sample_rate=adc_sampling_frequency_0,
                     adc_samples_per_chirp=adc_samples_count_8)

receiver1 = Receiver(adc_sample_rate=adc_sampling_frequency_0,
                     max_adc_buffer_size=1025,
                     adc_samples_per_chirp=adc_samples_count_1024)

receiver_fs0_64adc = Receiver(adc_sample_rate=adc_sampling_frequency_0,
                              adc_samples_per_chirp=adc_samples_count_64)

receiver_dmax_25m = Receiver(adc_sample_rate=adc_sampling_frequency_0/3,
                             adc_samples_per_chirp=adc_samples_count_64)

receiver_dmax_50m = Receiver(adc_sample_rate=adc_sampling_frequency_0/3,
                             max_adc_buffer_size=1025,
                             adc_samples_per_chirp=64)

receiver_dmax_100m = Receiver(adc_sample_rate=adc_sampling_frequency_0*4.1/3,  #MCV: was 1/3
                             max_adc_buffer_size=1025,
                             adc_samples_per_chirp=64)

receiver_ULA_64 = Receiver(adc_sample_rate=adc_sampling_frequency_0,
                           antennas=antennas_ULA_x_64_60G,
                           adc_samples_per_chirp=adc_samples_count_64)

radar_tdm_1_chirp_8_adc = Radar(transmitter=tdm_1chirp_8adc,
                                receiver=receiver0)
radar_tdm_1_chirp_64_adc = Radar(transmitter=tdm_1chirp_64adc,
                                 receiver=receiver_fs0_64adc)
radar_tdm_1_chirp_1024_adc = Radar(transmitter=tdm_1chirp_1024adc,
                                   receiver=receiver1)
radar_tdm_2_chirp_8adc = Radar(transmitter=tdm_2chirp_8adc,
                               receiver=receiver0)
radar_tdm_2_chirp_1024adc = Radar(transmitter=tdm_2chirp_1024adc,
                                  receiver=receiver0)
radar_tdm_2_frames_2_chirps_8adc = Radar(transmitter=tdm_2frames_2chirp_8adc,
                                         receiver=receiver0)
radar_dmax_25m_vmax_2mps = Radar(transmitter=tdm_vmax_2mps,
                                 receiver=receiver_dmax_25m)
radar_tx_off = Radar(transmitter=transmitter_off,
                     receiver=receiver0,
                     debug=True)
radar_tx_cw = Radar(transmitter=transmitter_cw_60G,
                     receiver=receiver0,
                     debug=True)
radar_ula_64_RX = Radar(transmitter=tdm_1chirp_1024adc,
                        receiver=receiver_ULA_64)
radar_ula_64_TX = Radar(transmitter=tdm_1chirp_1024adc_ula_z_64,
                        receiver=receiver_dmax_100m)

target_static_0 = Target(xt=lambda t: d_0m+0*t)
target_static_5p1m = Target(xt=lambda t: d_5p1m+0*t)
target_static_10p1m = Target(yt=lambda t: d_10p1m+0*t)
target_static_z_15p1m = Target(zt=lambda t: 15.1+0*t)  #d_15p1m+0*t)
target_linear_speed_5p1m_1mps = Target(xt=lambda t: d_5p1m + v_1mps*t)
target_linear_speed_10p1m_1mps = Target(xt=lambda t: d_10p1m + v_1mps*t)

target_5p1m_radar1_bin_1 = 1
target_10p1m_radar1_bin_3 = 3

targets = [target_static_5p1m, target_static_10p1m,
           target_linear_speed_5p1m_1mps, target_linear_speed_10p1m_1mps,
           target_static_0]
distances = [d_5p1m, d_10p1m, d_5p05m, d_0m, d_0m]
