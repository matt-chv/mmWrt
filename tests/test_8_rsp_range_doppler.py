""" testing RSP.range_doppler
Covers:
 - TDM mode
(not written DDM, SFMCW)
v0.0.11: 2
"""
import logging
from os.path import abspath, join, pardir
import numpy as np
import sys

from numpy import sin, pi, zeros
from numpy.fft import fftshift, fft, fft2
from numpy import complex128 as complex
from scipy.signal import find_peaks

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.RadarSignalProcessing import range_doppler
# from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402
from test_assets import radar_dmax_25m_vmax_2mps, target_static_5p1m, \
    target_linear_speed_10p1m_1mps, radar_vibrate, target_vibrate


def test_rsp_range_doppler():
    radar = radar_dmax_25m_vmax_2mps
    targets = [target_static_5p1m, target_linear_speed_10p1m_1mps]

    # logging.basicConfig(level=logging.WARNING,
    #                    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    # logging.getLogger("Transmitter").setLevel(logging.WARNING)
    # logging.getLogger("mmWrt.RadarSignalProcessing._cfar_ca").setLevel(logging.DEBUG)

    bb = rt_points([radar], targets, radar)
    rd = range_doppler(bb["adc_cube"][0,:,0,:],
                       adc_sample_rate=radar.adc_sample_rate,
                       chirp_slope=radar.chirp_slope,
                       wavelength=3e8/radar.chirp_start_freq,
                       chirp_period=radar.t_inter_chirp)
    print("detections", rd)
    assert rd.shape == (2,2)
    assert np.allclose(rd, np.array([[ 4.734375, 0.],
                                     [10.2578125, 0.958125]]))


def test_FMCW_vibration():
    """ validates the uDoppler logic
    """
    from test_assets import NF_vibrate, NA_vibrate, NC_vibrate, F1_vibrate
    NC = NC_vibrate
    NF = NF_vibrate

    # logging.getLogger("mmWrt.Raytracing.sample_all_rays").setLevel(logging.DEBUG)
    radar = radar_vibrate
    target = target_vibrate
    fn = "adc_cube_vibrate.npy"
    fp = abspath(join(__file__, pardir, fn))

    run = False
    if run:
        bb = rt_points([radar], [target], radar)
        adc_cube = bb["adc_cube"]
        np.save(fp, adc_cube)
    else:
        adc_cube = np.load(fp)

    udops1 = zeros((NC, NF))
    dops = []
    for frame_idx in range(NF):
        r_fft = np.fft.fft(adc_cube[frame_idx, :, 0, :], axis=1)
        try:
            r_peak = find_peaks(np.abs(r_fft[0,:]), height=12)[0][0]
        except Exception as ex:
            print(f"r_peak - ex: {str(ex)} -- fidx: {frame_idx}")
            raise ValueError("failed to find peak")
        rd_fft = np.fft.fftshift(np.fft.fft(r_fft, axis=0), axes=0)
        try:
            d_peak = find_peaks(np.abs(rd_fft[:, r_peak]), height=300)[0][0]
        except Exception as ex:

            if np.abs(np.abs(rd_fft[0, r_peak])) >=300:
                d_peak = 0
            elif np.abs(np.abs(rd_fft[31, r_peak])) >=300:
                d_peak = 31
            else:
                print(f"d_peak - ex: {str(ex)} -- fidx: {frame_idx}")
                raise ValueError("failed to find peak")
        udops1[:, frame_idx] = np.abs(rd_fft[:, r_peak])
        dops.append(d_peak)
    dops = np.array(dops)

    Y = fft(dops - np.mean(dops))
    ym = find_peaks(np.abs(Y))[0][0]
    assert ym == F1_vibrate
