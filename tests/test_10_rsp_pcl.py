""" testing RSP.pcl
Covers:
 - TDM mode
(not written DDM, SFMCW)
v0.0.11: 0
"""
import logging  # noqa: F401
import numpy as np
from os.path import abspath, join, pardir
from test_assets import radar_tdm_32loop_16T16R_64adc, scatterer_static_5p1m, scatterer_static_10p1m, scatterer_static_z_15p1m, fif_tdm0_15m
from mmWrt.Raytracing import rt_points
from mmWrt.RadarSignalProcessing import pcl


def test_pcl():
    radar = radar_tdm_32loop_16T16R_64adc
    run = False
    scatterers = [scatterer_static_5p1m,
                        scatterer_static_10p1m,
                        scatterer_static_z_15p1m]
    if run:
        import time

        start = time.time()     
        bb = rt_points([radar],
                       [scatterer_static_5p1m,
                        scatterer_static_10p1m,
                        scatterer_static_z_15p1m],
                       radar)
        end = time.time()
        print("total time", end-start) #10 s
        np.save("adc_cube_pcl.npy", adc_cube)
        #-> TX * RX = 16x16
        # loops = 64 
        # ADC: 64

        adc_cube = bb["adc_cube"]
    # NumPy fills in row-major (C) order by default,
    # so the first axis gets split
    # (0..63,:) -> (0, 0..63, :)
    # (1*64..1*64+63) -> (1, 0..63, :)
    # (2*64..2*64+63) -> (2, 0..63, :)
    # the first 64 rows become [0, :, :], the next 64 become [1, :, :], etc.

    fn = "adc_cube_pcl.npy"
    fp = abspath(join(__file__, pardir, fn))
    adc_cube = np.load(fp)
    print("adc_cube.shape", adc_cube.shape) # 1,1024,16,64
    print("fif_tdm0_15m", fif_tdm0_15m)
    print("fs", radar.adc_sample_rate)

    assert radar.adc_sample_count == 64
    assert radar.chirp_count == 32*16
    assert len(radar.tx_antennas) == 16
    assert adc_cube.shape == (1, 512,16, 64)


    from mmWrt.RadarSignalProcessing import ranges_dft_cfar, range_doppler, detection_xy, range_aoa

    # adc values are (frame_count, tx_count * chirp_per_tx, rx_count, adc_count)
    # reshape to (frame_count, chirp_per_tx, virtual_antennas_z, virtual_antennas_x, adc_count)
    fctra_cube = adc_cube.reshape(1, 32, 16, 16, 64)

    detection_range = ranges_dft_cfar(adc_cube[0,0,0,:],
                                      radar.adc_sample_rate,
                                      radar.chirp_slope, pfa=0.01)
    expected_range = np.array([ 4.9546875, 9.909375, 14.8640625])
    assert np.allclose(detection_range, expected_range)

    detections = range_doppler(adc_cube[0,::16,0,:],
                               radar.adc_sample_rate,
                               radar.chirp_slope,
                               wavelength=3e8/radar.chirp_start_freq,
                               chirp_period=radar.chirp_period)
    expected_rd = np.array([[ 4.9546875,0.       ],
                            [ 9.909375, 0.       ],
                            [14.8640625, 0.       ]])
    assert np.allclose(detections, expected_rd)
    
    # print("XZ", detection_list_xz)

    detections_xyz = pcl(fctra_cube, radar)  # np.array(detections_xyz)
    expected_xyz = np.array([[4.9546875, 0, 0],
                             [0., 9.909375, 0.],
                             [0., 0., 14.8640625]])

    assert np.allclose(detections_xyz, expected_xyz)
