"""
"""
import numpy as np
from test_assets import radar_dmax_25m_vmax_2mps, scatterer_static_5p1m
from mmWrt.Raytracing import rt_points

def test_interference_mitigation():
    radar = radar_dmax_25m_vmax_2mps
    import copy
    interferer = copy.deepcopy(radar_dmax_25m_vmax_2mps)
    interferer.transmitter.chirp_start_freq = 59.5e9
    interferer.transmitter.chirp_slope *= 14.8

    bb = rt_points([radar],
                    [scatterer_static_5p1m],
                    radar)["adc_cube"]

    adc_values_interfered = rt_points([radar, interferer],
                                    [scatterer_static_5p1m],
                                    radar)["adc_cube"][0,0,0,:]
    interference_idx = np.flatnonzero((adc_values_interfered > 1) | (adc_values_interfered < -1))
    # Use broadcasting to subtract 1, keep the original, and add 1
    extended = interference_idx[:, None] + np.array([-1, 0, 1])

    # Flatten and get unique, sorted values
    interference_idx = np.unique(extended)
    adc_interferences_zeroed = adc_values_interfered.copy()
    adc_interferences_zeroed[interference_idx] = 0

    # print(adc_values_interfered[0,0,0,:])
    import matplotlib.pyplot as plt
    plt.plot(adc_values_interfered, 'r-')
    plt.plot(adc_interferences_zeroed, 'g-')
    plt.show()

test_interference_mitigation()