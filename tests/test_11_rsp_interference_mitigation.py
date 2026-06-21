""" testing interfereence mitigation
Covers:
 - TDM mode
(not written DDM, SFMCW)
v0.0.11: 1
"""
import copy
import numpy as np
from test_assets import radar_dmax_25m_vmax_2mps, scatterer_static_5p1m
from mmWrt.Raytracing import rt_points


def test_interference_mitigation():
    radar = radar_dmax_25m_vmax_2mps

    interferer = copy.deepcopy(radar_dmax_25m_vmax_2mps)
    interferer.transmitter.chirp_start_freq = 59.5e9
    interferer.transmitter.chirp_slope *= 14.8

    adc_reference = rt_points([radar],
                              [scatterer_static_5p1m],
                              radar)["adc_cube"][0, 0, 0, :]
    fft_reference = np.fft.fft(adc_reference)
    fft_mag = np.abs(fft_reference)[:adc_reference.shape[0]//2]
    tone_peak = np.argmax(fft_mag)
    energy_out_peak = np.delete(fft_mag,
                                [tone_peak-1, tone_peak, tone_peak+1]).sum()

    # 2 inteferences
    adc_values_interfered = rt_points([radar, interferer],
                                      [scatterer_static_5p1m],
                                      radar)["adc_cube"][0, 0, 0, :]

    fft_mag_interfered = np.abs(np.fft.fft(adc_values_interfered)
                                [:adc_reference.shape[0]//2])
    energy_interference = np.delete(fft_mag_interfered, [tone_peak-1,
                                                         tone_peak,
                                                         tone_peak+1]).sum()

    interference_idx = np.flatnonzero((adc_values_interfered > 1)
                                      | (adc_values_interfered < -1))
    # Use broadcasting to subtract 1, keep the original, and add 1
    extended = interference_idx[:, None] + np.array([-1, 0, 1])
    # Flatten and get unique, sorted values
    interference_idx = np.unique(extended)
    interference_range = np.arange(min(interference_idx),
                                   max(interference_idx), 1)

    # first interference mitigation zero values
    adc_interferences_zeroed = adc_values_interfered.copy()
    adc_interferences_zeroed[interference_range] = 0
    fft_mag_zerod = np.abs(np.fft.fft(adc_interferences_zeroed))
    energy_zeroed = np.delete(fft_mag_zerod, [tone_peak-1,
                                              tone_peak, tone_peak+1]).sum()

    # 2. reconstruct with tone
    adc_interfered_reconstructed = adc_values_interfered.copy()
    # start with the top tone, maybe improve with list of top tones?
    top_tone_in_zeroed = np.argmax(fft_mag_zerod[:adc_reference.shape[0]//2])

    ideal_fft = np.zeros(adc_reference.shape[0]//2, dtype=complex)
    ideal_fft[top_tone_in_zeroed-1:top_tone_in_zeroed+2] = \
        np.fft.fft(adc_interferences_zeroed)[tone_peak-1: tone_peak+2]
    adc_interfered_reconstructed[interference_range] = \
        np.fft.irfft(ideal_fft)[interference_range]

    fft_mag_reconstructed = np.abs(np.fft.fft(adc_interfered_reconstructed)
                                   [:adc_reference.shape[0]//2])
    energy_reconstructed = np.delete(fft_mag_reconstructed,
                                     [tone_peak-1,
                                      tone_peak,
                                      tone_peak+1]).sum()

    print("E0, E1, E2, E3", energy_out_peak, energy_interference,
          energy_zeroed, energy_reconstructed)
    assert energy_reconstructed < energy_zeroed
    assert energy_reconstructed < energy_interference
    assert energy_out_peak < energy_reconstructed
