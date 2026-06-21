""" testing the RX side of the system.
combination of TX_Freq and ToF
f_rx = receiver_radar.TX_freq(adc_times-time_of_flight)
v0.0.11: 1 passed
"""

import numpy as np
from test_assets import radar_tdm_1_chirp_8_adc, adc_sampling_times_8_samples


def test_0_tof():
    # test that f_tx == f_tx_expected
    # and computing f_tx_expected explicitely
    radar = radar_tdm_1_chirp_8_adc
    chirp_start_freq = radar.transmitter.chirp_start_freq
    timestamps = adc_sampling_times_8_samples
    timestamps = timestamps[:, None, None, None]
    f_tx_expected = np.array([chirp_start_freq +
                              radar.transmitter.slope*t for t in timestamps])
    f_tx = radar.TX_freq(timestamps=timestamps)
    assert f_tx_expected.shape == f_tx.shape
    assert np.allclose(f_tx_expected, f_tx, atol=1e-3)
