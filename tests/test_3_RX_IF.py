""" This is mostly to test the RX_freq and IF signal
"""

from os.path import abspath, join, pardir
import sys
from test_assets import radar_tdm_1_chirp_8_adc

from numpy import allclose, array

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

def test_RX_frequency():
    yif = radar_tdm_1_chirp_8_adc.BB_IF(Tc=array([0, 3e-7, 6.6e-7, 1.e-6]),
                                        f_rx=array([6e10, 6.0003e10, 6.0006e+10, 6.001e10]),
                                        f_tx=array([6.1e10, 6.1003e10, 6.1006e10, 6.101e10]),
                                        rx_hpf=1e3, rx_lpf=1e8)
    assert allclose(yif, array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]), atol=1e-8)

def test_RX_freq_2():
    yif = radar_tdm_1_chirp_8_adc.BB_IF(Tc=array([0, 1.3e-7, 2.6e-7, 4e-7,
                                                    5.3e-7, 6.6e-7, 8e-7, 9.3e-7,
                                                    1e-6, 1.2e-6, 1.3e-6, 1.46e-6,
                                                    1.6e-6, 1.73e-6, 1.86e-6, 2e-6]),
                                        f_rx=array([6.001e9, 6.007e9,6.014e9,6.021e9,
                                            6.027e9,6.034e9,6.041e9,6.047e9,
                                            6.054e9,6.061e9,6.067e9,6.074e9,
                                            6.081e9,6.087e9,6.094e9,6.101e9]),
                                        f_tx=array([6e9,6.006e9,6.013e9,6.02e9,
                                            6.026e9,6.033e9,6.04e9,6.046e9,
                                            6.053e9,6.06e9,6.066e9,6.073e9,
                                            6.08e9,6.086e9,6.093e9,6.1e9]),
                                        rx_hpf=1e3, rx_lpf=1e8)
    assert allclose(yif, array([ 1.        +0.00000000e+00j,  0.68454711+7.28968627e-01j,
                                -0.06279052+9.98026728e-01j, -0.80901699+5.87785252e-01j,
                                -0.98228725-1.87381315e-01j, -0.53582679-8.44327926e-01j,
                                0.30901699-9.51056516e-01j,  0.90482705-4.25779292e-01j,
                                1.        -1.13310778e-15j,  0.30901699+9.51056516e-01j,
                                -0.30901699+9.51056516e-01j, -0.96858316+2.48689887e-01j,
                                -0.80901699-5.87785252e-01j, -0.12533323-9.92114701e-01j,
                                0.63742399-7.70513243e-01j,  1.        -2.26621556e-15j]), atol=1e-8)