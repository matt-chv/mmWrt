""" This is mostly to test the TX_freq and phases
"""
import copy
import logging
from os.path import abspath, join, pardir
import sys
from time import time, perf_counter
from numpy import allclose, arange, array, linspace, pi, tile, repeat, zeros
import pytest

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

# from mmWrt.Raytracing import BB_IF
from mmWrt.Scene import Radar, Transmitter, TransmitterDDM, Antenna, Receiver, Target  # noqa: E402
from test_assets import ddm_4chirps_0_half_pi, phase_slope_half_pi
from test_assets import tdm_1chirp_8adc



"""number_adc_samples = 8
NA = number_adc_samples
NC = 1
NF = 1
TIF = 1.2e-3
TIC = 1.2e-6
chirp_bw = 0.01e9
chirp_slope = 5e12
t_end = chirp_bw/chirp_slope
phaser1_slope = 0.5
adc_sampling_frequency = 5e5
fs = adc_sampling_frequency"""

"""tdm1 = Transmitter(f0_min=60e9,
                   ramp_end_time=1.2*NA/fs,
                   slope=chirp_slope,
                   t_inter_chirp=TIC,
                   chirps_count=NC,
                   t_inter_frame=TIF,
                   frames_count=NF)

ddm1 = TransmitterDDM(f0_min=60e9,
                      bw=chirp_bw, slope=chirp_slope,
                      t_inter_chirp=TIC,
                      antennas=[Antenna() for _ in range(2)],
                      chirps_count=NC,
                      t_inter_frame=TIF,
                      frames_count=NF,
                      conf={"TX_phaser_slopes": [0, phaser1_slope]})

receiver1 = Receiver(antennas=[Antenna() for _ in range(1)],
                     fs=adc_sampling_frequency,
                     max_fs=110e6,
                     n_adc=number_adc_samples)
target1 = Target(xt=lambda t: 5.1+0*t,
                 rcs_f=1.0)

# radar1 = Radar(transmitter=tdm1, receiver=receiver1)
"""


def test_tx_frequency_ramp_1_tx():
    """ check that the transmitter frequencies using default values match expected ones"""
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    times = linspace(0, ramp_end_time, 4)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freqs(times)

    expected = array([[6.00000000e+10],[6.00198020e+10],
                      [6.00396040e+10], [6.00594059e+10]])
    try:
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def test_tx_frequency_ramp_2_tx_tdm():
    # check that the transmitter frequencies using default values match expected ones
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    radar.antennas = array([Antenna(), Antenna()])
    radar.chirps_count = 1
    radar.t_inter_chirp = ramp_end_time * 2
    times = linspace(0, ramp_end_time, 4)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freqs(times)

    expected = array([[6.00000000e+10, 0],
                      [6.00198020e+10, 0],
                      [6.00396040e+10, 0],
                      [6.00594059e+10, 0]])
    try:
        assert txfs.shape == expected.shape, f"Shape mismatch: got {txfs.shape}, expected {expected.shape}"
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise

def test_tx_frequency_ramp_2_tx_ddm():
    """ check that the transmitter frequencies using default values match expected ones"""
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    radar.antennas = array([Antenna(), Antenna()])
    times = linspace(0, ramp_end_time, 4)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freqs(times, multiplexing="DDM")

    expected = array([[6.00000000e+10, 6.00000000e+10],
                      [6.00198020e+10, 6.00198020e+10],
                      [6.00396040e+10, 6.00396040e+10],
                      [6.00594059e+10, 6.00594059e+10]])
    try:
        assert txfs.shape == expected.shape, f"Shape mismatch: got {txfs.shape}, expected {expected.shape}"
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def test_tx_frequency_out_of_ramp():
    """ set a simple TDM transmitter and check at 4 points inside the chirp that the freq is correct
    add a 5th point and check the freq is 0 outside the chirp"""
    # from test_assets import tdm_1chirp_8adc
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    times = linspace(-ramp_end_time, 2*ramp_end_time, 4)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freqs(times)

    expected = array([[0.00000000e+00], [0.00000000e+00],
                      [6.00594059e+10], [0.00000000e+00]])
    try:
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def test_transmitterTDM_phase():
    # setup a simple 1 chirp DDM
    from test_assets import tdm_1chirp_8adc
    ramp_end_time = tdm_1chirp_8adc.ramp_end_time
    times = linspace(0, ramp_end_time, 4)

    tx_phis = tdm_1chirp_8adc.TX_phases(times)
    expected = array([0, 0, 0, 0])
    try:
        assert allclose(tx_phis, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", tx_phis)
        raise
    else:
        print("test_transmitterTDM_phase OK")


@pytest.mark.parametrize("start_time, offset, tx_idx, phase", [
    (0, ddm_4chirps_0_half_pi.ramp_end_time, 0, 0),
    (0, ddm_4chirps_0_half_pi.ramp_end_time, 1, 0),
    (ddm_4chirps_0_half_pi.t_inter_chirp, ddm_4chirps_0_half_pi.ramp_end_time, 1, phase_slope_half_pi)
    ])
def test_transmitterDDM_phaser0_chirp0(start_time, offset, tx_idx, phase):
    # test that the phase first antenna is always 0 for chirp idx 0
    # second antenna phase offset is 0 for chirp 0
    # second antenna phase offset is phaser1_slope for chirp 1"" "
    # ramp_end_time = ddm_4chirps_0_half_pi.ramp_end_time
    times = linspace(start_time, start_time+offset, 4)
    tx_phis = ddm_4chirps_0_half_pi.TX_phases(times, tx_idx=tx_idx)
    expected = array([phase, phase, phase, phase]) * pi
    assert allclose(tx_phis, expected, atol=1e-8)