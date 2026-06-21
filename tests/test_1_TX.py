""" This is mostly to test the TX_freq and phases
covers TDM and DDM, SFMCW not included
v0.0.11: 8 passed
"""
import copy
import logging
import numpy as np
from os.path import abspath, join, pardir
import sys
from numpy import allclose, arange, array, concatenate, linspace, pi
import pytest

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

# from mmWrt.Raytracing import BB_IF
from mmWrt.Scene import Antenna  # noqa: E402
from test_assets import ddm_4chirps_0_half_pi, phase_slope_half_pi
from test_assets import tdm_1chirp_8adc, radar_vibrate, scatterer_vibrate, tof_10p1m



"""adc_sample_count = 8
NA = adc_sample_count
NC = 1
NF = 1
TIF = 1.2e-3
TIC = 1.2e-6
chirp_bw = 0.01e9
chirp_slope = 5e12
t_end = chirp_bw/chirp_slope
phaser1_slope = 0.5
adc_sampling_frequency = 5e5
adc_sample_rate = adc_sampling_frequency"""

"""tdm1 = Transmitter(f0_min=60e9,
                   ramp_end_time=1.2*NA/adc_sample_rate,
                   slope=chirp_slope,
                   chirp_period=TIC,
                   chirp_count=NC,
                   frame_period=TIF,
                   frame_count=NF)

ddm1 = TransmitterDDM(f0_min=60e9,
                      bw=chirp_bw, slope=chirp_slope,
                      chirp_period=TIC,
                      antennas=[Antenna() for _ in range(2)],
                      chirp_count=NC,
                      frame_period=TIF,
                      frame_count=NF,
                      conf={"TX_phaser_slopes": [0, phaser1_slope]})

receiver1 = Receiver(antennas=[Antenna() for _ in range(1)],
                     adc_sample_rate=adc_sampling_frequency,
                     adc_sample_rate_max=110e6,
                     adc_sample_count=adc_sample_count)
scatterer1 = Scatterer(xt=lambda t: 5.1+0*t,
                 rcs_f=1.0)

# radar1 = Radar(transmitter=tdm1, receiver=receiver1)
"""

def test_tx_frequency_ramp_1_tx():
    # check that the transmitter frequencies using default values match expected ones
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    # times is (timestamps, TX antenna count, Scatterer count, RX antenna count)
    times = linspace(0, ramp_end_time, 4)
    # test is for 1TX, 1 Scatterer, 1 RX
    # reshaping times
    times = times[:, None, None, None]

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freq(times)

    assert txfs.shape == (4, 1, 1, 1)
    # change shape for easier validation
    txfs = txfs[:, 0, 0, 0]
    expected = array([6.00000000e+10, 6.00198020e+10,
                      6.00396040e+10, 6.00594059e+10])
    try:
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def tbd_tx_frequency_ramp_2_tx_tdm0():
    # changing the ramp end time to have 2 chirps
    # changing the antenna count to 2 antennas
    # making sure that times are all withing
    # first chirp
    # so 2nd antennas does not transmit
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    radar.antennas = array([Antenna(), Antenna()])
    radar.chirp_count = 1
    radar.chirp_period = ramp_end_time * 2
    # times is:
    # times = linspace(0, ramp_end_time, 4)
    # since test is for 2TX, 1 Scatterer, 1 RX
    # reshaping times by appending on axis=1
    # axis=1 is the axis of TX antennas
    times = concatenate([arange(0, ramp_end_time*4, ramp_end_time)[:, None, None, None],
                         arange(0, ramp_end_time*4, ramp_end_time)[:, None, None, None]], axis=1)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freq(times)

    # change shape for easier validation
    tx0 = txfs[:, 0, 0, 0]
    tx1 = txfs[:, 1, 0, 0]
    print("antenna 0", tx0)
    print("antenna 1", tx1)

    tx0_expected = array([6.00000000e+10, 6.00198020e+10,
                          6.00396040e+10, 6.00594059e+10])
    tx1_expected = np.zeros((4))
    assert txfs.shape == (4, 2, 1, 1), f"Shape mismatch: got {txfs.shape}, expected {(4, 2, 1, 1)}"
    assert allclose(tx0, tx0_expected, atol=1e-8), "antenna 0 TX frequencies does not match expected"
    assert allclose(tx1, tx1_expected, atol=1e-8), "antenna 1 TX frequencies does not match expected"


def test_tx_frequency_ramp_2_tx_tdm1():
    # changing the ramp end time to have 2 chirps
    # changing the antenna count to 2 antennas
    # making sure that times are all withing
    # first chirp
    # so 2nd antennas does not transmit
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    radar.antennas = array([Antenna(), Antenna()])
    radar.chirp_count = 1
    radar.chirp_period = ramp_end_time * 2 # ~1.18e-5*2
    print("ramp_end_time", ramp_end_time)
    print("linspace(0, ramp_end_time*2, 4)", arange(0, ramp_end_time*4, ramp_end_time))
    # times is:
    # times = linspace(0, ramp_end_time, 4)
    # since test is for 2TX, 1 Scatterer, 1 RX
    # reshaping times by appending on axis=1
    # axis=1 is the axis of TX antennas
    times = concatenate([arange(0, ramp_end_time*4, ramp_end_time)[:, None, None, None],
                         arange(0, ramp_end_time*4, ramp_end_time)[:, None, None, None]],
                         axis=1)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freq(times)

    # change shape for easier validation
    tx0 = txfs[:, 0, 0, 0]
    tx1 = txfs[:, 1, 0, 0]

    # in TDM the first chirp is transmitted by TX0
    tx0_expected = array([6.00000000e+10, 6.00594059e+10, 0, 0])
    # and the 2nd chirp is transmitted by TX1, with same values
    # just delayed in time
    tx1_expected = array([0, 0, 6.00000000e+10, 6.00594059e+10])
    assert txfs.shape == (4, 2, 1, 1), f"Shape mismatch: got {txfs.shape}, expected {(4, 2, 1, 1)}"
    assert allclose(tx0, tx0_expected, atol=1e-8), "antenna 0 TX frequencies does not match expected"
    assert allclose(tx1, tx1_expected, atol=1e-8), "antenna 1 TX frequencies does not match expected"



def test_tx_frequency_ramp_2_tx_ddm():
    # check that the transmitter frequencies using default values match expected ones
    # when changing multiplexing from TDM to DDM
    #
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    radar.antennas = array([Antenna(), Antenna()])
    times = concatenate([linspace(0, ramp_end_time, 4)[:, None, None, None],
                         linspace(0, ramp_end_time, 4)[:, None, None, None]], axis=1)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    radar.multiplexing = "DDM"
    txfs = radar.TX_freq(times)

    tx0 = txfs[:, 0, 0, 0]
    tx1 = txfs[:, 1, 0, 0]
    tx0_expected = array([6.00000000e+10, 6.00198020e+10,
                          6.00396040e+10, 6.00594059e+10])
    tx1_expected = array([6.00000000e+10, 6.00198020e+10,
                          6.00396040e+10, 6.00594059e+10])

    assert tx0.shape == tx0_expected.shape, f"Shape mismatch: got {tx0.shape}, expected {tx0_expected.shape}"
    assert tx1.shape == tx1_expected.shape, f"Shape mismatch: got {tx1.shape}, expected {tx1_expected.shape}"
    assert allclose(tx0, tx0_expected, atol=1e-8), "antenna TX0 frequencies does not match expected"
    assert allclose(tx1, tx1_expected, atol=1e-8), "antenna TX1 frequencies does not match expected"


def test_tx_frequency_out_of_ramp():
    # set a simple TDM transmitter and check at 1 points inside the chirp that the freq is correct
    # add a 2 points and one after and check the freq is 0 outside the chirp
    # from test_assets import tdm_1chirp_8adc
    radar = copy.deepcopy(tdm_1chirp_8adc)
    ramp_end_time = radar.ramp_end_time
    times = linspace(-ramp_end_time, 2*ramp_end_time, 4)[:,None, None, None]

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    logging.getLogger("Transmitter").setLevel(logging.DEBUG)
    txfs = radar.TX_freq(times)
    tx0 = txfs[:, 0, 0, 0]
    tx0_expected = array([0, 0, 6.00594059e+10, 0])
    assert allclose(tx0, tx0_expected, atol=1e-8), "antenna TX0 frequencies does not match expected"


def test_transmitterTDM_phase():
    # setup a simple 1 chirp DDM
    from test_assets import tdm_1chirp_8adc
    ramp_end_time = tdm_1chirp_8adc.ramp_end_time
    times = linspace(0, ramp_end_time, 4)[:, None, None, None]

    tx_phis = tdm_1chirp_8adc.TX_phases(times)
    expected = array([0, 0, 0, 0])

    assert tx_phis.shape == (4, 1, 1, 1)
    assert allclose(tx_phis, expected, atol=1e-8), "TX0 phases not as expected"


"""@pytest.mark.parametrize("start_time, offset, tx_idx, phase", [
    (0, ddm_4chirps_0_half_pi.ramp_end_time, 0, 0),
    (0, ddm_4chirps_0_half_pi.ramp_end_time, 1, 0),
    (ddm_4chirps_0_half_pi.chirp_period, ddm_4chirps_0_half_pi.ramp_end_time, 1, phase_slope_half_pi)
    ])
def test_transmitterDDM_phaser0_chirp0(start_time, offset, tx_idx, phase):
    # test that the phase first antenna is always 0 for chirp idx 0
    # second antenna phase offset is 0 for chirp 0
    # second antenna phase offset is phaser1_slope for chirp 1"" "
    # ramp_end_time = ddm_4chirps_0_half_pi.ramp_end_time
    times = linspace(start_time, start_time+offset, 4)[:, None, None, None]
    tx_phis = ddm_4chirps_0_half_pi.TX_phases(times, tx_idx=tx_idx)
    expected = array([phase, phase, phase, phase]) #* pi
    assert tx_phis.shape == (4, 1, 1, 1)
    assert allclose(tx_phis, expected, atol=1e-8)"""


def tbd_TX_Freqs_64TX_64loops():
    # v0.0.11-rc1 fix
    # times(64,64,3,64) -> freq is (64, 64, 3, 64, 4096) which is 24GiB
    # which crashes simulation
    from test_assets import radar_ura_64_TX_z
    radar = radar_ura_64_TX_z
    ts = np.array([0])
    times = np.zeros((1,64,1,64))  # (64, 64, 3, 64)
    times = np.zeros((64, 64, 3, 64))
    # (1, 64, 1, 64)
    txfs = radar.TX_freq(times)
    print(txfs.shape)


def test_tx_freq_frames():
    """ basic test on TX_Freq when frame_idx > 0"""
    radar, _ = radar_vibrate, scatterer_vibrate

    """    frameidx 0
    chirpidx 1
    [1.20e-06 1.21e-06 1.22e-06 1.23e-06 1.24e-06 1.25e-06 1.26e-06 1.27e-06
    1.28e-06 1.29e-06 1.30e-06 1.31e-06 1.32e-06 1.33e-06 1.34e-06 1.35e-06
    1.36e-06 1.37e-06 1.38e-06 1.39e-06 1.40e-06 1.41e-06 1.42e-06 1.43e-06
    1.44e-06 1.45e-06 1.46e-06 1.47e-06 1.48e-06 1.49e-06 1.50e-06 1.51e-06
    1.52e-06 1.53e-06 1.54e-06 1.55e-06 1.56e-06 1.57e-06 1.58e-06 1.59e-06
    1.60e-06 1.61e-06 1.62e-06 1.63e-06 1.64e-06 1.65e-06 1.66e-06 1.67e-06
    1.68e-06 1.69e-06 1.70e-06 1.71e-06 1.72e-06 1.73e-06 1.74e-06 1.75e-06
    1.76e-06 1.77e-06 1.78e-06 1.79e-06 1.80e-06 1.81e-06 1.82e-06 1.83e-06]
    """
    f_mix_list = []
    for frame_idx in [0,1]:
        chirp_idx = 1
        adc_sample_count = radar.adc_sample_count
        # adc_sample_count = radar.adc_sample_count
        start_of_chirp = frame_idx * radar.frame_period + \
            chirp_idx * radar.chirp_period
        adc_sample_time = 1/radar.adc_sample_rate

        adc_times = arange(0, adc_sample_count*adc_sample_time,
                           adc_sample_time) + start_of_chirp
        timestamp_tensor = adc_times[:, None, None, None]
        timestamp_tensor = np.repeat(timestamp_tensor, 1, axis=1)
        timestamp_tensor = np.repeat(timestamp_tensor, 1, axis=2)
        timestamp_tensor = np.repeat(timestamp_tensor,
                                    1, axis=3)

        tx_freq = radar.TX_freq(timestamp_tensor)
        print("fTX", tx_freq.T)
        rx_freq = radar.TX_freq(timestamp_tensor-tof_10p1m)
        print("fTX", tx_freq.T)
        f_mix = radar.mixer(adc_times, rx_freq)
        print("f_mix", f_mix.T)
        f_mix_list.append(f_mix)
    assert np.allclose(f_mix_list[0], f_mix_list[1])
    print("ok")


def test_DDM_LO_TX_Freq():
    # check that TX_Freq generates the same good
    # frequency for both antennas in DDM
    from test_assets import ddm_4chirps_0_half_pi, chirp_end_time_8adc, tdm_1chirp_8adc
    ddm_transmitter = ddm_4chirps_0_half_pi  # ddm_4chirps_0_half_pi
    adc_times = np.array([0, chirp_end_time_8adc,
                           1.1*chirp_end_time_8adc, 1.1*chirp_end_time_8adc+chirp_end_time_8adc])
    timestamp_tensor = adc_times[:, None, None, None]
    timestamp_tensor = np.repeat(timestamp_tensor, 2, axis=1)
    timestamp_tensor = np.repeat(timestamp_tensor, 1, axis=2)
    timestamp_tensor = np.repeat(timestamp_tensor,
                                1, axis=3)
    tx_freq = ddm_transmitter.TX_freq(timestamp_tensor)

    expected_tx_freq = np.array([[[[6.00000000e+10]], [[6.00000000e+10]]],
                                 [[[6.00594059e+10]],[[6.00594059e+10]]],
                                 [[[6.00000000e+10]], [[6.00000000e+10]]],
                                 [[[6.00594059e+10]],[[6.00594059e+10]]]])
    assert tx_freq.shape == expected_tx_freq.shape
    assert np.allclose(tx_freq, expected_tx_freq)


def test_DDM_phasers():
    # check that each phaser has a different phase
    from test_assets import ddm_4chirps_0_half_pi, chirp_end_time_8adc, tdm_1chirp_8adc
    ddm_transmitter = ddm_4chirps_0_half_pi  # ddm_4chirps_0_half_pi
    adc_times = np.array([0, chirp_end_time_8adc,
                           1.1*chirp_end_time_8adc, 1.1*chirp_end_time_8adc+chirp_end_time_8adc])
    timestamp_tensor = adc_times[:, None, None, None]
    timestamp_tensor = np.repeat(timestamp_tensor, 2, axis=1)
    timestamp_tensor = np.repeat(timestamp_tensor, 1, axis=2)
    timestamp_tensor = np.repeat(timestamp_tensor,
                                1, axis=3)
    tx_phase = ddm_transmitter.TX_phases(timestamp_tensor)

    expected_tx_phase = np.array([[[[0]], [[0]]],
                                 [[[0]],[[0]]],
                                 [[[0]], [[1.57079633]]],
                                 [[[0]],[[1.57079633]]]])

    assert tx_phase.shape == expected_tx_phase.shape
    assert np.allclose(tx_phase, expected_tx_phase)
