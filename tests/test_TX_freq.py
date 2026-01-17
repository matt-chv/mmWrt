from os.path import abspath, join, pardir
import sys
from time import time, perf_counter
from numpy import allclose, arange, array, linspace, pi, tile, repeat, zeros

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import BB_IF, adc_cube_v2
from mmWrt.Scene import Radar, Transmitter, TransmitterDDM, Antenna, Receiver, Target  # noqa: E402

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"

number_adc_samples = 8
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
fs = adc_sampling_frequency

tdm1 = Transmitter(f0_min=60e9,
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

def test_radar_f1():
    times = linspace(0, t_end, 4)
    txfs = r1.TX_freq(times)
    expected = array([[0.00000000e+00,3.33333333e-07,6.66666667e-07,1.00000000e-06],
                      [6.00000000e+10,6.00033333e+10,6.00066667e+10,6.00100000e+10]])
    try:
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def test_transmitter_f1():
    """ set a simple TDM transmitter and check at 4 points inside
    the chirp that the freq is correct"""

    times = linspace(0, t_end, 4)
    txfs = t1.TX_freq(times)
    expected = array([[0.00000000e+00,3.33333333e-07,6.66666667e-07,1.00000000e-06],
                      [6.00000000e+10,6.00033333e+10,6.00066667e+10,6.00100000e+10]])
    try:
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def test_transmitter_f2():
    """ set a simple TDM transmitter and check at 4 points inside the chirp that the freq is correct
    add a 5th point and check the freq is 0 outside the chirp"""
    NA = 64
    NC = 2
    NF = 2
    TIF = 1.2e-3
    TIC = 1.4e-6
    chirp_bw = 0.01e9
    chirp_slope = 10e12
    t_end = 4/3*chirp_bw/chirp_slope  # 1e-6

    t1=Transmitter(f0_min=60e9,
                   bw=chirp_bw, slope=chirp_slope,
                   t_inter_chirp=TIC,
                   chirps_count=NC,
                   t_inter_frame=TIF,
                   frames_count=NF)
    times = linspace(0, t_end, 5)
    txfs = t1.TX_freq(times)

    expected = array([[0.00000000e+00,3.33333333e-07,6.66666667e-07,1.00000000e-06, 1.33333333e-06],
                      [6.00000000e+10,6.00033333e+10,6.00066667e+10,6.00100000e+10, 0]])
    try:
        assert allclose(txfs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", txfs)
        raise


def test_BBIF1():
    # TX is too high, IF should be zeros
    times = linspace(0, 1e-6, 4)
    f_rx = array([6.00000000e+10,6.00033333e+10,6.00066667e+10,6.00100000e+10])
    f_tx = f_rx + 1e9
    adcs = BB_IF(times, f_rx, f_tx, rx_hpf=1e3, rx_lpf=1e8, tx_phase_offset=0)
    expected = zeros(4)
    try:
        assert allclose(adcs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", adcs)
        raise
    else:
        print("passed test_BBIF1")

def test_BBIF2():
    """ TX-RX < LPF so returns sinewave
    res = BB_IF(array([0, 1.3e-7, 2.6e-7, 4e-7,
                    5.3e-7, 6.6e-7, 8e-7, 9.3e-7,
                    1e-6, 1.2e-6, 1.3e-6, 1.46e-6,
                    1.6e-6, 1.73e-6, 1.86e-6, 2e-6]),
            array([6.001e9, 6.007e9,6.014e9,6.021e9,
                    6.027e9,6.034e9,6.041e9,6.047e9,
                    6.054e9,6.061e9,6.067e9,6.074e9,
                    6.081e9,6.087e9,6.094e9,6.101e9]),
            array([6e9,6.006e9,6.013e9,6.02e9,
                    6.026e9,6.033e9,6.04e9,6.046e9,
                    6.053e9,6.06e9,6.066e9,6.073e9,
                    6.08e9,6.086e9,6.093e9,6.1e9]))
    print("RES", res)"""

    from numpy import exp, pi
    f1 = 1e6
    NA = 16
    times = linspace(0, 2/f1, NA)
    f_rx = linspace(6e9, 6.1e9, NA)
    f_tx = f_rx + f1
    adcs = BB_IF(times, f_rx, f_tx, rx_hpf=1e3, rx_lpf=1e8, tx_phase_offset=0)
    expected = exp(2*pi*1j*times*f1)
    try:
        assert allclose(adcs, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", adcs)
        raise Exception(ex)
    else:
        print("passed test_BBIF2")


def test_transmitterTDM_phase():
    times = linspace(0, t_end, 4)
    tx_phis = tdm1.TX_phases(times)
    expected = array([0, 0, 0, 0])
    try:
        assert allclose(tx_phis, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", tx_phis)
        raise
    else:
        print("test_transmitterTDM_phase OK")

def test_transmitterDDM_phaser0_chirp0():
    """ test that the phase of phase 0 for chirp idx 0 is zero"""
    times = linspace(0, t_end, 4)
    tx_phis = ddm1.TX_phases(times, tx_idx=0)
    expected = array([0, 0, 0, 0])
    try:
        assert allclose(tx_phis, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", tx_phis)
        raise
    else:
        print("test_transmitterRDM_phaser0 OK")

def test_transmitterDDM_phaser1_chirp0():
    """ test that the phase of phase 1 for chirp idx 0 is zero (phaser1_slope * 0)"""
    times = linspace(0, t_end, 4)
    tx_phis = ddm1.TX_phases(times, tx_idx=1)
    expected = array([0, 0, 0, 0])
    try:
        assert allclose(tx_phis, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", tx_phis)
        raise
    else:
        print("test_transmitterRDM_phaser1 chirp 0 OK")

def test_transmitterDDM_phaser1_chirp1():
    """ test that the phase of phaser 1 for chirp idx 0 is pi/2 (phaser1_slope * 1)"""
    times = linspace(0, t_end, 4) + TIC
    tx_phis = ddm1.TX_phases(times, tx_idx=1)
    expected = array([phaser1_slope, phaser1_slope, phaser1_slope, phaser1_slope]) * pi
    try:
        assert allclose(tx_phis, expected, atol=1e-8)
    except Exception as ex:
        print("expected", expected)
        print("got", tx_phis)
        raise
    else:
        print("test_transmitterRDM_phaser1 chirp 1 OK")
        print(GREEN+"OK"+DEFAULT)



def test_adc_chirp_v0():
    from mmWrt.Raytracing import adc_chirp_v0, adc_chirp_v1
    from numpy.fft import fft
    from scipy.signal import find_peaks

    f0_start, number_adc_samples, adc_sampling_frequency = 60e9, 8, 5e5
    chirp_slope = 5e12

    tdm1 = Transmitter(f0_min=f0_start,
                       ramp_end_time=1.2*number_adc_samples/adc_sampling_frequency,
                       slope=chirp_slope,
                       chirps_count=1,
                       frames_count=1)
    receiver1 = Receiver(antennas=[Antenna() for _ in range(1)],
                         fs=adc_sampling_frequency,
                         n_adc=number_adc_samples)
    radar1 = Radar(transmitter=tdm1, receiver=receiver1)
    target1 = Target(xt=lambda t: 5.1 + 0*t)

    adc_times_chirp0 = arange(0, radar1.number_adc_samples, 1)*(1/radar1.fs)
    start_time = perf_counter()
    adc_samples = adc_chirp_v0(adc_times_chirp0, radar1, targets=[target1],
                               radars=[radar1], datatype=complex)
    end_time = perf_counter()  # 0.87 ms
    r_fft = abs(fft(adc_samples[0, :]))
    pk0 = find_peaks(r_fft)[0][0]
    result = radar1.fs*pk0/radar1.number_adc_samples
    expected = 187500  # f_if is 170k but scalopping makes it 187.5k 
    try:
        assert result == expected
    except Exception as ex:
        print(RED+f"expected:"+DEFAULT,expected)
        print("got", result)
        raise
    else:
        print("test_adc_chirp_v0 OK")
        print(f"total time {(end_time-start_time)*1000:.2f} ms")
        print(GREEN+"OK"+DEFAULT)



def test_adc_chirp_v1():
    from mmWrt.Raytracing import adc_chirp_v0, adc_chirp_v1
    from numpy.fft import fft
    from scipy.signal import find_peaks

    f0_start, number_adc_samples, adc_sampling_frequency = 60e9, 8, 5e5
    chirp_slope = 5e12

    tdm1 = Transmitter(f0_min=f0_start,
                       ramp_end_time=1.2*number_adc_samples/adc_sampling_frequency,
                       slope=chirp_slope,
                       chirps_count=1,
                       frames_count=1)
    receiver1 = Receiver(antennas=[Antenna() for _ in range(1)],
                         fs=adc_sampling_frequency,
                         n_adc=number_adc_samples)
    radar1 = Radar(transmitter=tdm1, receiver=receiver1)
    target1 = Target(xt=lambda t: 5.1 + 0*t)

    adc_times_chirp0 = arange(0, radar1.number_adc_samples, 1)*(1/radar1.fs)
    start_time = perf_counter()
    adc_samples = adc_chirp_v1(adc_times_chirp0, radar1, targets=[target1],
                               radars=[radar1], datatype=complex)
    end_time = perf_counter()  # 0.5ms
    r_fft = abs(fft(adc_samples[0, :]))
    pk0 = find_peaks(r_fft)[0][0]
    result = radar1.fs*pk0/radar1.number_adc_samples
    expected = 187500  # f_if is 170k but scalopping makes it 187.5k 
    try:
        assert result == expected
    except Exception as ex:
        print("expected", expected)
        print("got", result)
        raise
    else:
        print("test_adc_chirp_v1 OK")
        print(f"total time {(end_time-start_time)*1000:.2f} ms")


def test_adc_chirp_v2():
    from mmWrt.Raytracing import adc_chirp_v0, adc_chirp_v1, adc_samples_v2
    from numpy.fft import fft
    from scipy.signal import find_peaks

    f0_start, number_adc_samples, adc_sampling_frequency = 60e9, 8, 5e5
    chirp_slope = 5e12

    tdm1 = Transmitter(f0_min=f0_start,
                       ramp_end_time=1.2*number_adc_samples/adc_sampling_frequency,
                       slope=chirp_slope,
                       chirps_count=1,
                       frames_count=1)
    receiver1 = Receiver(antennas=[Antenna() for _ in range(1)],
                         fs=adc_sampling_frequency,
                         n_adc=number_adc_samples)
    radar1 = Radar(transmitter=tdm1, receiver=receiver1)
    my_targets = [target1 for _ in range(100)]

    adc_times_chirp0 = arange(0, radar1.number_adc_samples, 1)*(1/radar1.fs)
    start_time = perf_counter()

    adc_samples = adc_samples_v2(adc_times_chirp0, radar1, targets=my_targets,
                               radars=[radar1], datatype=complex)
    print(adc_samples)
    end_time = perf_counter()  # 0.3ms for 1 target up to 1.38ms for 100 targets
    r_fft = abs(fft(adc_samples[0, :]))
    pk0 = find_peaks(r_fft)[0][0]
    result = radar1.fs*pk0/radar1.number_adc_samples
    expected = 187500  # f_if is 170k but scalopping makes it 187.5k 
    try:
        assert result == expected
    except Exception as ex:
        print("expected", expected)
        print("got", result)
        raise
    else:
        print("test_adc_chirp_v2 OK")
        print(f"total time {(end_time-start_time)*1000:.2f} ms")
        print(GREEN+"OK"+DEFAULT)

def test_adc_frame_v2():
    """ ADD HERE CODE to call adc_cube_v2"""
    from mmWrt.Raytracing import adc_chirp_v0, adc_chirp_v1, adc_samples_v2
    f0_start, number_adc_samples, adc_sampling_frequency = 60e9, 8, 5e5
    chirps_per_frame=1
    total_frame_count=2
    chirp_slope = 5e12

    tdm1 = Transmitter(f0_min=f0_start,
                       ramp_end_time=1.2*number_adc_samples/adc_sampling_frequency,
                       slope=chirp_slope,
                       chirps_count=1,
                       frames_count=total_frame_count)
    receiver1 = Receiver(antennas=[Antenna() for _ in range(1)],
                         fs=adc_sampling_frequency,
                         n_adc=number_adc_samples)
    radar1 = Radar(transmitter=tdm1, receiver=receiver1)
    my_targets = [target1 for _ in range(100)]

    # A FAIRE:
    # adc_cube_v2: tstart/tend - independant chirp/frame
    # adc_times_chirp0 = arange(0, radar1.number_adc_samples, 1)*(1/radar1.fs)
    # adc_times_N_frames_M_chirps = ....
    tdm1.t_inter_frame = 2e-5
    chirp_idx = tile(arange(0, chirps_per_frame), total_frame_count)
    frame_idx = repeat(arange(0, total_frame_count), chirps_per_frame)
    start_time = tdm1.t_inter_frame*frame_idx + \
                    tdm1.t_inter_chirp*chirp_idx + tdm1.tx_start_time

    end_time = start_time + tdm1.ramp_end_time
    adc_times_2d = linspace(start_time, end_time, num=number_adc_samples, axis=1)
    adc_times = adc_times_2d.flatten()
    adc_values = adc_samples_v2(adc_times, radar1, targets=my_targets,
                                radars=[radar1], datatype=complex)
    print(adc_values.shape)
    print(GREEN+"OK"+DEFAULT)

if __name__ == "__main__":
    # test_BBIF1()
    # test_BBIF2()
    #res = BB_IF(array([0,3e-7,6.6e-7,1.e-6]),
    #            array([6e10,6.0003e10,6.0006e+10,6.001e10]),
    #            array([6.1e10,6.1003e10,6.1006e10,6.101e10]))

    # test_transmitter_f1()
    # test_transmitter_f2()
    # test_radar_f1()
    # test_transmitterDDM_phaser0_chirp0()
    # test_transmitterDDM_phaser1_chirp0()
    # test_transmitterDDM_phaser1_chirp1()
    test_adc_chirp_v0()
    test_adc_chirp_v1()
    test_adc_chirp_v2()
    test_adc_frame_v2()