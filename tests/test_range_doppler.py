from os.path import abspath, join, pardir
import sys

from numpy import sin, pi, zeros
from numpy.fft import fftshift, fft, fft2
from numpy import complex_ as complex
from scipy.signal import find_peaks

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt.Scene import Radar, Transmitter, Receiver, Target  # noqa: E402


def test_FMCW_vibration():
    """ validates the uDoppler logic
    """
    NA = 64
    NC = 32
    NF = 256
    TIF = 1.2e-3
    radar = Radar(transmitter=Transmitter(bw=0.01e9, slope=10e12,
                                          t_inter_chirp=1.2e-6,
                                          chirps_count=NC,
                                          t_inter_frame=TIF,
                                          frames_count=NF),
                  receiver=Receiver(fs=100e6, max_adc_buffer_size=1024,
                                    max_fs=110e6,
                                    n_adc=NA),
                  debug=False)

    F1 = 6
    f1 = F1/(NF*TIF)
    v0 = 160
    A0 = v0/(2*pi*f1)

    target = Target(xt=lambda t: 4*A0*sin(2*pi*f1*t)+300)

    bb = rt_points(radar, [target], datatype=complex)  # type: ignore
    udops1 = zeros((NC, NF))
    tx_i, rx_i = 0, 0

    for frame_idx in range(NF):
        # compute range doppler
        cube = bb["adc_cube"][frame_idx, :, tx_i, rx_i, :]
        Z_fft2 = abs(fftshift(fft2(cube)))
        # find peak in range
        pk = find_peaks(abs(fft(cube[0, :])))[0][0]
        # append doppler bin at peak range
        udops1[:, frame_idx] = Z_fft2[:, pk]

    dops = []
    for frame_idx in range(NF):
        dop = find_peaks(udops1[:, frame_idx])[0][0]
        dops.append(dop)
    Y = fft(dops)
    ym = find_peaks(Y)[0][0]
    assert ym == F1
