from os.path import abspath, join, pardir
import sys
from scipy.fft import fft
from scipy.signal import find_peaks
from numpy import zeros
import pytest

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Scene import Antenna, Medium, Radar, Receiver, \
    Target, Transmitter  # noqa: E402
from mmWrt.Raytracing import rt_points  # noqa: E402
from mmWrt import __version__  # noqa: E402


@pytest.mark.skipif(__version__.find("rc") >= 0, reason="only for release")
def test_SIMO_AoA():
    f0 = 62e9
    # Number of ADC samples
    NA = 64
    # Number of RX channels
    NR = 64

    void = Medium()
    lambda0 = void.v/f0
    RXs = [Antenna(x=lambda0/2*i) for i in range(NR)]
    radar = Radar(transmitter=Transmitter(bw=3.5e9, slope=70e8),
                  receiver=Receiver(fs=4e3, max_adc_buffer_size=2048,
                                    n_adc=NA,
                                    antennas=RXs),
                  debug=False)
    target1 = Target(5.1, 0, 0)
    target2 = Target(0, 10.1, 0)
    targets = [target1, target2]

    bb = rt_points(radar, targets,
                   debug=False)
    cube = bb["adc_cube"]
    # Range FFT
    R_fft = fft(cube, axis=4)
    # AoA FFT
    A_FFT = fft(R_fft, axis=3)
    for rx_idx in range(NR//4):
        A_FFT[0, 0, 0, 4*rx_idx+1, :] = zeros(NA)
        A_FFT[0, 0, 0, 4*rx_idx+2, :] = zeros(NA)
        A_FFT[0, 0, 0, 4*rx_idx+3, :] = zeros(NA)

    # Z_fft = abs(A_FFT[0, 0, 0, :, :])
    # plt.xlabel("Range")
    # plt.ylabel("AoA")
    # plt.title('AoA-Range 2D FFT')
    # plt.imshow(Z_fft)
    # plt.savefig("ZFFT.png")
    # compute spare array AoA FFT
    for idx in range(10):
        pk = find_peaks(abs(R_fft[0, 0, 0,
                                  idx, :NA//2]))
        print(pk[0][0])
        assert pk[0][0] == 4
        # print(pk[0])
        # print("--")
    # plt.title("AoA FFT")
    # plt.plot(abs(A_FFT[0,0,0,:,4]))
    # plt.savefig("AoA FFT B4.png")
    # plt.savefig("AoA FFT A8_3.png")
