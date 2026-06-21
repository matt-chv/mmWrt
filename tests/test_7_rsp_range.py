""" testing RSP.Range
Covers:
 - TDM mode
(not written DDM, SFMCW)
Does not cover yet:
- non point scatterers
- attenuation
v0.0.11: 1
"""
import logging
import numpy as np

from mmWrt.Raytracing import sample_all_rays
from mmWrt.RadarSignalProcessing import ranges_from_fft_threshold, \
    ranges_dft_cfar

from test_assets import adc_sampling_times_8_samples, \
    adc_sampling_times_64_samples, radar_tdm_1_chirp_8_adc, \
    scatterer_static_5p1m, radar_tdm_1_chirp_64_adc

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"


def test_rsp_range_with_peak_find():
    # test that given a given scatterer, we get expected adc value
    # timesamples = adc_sampling_times_8_samples  # [:, None, None, None]
    radar = radar_tdm_1_chirp_8_adc
    chirp_slope = radar.transmitter.chirp_slope
    adc_sample_rate = radar.receiver.adc_sample_rate
    logging.getLogger("mmWrt.Raytracing.sample_all_rays")\
        .setLevel(logging.DEBUG)

    adc_values = sample_all_rays(adc_sampling_times_8_samples,
                                 [radar],
                                 [scatterer_static_5p1m],
                                 radar)
    adc0 = adc_values[:, 0]

    ranges = ranges_from_fft_threshold(adc0,
                                       chirp_slope=chirp_slope,
                                       adc_sample_rate=adc_sample_rate,
                                       fft_threshold=1)
    assert ranges.shape == (1,), "only one value should be reported"
    assert np.allclose(ranges, [3.7875]), \
        "computed range with default setup should 3.7875 \
         (RMSE within one range bin)"


def tbd_rsp_range_with_cfar():
    # FIXME: this code needs to be fixed - today
    # calling CFAR with FFT size 8 -> does not work anymore
    # test that given a given scatterer, we get expected adc value
    # timesamples = adc_sampling_times_8_samples  # [:, None, None, None]
    # logging.basicConfig(level=logging.DEBUG, force=True)
    # logging.getLogger("mmWrt.RadarSignalProcessing._cfar_ca").setLevel(logging.DEBUG)
    # logging.getLogger("mmWrt.Scene").setLevel(logging.DEBUG)
    radar = radar_tdm_1_chirp_64_adc
    chirp_slope = radar.transmitter.chirp_slope
    adc_sample_rate = radar.receiver.adc_sample_rate
    adc_values = sample_all_rays(adc_sampling_times_64_samples,
                                 [radar],
                                 [scatterer_static_5p1m],
                                 radar)
    adc0 = adc_values[:, 0]
    # NOTE: if setting pfa=1e-6 below the CFAR threshold
    # will reject the main lobe
    # to be highlighted in examples
    ranges = ranges_dft_cfar(adc0,
                             chirp_slope=chirp_slope,
                             adc_sample_rate=adc_sample_rate,
                             pfa=0.01)

    assert ranges.shape == (1,), \
        f"only one value should be reported, 1 b/o real sampling \
          and no grouping anything else is regression. Got: {ranges.shape}"
    assert np.allclose(ranges[0], [5.2078125]), \
        "computed range with default setup should \
            3.7875 (RMSE within one range bin)"


# FIXME: turn below back into unit test

"""
def __range__wrapper(scatterer_idxes=[0], radars_idxes=[0],
                     distance_idxes=[0],
                     cfar_peak_detect=False,
                     data_type=float32,
                     peak_threshold=2):
    "" " This is the function which wraps all the range tests, to avoid code repetition. 
    
    Parameters
    ----------
    scatterer_idxes: List[int]
        indexes of scatterers from the test assets
    radars_idxes: List[int]
        indexes of radars from the test assets
    distance_idxes: List[int]
        index of ground truth distance from the test assets
    cfar_peak_detect: bool
        True uses CFAR to detect peaks, False uses a simple threshold on the FFT
        Should be set to 
    data_type: numpy.dtype
    peak_threshold: float
    
    Returns
    -------
    None

    Raises
    ------
    ValueError: "Error too large"
        Raised if the error is too large, mostly used to ensure code does not pass silently
    "" "
    c = 3e8

    radar = radars[radars_idxes[0]]

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_sample_count = radar.receiver.adc_sample_count

    adc_sample_rate = radar.receiver.adc_sample_rate
    chirp_slope = radar.transmitter.chirp_slope
    adc_sample_count = radar.receiver.adc_sample_count
    adc_times = arange(0, radar.adc_sample_count, 1) * \
        (1/adc_sample_rate)
    adc_values = adc_samples(adc_times, radar,
                             [scatterers[idx] for idx in scatterer_idxes],
                             radars=[radars[idx] for idx in radars_idxes])
    if cfar_peak_detect:

        ranges = ranges_dft_cfar(adc_values[0, :],
                             chirp_slope=chirp_slope,
                             adc_sample_rate=adc_sample_rate,
                             fft_threshold=peak_threshold)
    else:
        ranges = ranges_from_fft_threshold(adc_values[0, :],
                                        chirp_slope=chirp_slope,
                                        adc_sample_rate=adc_sample_rate,
                                        fft_threshold=peak_threshold)
    
    range_bin_width = adc_sample_rate * c / \
        (2*chirp_slope*adc_sample_count)

    for idx, _ in enumerate(scatterer_idxes):
        error = abs(ranges[idx]-distances[distance_idxes[idx]])
        try:
            assert  error < range_bin_width
        except Exception as ex:
            raise ValueError("Error too large")


def tbd_if_error_radar_tdm_1_chirp_8_adc_scatterer_static_5p1m():
    # Test that in TDM mode the range estimation is within one range bin width
    # Given a 8 ADC samples per chirp

    radar = radar_tdm_1_chirp_8_adc
    adc_times = adc_sampling_times_8_samples
    scatterer = scatterer_static_5p1m
    radar.tx_antennas = [Antenna() for _ in range(2)]
    radar.rx_antennas = [Antenna() for _ in range(2)]
    adc_values = adc_samples(adc_times, radar,
                             [scatterer],
                             radars=[radar])
    print(adc_values.shape)
    assert False


def test_if_error_radar_tdm_1_chirp_8_adc_scatterer_static_5p1m_cfar():
    "" "
    Test that in TDM mode the range estimation is within one range bin width
    Given a 8 ADC samples per chirp
    "" "
    __range__wrapper(scatterer_idxes=[0], radars_idxes=[0],
                     distance_idxes=[0], cfar_peak_detect=True)


def test_if_error_radar_tdm_1_chirp_1024_adc_scatterer_static_5p1m():
    "" "
    Test that in TDM mode the range estimation is within one range bin width
    Given a 1024 ADC samples per chirp
    "" "
    __range__wrapper(scatterer_idxes=[0], radars_idxes=[1], distance_idxes=[0])


def test_if_error_radar_tdm_1_chirp_1024_adc_scatterer_linear_speed_1mps():
    "" "
    Test: TDM mode 1 radar 1 scatterer linear motion 
    Given a 1024 ADC samples per chirp

    Ensures range error is within one range bin
    "" "
    __range__wrapper(scatterer_idxes=[2], radars_idxes=[1], distance_idxes=[0])


def test_error_raised():
    "" "
    Test: TDM, 1 radar, 1 scatterers ensure fails with *IN*-valid range reference
    "" "
    try:
        __range__wrapper(scatterer_idxes=[2], radars_idxes=[1], distance_idxes=[1])
    except Exception as ex:
        try:
            assert str(ex) == "Error too large"
        except:
            raise
    else:
        raise ValueError("Error not raised")


def test_if_error_radar_tdm_1_chirp_1024_adc_scatterer_linear_speed_1mps_complex64():
    "" "
    Test that in TDM mode the range estimation is within one range bin width
    even with a linear moving scatterer
    Given a 1024 ADC samples per chirp
    >> with datatype being complex64 (instead of float32)
    "" "
    from test_assets import radar_tdm_1_chirp_1024_adc, scatterer_linear_speed_5p1m_1mps, d_5p1m
    from numpy import complex64
    __range__wrapper(scatterer_idxes=[2], radars_idxes=[1], distance_idxes=[0],
                     data_type=complex64)


def test_if_error_radar_tdm_1_chirp_1024_adc_scatterer_linear_speed_1mps_complex128():
    "" "
    Test that in TDM mode the range estimation is within one range bin width
    even with a linear moving scatterer
    Given a 1024 ADC samples per chirp
    >> with datatype being complex128 (instead of float32)
    "" "
    __range__wrapper(scatterer_idxes=[2], radars_idxes=[1], distance_idxes=[0],
                     data_type=complex128)


def test_if_radar1_2_scatterers():
    "" "
    Test: TDM, 1 radar, 2 scatterers ensure passes with 2 valid ranges
    "" "
    __range__wrapper(scatterer_idxes=[2,3], radars_idxes=[1], distance_idxes=[0,1])


def test_error_radar1_2_scatterers():
    "" "
    Test: TDM, 1 radar, 2 scatterers ensure fails with 2 *IN*-valid ranges
    "" "
    try:
        __range__wrapper(scatterer_idxes=[2,3], radars_idxes=[1],
                         distance_idxes=[0,0])
    except Exception as ex:
        try:
            assert str(ex) == "Error too large"
        except:
            raise
    else:
        raise ValueError("Error not raised")


def test_error_eps_rangebin():
    "" "
    Test: TDM, 1 radar, 2 scatterers ensure fails with 1 *IN*-valid range just
    slightly larger than one range bin
    "" "
    try:
        __range__wrapper(scatterer_idxes=[0], radars_idxes=[1],
                         distance_idxes=[2])
    except Exception as ex:
        try:
            assert str(ex) == "Error too large"
        except:
            raise
    else:
        raise ValueError("Error not raised")

def test_cfar():
    __range__wrapper(scatterer_idxes=[0], radars_idxes=[0],
                     distance_idxes=[0], cfar_peak_detect=True)

def test_cfar_error():
    try:
        __range__wrapper(scatterer_idxes=[0], radars_idxes=[0],
                         distance_idxes=[1], cfar_peak_detect=True)
    except Exception as ex:
        try:
            assert str(ex) == "Error too large"
        except:
            raise
    else:
        raise ValueError("Error not raised")

"""  # noqa 501
