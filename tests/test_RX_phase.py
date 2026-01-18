
from os.path import abspath, join, pardir
import sys

from numpy import angle, arange, linspace, pi, tile, repeat
from numpy.fft import fft
from scipy.signal import find_peaks

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import adc_samples
from test_assets import RED, GREEN, DEFAULT


def test_phase_delta_chirp_to_chirp():
    """ test the phase change in the target range bin change from one
    chirp to another chirp is within the expected error range

    for DFT, the d(phase)/dt = [phase(DFT[idx_peak_1st_chirp])-phase(DFT[idx_peak_2nd_chirp]) ] / (t_inter_chirp)
    the expected change of phase from chirp to chirp is for a target moving at velocity v (assuming no range bin change)
    dphase_dt = 4*pi*v/lambda_60G
    given the number of DFT bins in the velocity dimension (doppler dimension), which is the number of chirps
    the maximum error is 1 range bin:
    2*pi/t_inter_chirp/number_adc_samples
    """

    from test_assets import radar_tdm_2_chirp_8adc, \
        target_linear_speed_1mps, \
        dphase_dt_1mps, lambda_60G

    number_samples = radar_tdm_2_chirp_8adc.receiver.number_adc_samples

    chirp_idx = tile(arange(0, radar_tdm_2_chirp_8adc.transmitter.chirps_count),
                     radar_tdm_2_chirp_8adc.transmitter.frames_count)
    frame_idx = repeat(arange(0, radar_tdm_2_chirp_8adc.transmitter.frames_count),
                       radar_tdm_2_chirp_8adc.transmitter.chirps_count)
    start_time = radar_tdm_2_chirp_8adc.t_inter_frame*frame_idx + \
        radar_tdm_2_chirp_8adc.t_inter_chirp*chirp_idx + radar_tdm_2_chirp_8adc.transmitter.tx_start_time

    end_time = start_time + radar_tdm_2_chirp_8adc.transmitter.ramp_end_time
    adc_times_2d = linspace(start_time, end_time,
                            num=radar_tdm_2_chirp_8adc.receiver.number_adc_samples,
                            axis=1)
    adc_times = adc_times_2d.flatten()

    adc_values = adc_samples(adc_times, radar_tdm_2_chirp_8adc,
                             [target_linear_speed_1mps],
                             radars=[radar_tdm_2_chirp_8adc],
                             debug=True)

    r_fft_1st_chirp = fft(adc_values[0, :number_samples])
    r_fft_2nd_chirp = fft(adc_values[0, number_samples:])

    peak_1st_chirp = find_peaks(abs(r_fft_1st_chirp), height=1)[0][0]
    peak_2nd_chirp = find_peaks(abs(r_fft_2nd_chirp), height=1)[0][0]
    phase_peak_1st_chirp = angle(r_fft_1st_chirp[peak_1st_chirp])
    phase_peak_2nd_chirp = angle(r_fft_2nd_chirp[peak_2nd_chirp])

    dphase_dt = (phase_peak_1st_chirp-phase_peak_2nd_chirp)/radar_tdm_2_chirp_8adc.t_inter_chirp

    max_expected_dphase_dt_error = 2*pi/radar_tdm_2_chirp_8adc.t_inter_chirp/radar_tdm_2_chirp_8adc.receiver.number_adc_samples

    try:
        assert abs(dphase_dt-dphase_dt_1mps) < max_expected_dphase_dt_error
    except Exception as ex:
        print(RED+f"NOK: expected{dphase_dt_1mps}"+DEFAULT)
        print("computed", dphase_dt)
        print("delta", dphase_dt_1mps-dphase_dt)
        print("bin size", max_expected_dphase_dt_error)
        raise
    else:
        print("test_if_1_target_1_chirp:"+GREEN+"OK"+DEFAULT)

if __name__ == "__main__":
    test_phase_delta_chirp_to_chirp()
