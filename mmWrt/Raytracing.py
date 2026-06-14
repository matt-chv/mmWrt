""" This is where the raytracing happens
rt_points - main function to perform raytracing with point scatterers
BB_IF - function to compute the BaseBand Intermediate Frequency

BB_IF is called by rt_points to compute each respective scatterer's IF contribution
"""
import logging
from numpy import any, arctan2, arange, array, concatenate, exp, mean, newaxis, pi, real, sum, sqrt, where, zeros, empty
from numpy import abs as np_abs, max as np_max
from numpy.typing import NDArray
from numpy import float64, float32, float16, int64, int32, int16  # alternatives: float16, float64
from numpy import complex128 as complex
from numpy import complex64
from numpy.linalg import norm as euclidian_distance
from numpy import tile, sum
import numpy as np

from time import perf_counter
from .mylogs import auto_log

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

from .Scene import two_way_range


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.ERROR)  # Default module level
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
module_logger.addHandler(ch)


@auto_log
def sample_all_rays(adc_times,
                    radars,
                    scatterers,
                    receiver_radar,
                    datatype=float32,
                    radar_equation=False,
                    debug=False, log=None) -> NDArray:
    """ Computes the ADC samples at the given ADC times for the receiver radar
    v2 (now fully vectorised)
    reserved for future release to replace rt_points ?!?!?

    Parameters
    ----------
    adc_times
        (T): absolute time
    Returns
    -------
    adc_samples
        (T, RX) - note this convention is changed in raytracing level to (RX, T)
    """
    rx_high_pass_freq = receiver_radar.receiver.rx_high_pass_freq
    rx_low_pass_freq = receiver_radar.receiver.rx_low_pass_freq
    number_adc_samples = adc_times.shape[0]
    adc_samples = zeros((number_adc_samples,
                         len(receiver_radar.rx_antennas)))

    rx_antennas_positions = receiver_radar.position_rx_antennas(adc_times)

    scatterer_position = empty((adc_times.shape[0], len(scatterers),  3))
    scatterer_count = len(scatterers)
    for i, scatterer in enumerate(scatterers):
        scatterer_position[:,i,:] = scatterer.pos_t1(adc_times)

    for radar in radars:
        tx_antennas_positions = radar.position_tx_antennas(adc_times)

        # total_distance: [T, TX, S, RX]
        total_distance = two_way_range(tx_antennas_positions,
                                       scatterer_position,
                                       rx_antennas_positions)
        log.debug(f"total_distance: {total_distance[0,:,:,:]}")

        time_of_flight = total_distance/receiver_radar.v
        log.debug(f"time_of_flight: {time_of_flight[0,:,:,:]}")

        # NOTE: adc_times has the shape (T,)
        # as timestamp_tensor will be used over multiple TX/scatterer/RX path
        # it has to be repeated
        # to become of the shape (T, TX, S, RX)
        timestamp_tensor = adc_times[:, None, None, None]
        timestamp_tensor = np.repeat(timestamp_tensor, len(radar.tx_antennas), axis=1)
        timestamp_tensor = np.repeat(timestamp_tensor, scatterer_count, axis=2)
        timestamp_tensor = np.repeat(timestamp_tensor,
                                     len(receiver_radar.rx_antennas), axis=3)

        radar_tx_times = timestamp_tensor - time_of_flight

        # NOTE: the convention is that the received signal by the receiving radar
        # is the frequency at which it was transmitted by the transmitting radar 
        # at sampling time - time_of_flight
        # which is why, only f_rx is passed to mixer
        # as in the mixer, the receiver_radar TX_freq is called with `0` delay since
        # it is the local oscillator
        f_rx = radar.TX_freq(radar_tx_times)
        ph_rx = radar.TX_phases(radar_tx_times)

        log.debug(f"f_rx:{f_rx[:,0,0,0]}")
        f_if = receiver_radar.mixer(adc_times, f_rx)
        log.debug(f"fif:{f_if[:,0,0,0]}")
        log.debug(f"adc_samples t=0, before incremental sums:{adc_samples[0:10,0]}")

        adc_sampled = receiver_radar.adc_sampling(f_if=f_if,
                                                   ph_rx=ph_rx,
                                                   adc_times=timestamp_tensor,
                                                   time_of_flight=time_of_flight,
                                                   datatype=datatype)

        adc_samples += adc_sampled
        log.debug(f"adc_samples t=0, *after* incremental sums:{adc_samples[0:10,0]}")

    return adc_samples


def rt_points(radars, scatterers, receiver_radar,
              radar_equation=False,
              datatype=float32, debug=False,
              **raytracing_opt):
    """ raytracing with points

    Parameters
    ----------
    receiver_radar: Radar
        instance of Radar for which the BB cube is computed. One of the radars in the scene.
        (renamed in 0.0.10 from radar)
    scatterers: List[Scatterer]
        list of scatterers in the Scene
    radar_equation: bool
        if True includes the radar equation when computing the IF signal
        else ignores radar equation
    datatype: Type
        type of data to be generate by rt: float16, float32, ... or complex
    debug: bool
        if True prints log messages
    raytracing_opt: dict
        compute: bool
            if True computes raytracing (use False for radar statistics tuning)
        T_start: float
            time offset to start simulation
        radars: List[radar]
            list of interferer radars to include in the simulation, including own radar TX
    Returns
    -------
    baseband: dict
        dictonnary with adc values and other parameters used later in analysis
        adc_cube[frame_idx, chirp_idx, None, rx_idx, adc_idx]
    Raises
    ------
    ValueError
        if Nyquist rule is not upheld
    """
    n_frames = receiver_radar.frames_count
    # n_chirps is the # chirps each TX antenna sends per frame
    n_chirps = receiver_radar.chirp_count
    n_tx = len(receiver_radar.tx_antennas)
    n_rx = len(receiver_radar.rx_antennas)
    adc_sample_count = receiver_radar.receiver.adc_sample_count
    adc_sample_rate = receiver_radar.receiver.adc_sample_rate
    adc_sample_time = 1/adc_sample_rate
    bw = receiver_radar.bw
    adc_cube = zeros((n_frames, n_chirps, n_rx, adc_sample_count)).astype(datatype)
    f0_min, slope, Tc, T = 0, 0, 0, 0

    if radars is None:
        self._log.warning("future versions will not allow radars to be None")


    baseband = {"adc_cube": adc_cube,
                "frames_count": n_frames,
                "chirp_count": receiver_radar.chirp_count,
                "chirp_period": receiver_radar.chirp_period,
                "n_tx": n_tx,
                "n_rx": n_rx,
                "n_adc": adc_sample_count,
                "datatype": datatype,
                "f0_min": f0_min,
                "slope": slope,
                "bw": bw,
                "Tc": Tc,
                "TFFT": adc_sample_count*adc_sample_time,
                "T": T,
                "fs": receiver_radar.fs, "v": receiver_radar.v}
    if "compute" not in raytracing_opt:
        raytracing_opt["compute"] = False
    from tqdm import tqdm

    for frame_idx in range(n_frames):
        for chirp_idx in tqdm(range(n_chirps),
                              total=n_chirps, desc=f"Processing Chirp from frame: {frame_idx}"):
            # TODO: need to make this code more flexibile to handle
            # cases where the chirp_start_time, chirp_slope and chirp_end_time
            # vary on chirp per chirp
            start_of_chirp = frame_idx * receiver_radar.frame_period + \
                chirp_idx * receiver_radar.chirp_period
            adc_times = arange(0, adc_sample_count*adc_sample_time,
                            adc_sample_time) + start_of_chirp

            adc_values = sample_all_rays(adc_times,
                                        radars,
                                        scatterers,
                                        receiver_radar,
                                        datatype=datatype)
            # switch axis as now timestamps becomes dimension -1
            # RX becomes axis -2
            # we need to add a new empty axis for TX for backward compatibility
            adc_values = adc_values.T
            adc_values = adc_values[None, :, :]

            adc_cube[frame_idx, chirp_idx, None, :, :] = adc_values
    baseband["adc_cube"] = adc_cube
    return baseband
