""" This is where the raytracing happens
rt_points - main function to perform raytracing with point targets
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


def __BB_IF_single_radar__(f0_min, slope, T, antenna_tx, antenna_rx, target,
          medium,
          TX_phase_offset=0.0,
          datatype=float32, radar_equation=False, debug=False):
    """ This function implements the mathematical IF defined in latex as
    y_{IF} = cos(2 \\pi [f_0\\delta + s * \\delta * t - s* \\delta^2])
    into following python code
    y_IF = cos (2*pi*(f_0 * delta + slope * delta * T + slope * delta**2))

    Parameters
    ----------
    f0_min: float
        the frequency at the beginning of the chirp
    slope: float
        the slope with which the chirp frequency increases over time
    T: ndarray
        the 1D vector containing time values'
    antenna_tx: Antenna
        x, y, z coordinates
    antenna_rx: Antenna
        x, y, z coordinates
    target: Target
        instance of Target()
    medium : Medium
        instance of Medium
    TX_phase_offset: Float
        phase offset (TX phase for given TX channel), defaults to 0
    datatype: type
        either float16, 32, 64 or complex128
    radar_equation: bool
        if True adds Radar Equation contribution to IF values
    debug: bool
        if True displays debug information
    Returns
    -------
    YIF : ndarray
        vector containing the IF values
    """
    # while T is the absolute time
    # Tc is the relative time to begining of chirp (and of the ramp)
    Tc = T-T[0]

    tx_x, tx_y, tx_z = antenna_tx.xyz
    rx_x, rx_y, rx_z = antenna_rx.xyz
    t_x, t_y, t_z = target.pos_t(T) # [0])
    v = medium.v
    L = medium.L
    distance = sqrt((tx_x - t_x)**2 + (tx_y - t_y)**2 + (tx_z - t_z)**2)
    distance += sqrt((rx_x - t_x)**2 + (rx_y - t_y)**2 + (rx_z - t_z)**2)
    # if debug:
    #    print(f"distance: {distance:.2g}")
    # note delta_time is d/v because d is already there and back
    # (usually 2*d in text books)
    delta = distance / v
    # if debug:
    #    print(f"delta t: {delta:.2g}")
    # compute fif_max for upper layer to ensure Nyquist
    fif_max = 2*slope*max(distance)/v
    # if debug:
    #    print("fi_if", fif_max)

    YIF = exp(2 * pi * 1j *
              (f0_min * delta + slope * delta * Tc - slope/2 * delta**2) +
              1j*TX_phase_offset)

    if not datatype == complex:
        YIF = real(YIF)
    # if datatype == complex:
    #    YIF = exp(2 * pi * 1j *
    #              (f0_min * delta + slope * delta * T + slope * delta**2))
    # else:
    #    YIF = cos(2 * pi *
    #              (f0_min * delta + slope * delta * T + slope * delta**2))
    # here bring in the radar equation
    # target_type and RCS
    # most targets will have 1/R*4, corner reflector as 1/R**2
    # and antenna radiation patterns in azimuth, elevation
    # and frequency response
    # f0 being the center frequency of the chirp
    # f0 = f0_min + slope*(T[-1]-T[0])/2
    # Ptc = conducted Power in W
    # Ptr = Ptc * Gt(azimuth, elevation, f0)
    # Ptarget = Ptr * 1/(4*pi*distance**2) * RCS
    # if target is `corner reflector
    # Prx = Ptarget * L
    # else
    # Prx = Ptarget * 1/(4*pi*distance**2) * L
    # Where L = Medium Losses during propagation *
    #       fluctuation Losses (often modeled w/ Swerling models)
    # Prx_e = Prx * AW (where AW is effective area RX antenna)
    # Prx_c = Prx_c * Gr(azimuth, elevation, f0)
    if radar_equation:
        # FIXME: add here that with physic samples should be `0`
        # for T<distance/v
        # because of ToF no mixing possible...
        azimuth_rx = arctan2(rx_x-t_x, rx_y-t_y)
        azimuth_tx = arctan2(tx_x-t_x, tx_y-t_y)
        elevation_rx = arctan2(rx_y-t_y, rx_z-t_z)
        elevation_tx = arctan2(tx_y-t_y, tx_z-t_z)

        f0 = f0_min + slope*(T[-1]-T[0])/2
        YIF = YIF * antenna_tx.gain(azimuth_tx, elevation_tx, f0) \
            * antenna_rx.gain(azimuth_rx, elevation_rx, f0)

        YIF = YIF * target.rcs(f0)
        if target.target_type == "corner_reflector":
            YIF = YIF / distance**2
        else:
            YIF = YIF / distance**4
        YIF = YIF * 10**(L*distance)
        # FIXME: add here that YIF = 0 for t<ToF
    IF = (YIF, fif_max)
    return IF


def __retarded_TX_F__old(adc_times, radars, scatterers, receiver_radar):
    """
    Returns
    -------
    tx_freqs
        (T, TX, S, RX)
    """
    tx_antenna_positions = concatenate([radar.position_tx_antennas(adc_times) for radar in radars],
                                       axis=1)
    rx_antenna_positions = receiver_radar.position_rx_antennas(adc_times)
    # scatterer_positions = empty((adc_times.shape[0], len(scatterers),  3))
    scatterer_positions = concatenate([scatterer.pos_t1(adc_times)[:, None, :] for scatterer in scatterers],
                                      axis=1)
    total_distance = two_way_range(tx_antenna_positions, scatterer_positions, rx_antenna_positions)
    time_of_flight = total_distance/receiver_radar.v
    retarded_time = adc_times[:, None, None, None] - time_of_flight
    print(retarded_time.shape)
    tx_freqs = receiver_radar.TX_freqs(retarded_time)
    print(tx_freqs.shape)

    return tx_freqs


def __FIF__old(adc_times, radars, scatterers, receiver_radar):
    # the RX frequency is the retarded TX frequency
    rx_freqs = retarded_TX_F(adc_times, radars, scatterers, receiver_radar)
    # the TX frequency is the TX frequency which is used at MIXER input
    # at the timestamps of the ADC samples
    tx_freqs = receiver_radar.TX_freqs(adc_times)
    if_freqs = tx_freqs-rx_freqs
    return if_freqs

@auto_log
def sample_all_rays(adc_times,
                    radars,
                    targets,
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
    adc_samples: NDArray
        (adc_times x receiver antenna count)
    """
    # FIXME: change from targets to scatterers everywhere
    scatterers = targets
    rx_high_pass_freq = receiver_radar.receiver.rx_high_pass_freq
    rx_low_pass_freq = receiver_radar.receiver.rx_low_pass_freq
    number_adc_samples = adc_times.shape[0]
    adc_samples = zeros((number_adc_samples,
                         len(receiver_radar.rx_antennas)))

    # adc_samples = zeros((n_rx, number_adc_samples)).astype(datatype)
    # rx_antennas_pos = zeros((n_rx, number_adc_samples))
    # tx_antennas_count = sum([len(radar.tx_antennas) for radar in radars])
    # tx_antennas_pos = zeros((n_rx, number_adc_samples))
    # targets_pos = zeros((len(targets), 3, numer_adc_samples))

    # rx_antennas_pos = array([rx_antenna.xyz for rx_antenna in receiver_radar.rx_antennas])
    rx_antennas_positions = receiver_radar.position_rx_antennas(adc_times)
    # rx_antennas_pos = tile(rx_antennas_pos, (len(targets), 1, number_adc_samples))

    scatterer_position = empty((adc_times.shape[0], len(scatterers),  3))
    scatterer_count = len(scatterers)
    for i, target in enumerate(scatterers):
        scatterer_position[:,i,:] = target.pos_t1(adc_times)
    # diff = targets_positions - tx_antennas_pos # 2000 targets * 1024 samples  operations
    # distance_tx_target = sqrt(sum(diff * diff, axis=-1))

    # tx_antennas_pos = array([tx_antenna.xyz for radar in radars for tx_antenna in radar.tx_antennas])
    # FIXME: from
    # tx_antennas_positions = receiver_radar.position_tx_antennas(adc_times)
    # FIXME: to
    # tx_antennas_positions = (any transmitter radar) .position_tx_antennas(adc_times)
    for radar in radars:
        tx_antennas_positions = radar.position_tx_antennas(adc_times)

        """distance_tx_target = scene_distance(targets_positions, tx_antennas_pos)
        print("distance_tx", distance_tx_target.shape)
        # Compute the distance from target to rx for each time point
        # distance_target_rx = euclidian_distance(targets_positions - rx_antennas_pos, axis=1)
        # diff = targets_positions - rx_antennas_pos
        # distance_target_rx = sqrt(sum(diff * diff, axis=-1))
        distance_target_rx = scene_distance(targets_positions, rx_antennas_pos)

        # Total distance is the sum of both distances for each time point
        total_distance = distance_tx_target + distance_target_rx
        if debug:
            print("TOTAL DISTANCE", total_distance)"""
        # total_distance: [T, TX, S, RX]
        total_distance = two_way_range(tx_antennas_positions,
                                       scatterer_position,
                                       rx_antennas_positions)
        log.debug(f"total_distance: {total_distance[0,:,:,:]}")


        time_of_flight = total_distance/receiver_radar.v
        log.debug(f"time_of_flight: {time_of_flight[0,:,:,:]}")

        # for radar in radars:
        # before f_rx = array([radar.TX_freqs(adc_times-time_of_flight) for radar in radars])
        # f_rx = array([receiver_radar.TX_freqs(adc_times-time_of_flight) for receiver_radar.rx_antenna in receiver_radar.rx_antennas])
        # FIXME: before here ???? adc_times has to be (T,)
        # now has to be (T, TX, S, RX)
        timestamp_tensor = adc_times[:, None, None, None]
        timestamp_tensor = np.repeat(timestamp_tensor, len(radar.tx_antennas), axis=1)
        timestamp_tensor = np.repeat(timestamp_tensor, scatterer_count, axis=2)
        timestamp_tensor = np.repeat(timestamp_tensor,
                                     len(receiver_radar.rx_antennas), axis=3)

        radar_tx_times = timestamp_tensor - time_of_flight

        f_rx = radar.TX_freqs(radar_tx_times)  # for receiver_radar.rx_antenna in receiver_radar.rx_antennas])

        ph_rx = radar.TX_phases(radar_tx_times)  #adc_times-time_of_flight)

        f_if = receiver_radar.mixer(adc_times, f_rx)
        log.debug(f"fif:{f_if}")
        log.debug(f"adc_samples t=0, before incremental sums:{adc_samples[0:10,0]}")
        adc_samples += receiver_radar.adc_sampling(f_if=f_if,
                                                   ph_rx=ph_rx,
                                                   adc_times=timestamp_tensor,
                                                   time_of_flight=time_of_flight,
                                                   datatype=datatype)
        log.debug(f"adc_samples t=0, *after* incremental sums:{adc_samples[0:10,0]}")

    return adc_samples


def rt_points__old(receiver_radar, targets, radar_equation=False,
              datatype=float32, debug=False,
              raytracing_opt={"compute": True}):
    """ raytracing with points

    Parameters
    ----------
    receiver_radar: Radar
        instance of Radar for which the BB cube is computed. One of the radars in the scene.
        (renamed in 0.0.10 from radar)
    targets: List[Target]
        list of targets in the Scene
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

    Raises
    ------
    ValueError
        if Nyquist rule is not upheld
    """
    n_frames = receiver_radar.frames_count
    # n_chirps is the # chirps each TX antenna sends per frame
    n_chirps = receiver_radar.chirps_count
    n_tx = len(receiver_radar.tx_antennas)
    n_rx = len(receiver_radar.rx_antennas)
    n_adc = receiver_radar.n_adc
    ts = 1/receiver_radar.fs
    bw = receiver_radar.bw

    adc_cube = zeros((n_frames, n_chirps, n_rx, n_adc)).astype(datatype)
    times = zeros((n_frames, n_chirps, n_rx, n_adc))
    f0_min = receiver_radar.f0_min
    slope = receiver_radar.slope
    T = arange(0, n_adc, 1)
    # T is the absolute time across the simulation
    T = T*ts

    if "logging_level" not in raytracing_opt:
        raytracing_opt["logging_level"] = 40
    if "compute" not in raytracing_opt:
        raytracing_opt["compute"] = True

    v = receiver_radar.medium.v
    Tc = n_adc*ts #bw/slope
    if n_chirps > 1:
        try:
            assert receiver_radar.t_inter_chirp > Tc
        except Exception as ex:  # pragma: no cover
            log_msg = f"{str(ex)} for Tc: {Tc:.2g} vs " + \
                f"T_interchip: {receiver_radar.t_inter_chirp: .2g}"
            if debug:
                print(log_msg)
            if raytracing_opt["logging_level"] >= 40:
                raise ValueError(log_msg)

    if n_frames > 1:
        try:
            assert receiver_radar.t_inter_frame > (receiver_radar.t_inter_chirp*n_chirps)
        except Exception as ex:  # pragma: no cover
            log_msg = f"{str(ex)} for TF: {receiver_radar.t_inter_frame:.2g} " +\
                f"vs NC*T_interchip: {receiver_radar.t_inter_chirp*n_chirps: .2g}"
            if debug:
                print(log_msg)
            if raytracing_opt["logging_level"] >= 40:
                raise ValueError(log_msg)

    baseband = {"adc_cube": adc_cube,
                "frames_count": n_frames,
                "chirps_count": receiver_radar.chirps_count,
                "t_inter_chirp": receiver_radar.t_inter_chirp,
                "n_tx": n_tx,
                "n_rx": n_rx,
                "n_adc": n_adc,
                "datatype": datatype,
                "f0_min": f0_min,
                "slope": slope,
                "bw": bw,
                "Tc": Tc,
                "TFFT": n_adc*ts,
                "T": T,
                "fs": receiver_radar.fs, "v": receiver_radar.v}

    # Tc is the relative time in the chirp
    # T is the absolute time: it is incremented at end of each chirp/ before start of new chirp and new frames
    Tc = T
    # compute can be set to False, when only interested in chirp statistics
    if raytracing_opt["compute"]:
        for frame_idx in range(n_frames):
            for chirp_idx in range(n_chirps):
                # classical ray-tracing now: starts from eye goes back to light
                t_start_chirp = receiver_radar.chirp_t_start(frame_idx, chirp_idx)
                T = Tc + t_start_chirp
                for rx_idx, rx_antenna in enumerate(receiver_radar.rx_antennas):
                    # FIXME: later these need to be parametrised
                    rx_hpf = 1e3
                    rx_lpf = 1e8
                    for target in targets:
                        for radar in raytracing_opt["radars"]:
                            for tx_idx, transmitter in enumerate(radar.transmitters):
                                # compute distance and then delta time from RX to TX
                                # to compute the received chirp frequency and phase
                                tx_x, tx_y, tx_z = transmitter.xyz
                                rx_x, rx_y, rx_z = rx_antenna.xyz
                                t_x, t_y, t_z = target.pos_t(T)
                                v = radar.v
                                # L = radar.L
                                distance = sqrt((tx_x - t_x)**2 + (tx_y - t_y)**2 + (tx_z - t_z)**2)
                                distance += sqrt((rx_x - t_x)**2 + (rx_y - t_y)**2 + (rx_z - t_z)**2)
                                time_of_flight = distance / v
                                f_rx = receiver_radar.TX_freq(T)
                                f_tx = transmitter.TX_freq(T-time_of_flight)
                                ph_tx = transmitter.TX_phase(T-time_of_flight, tx_idx) #0
                                # if transmitter.mimo_mode == "DDM":
                                #    ph_tx = 2*pi*transmitter.tx_conf["TX_phase_offsets"][tx_idx]*chirp_idx
                                #    # ph_tx = radar.TX_phase(T-time_of_flight)

                                if any(rx_hpf < np_abs(f_tx-f_rx) < rx_lpf):
                                    # skip computing the IF if all the frequencies are too low or too high
                                    # YIF += BB_IF(chirp_rx, chirp_tx, T, antenna_tx, antenna_rx, target, medium)

                                    YIFi = receiver_radar.BB_IF(f_rx, f_tx,
                                                                rx_hpf, rx_lpf,
                                                                ph_tx,
                                                                debug=debug)
                                    fif_max = max(abs(f_rx - f_tx))
                                    try:
                                        assert fif_max * 2 <= radar.fs
                                    except AssertionError:
                                        log_msg = "Nyquist will always prevail: " +\
                                            f"fs:{radar.fs:.2g} vs f_if:{fif_max:.2g}"
                                        if debug:
                                            print(f"!! Nyquist for target: {target}" +
                                                f"fif_max is: {fif_max} " +
                                                f"radar ADC fs is: {radar.fs}")
                                        if raytracing_opt["logging_level"] >= 40:
                                            raise ValueError(log_msg)
                                    YIF += YIFi

                                elif any(0 < np_abs(f_tx-f_rx)):
                                    raise ValueError("Corner Case for near DC frequency not handled")
                                else:
                                    raise ValueError("Corner Case")
                    # adc_cube[frame_idx, chirp_idx, tx_idx, rx_idx, :] = YIF
                    # times[frame_idx, chirp_idx, tx_idx, rx_idx, :] = T
                    # fix #4 - removed the tx_idx dimension as now we have TX which are potentially interferer
                    adc_cube[frame_idx, chirp_idx, rx_idx, :] = YIF
                    times[frame_idx, chirp_idx, rx_idx, :] = T
                    YIF, YIFi = None, None

                    # DONE 1 - HERE:
                    # add that TX is defined by TX slope and RX slope
                    # to allow for interferers
                    # increase n_tx to include possible interferers
                    # add code to assess if the current chirp has interference
                    # if not just `continue`

                    # TODO 2 - HERE:
                    # add code to include datatype as int16, int32
                    # and add code in the RSP to include Q15 format support

                    # TODO 3 - HERE:
                    # generate a new release 0.0.10-RC when this code is done and "works"

                    """ fix #4
                    for transmitters in raytracing_opt["radars"]:
                        for transmitter in transmitters:
                            f_tx = transmitter.TX_freq(T)
                            if f_tx:
                                # TODO 4: check that we enter here if and only if intereferer is active
                                # i.e. len of array is > 0
                                for TX_antenna in transmitter.tx_antennas:
                                    # chirp = (f0, slope, phi0, Tstart, Tend)
                                    # BB_IF = to compute freq delta, phase delta, 
                                    # define a new BB_IF - BB_IF_simple to BB_IF_slope_constant
                                    # F_RX = lambda T: slope(T-delta)
                                    # F_RX[T-delta<Tstart | T-delta>Tend] = 0 !!!!!!!
                                    # F_IF = F_RX - F_TX_local
                                    # BB_IF [F_IF>fmax] = 0 !!!!!!!!!!!!!!!

                                    # have provisions for lambda function for f_TX
                                    # remove amplitude from BB_IF and have it here 
                                    YIF += BB_IF(chirp_rx, chirp_tx, T, antenna_tx, antenna_rx, target, medium)

                                "" " !!!!!!!!!!!!!!!

                                first idea: when receiving cycle over ALL the active TX, each with slope, Fmin and phase and antenna distance ...
                                    so one active VCO for the receiver will then scan all active TX (either TDM, BPM, DDM & AND &&&&& interferers)
                                
                                how to get the phase and slope here?

                                NEED to have have phase in the TX info to allow also for artifacts in between chirps .. ?!?

                                how to have it work the same for normal TX and interferer TX
                                
                                
                                USE TX phase offset to do TX0 vs TX_n or interferer phase offset
                                
                                "" "

                    # TODO 4 - 
                    # naming convention to be added in READM:
                    # refer to PEP XXX:
                    #    UPPER CASE is for constants
                    #      tensors: t3, t4, ... tn_variable_name  for np arrays used as tensors of d>2
                    #      arr_name1_name2_values for 2D arrays
                    #      arr_name_values for 1D arrays
                    #      name_values for python list of values
                    # END OF updates for 0.0.10RC


                "" " #4
                for tx_i in range(n_tx):
                    phaser = 0
                    if mimo_mode == "TDM":
                        # in TDM TX transmit one after the other
                        # one chirp apart
                        # to T[0] is incremented by t_inter_chirp
                        T = Tc + (radar.t_inter_frame*frame_i) + \
                                (radar.t_inter_chirp*(chirp_i+1)*(tx_i+1))
                    elif mimo_mode == "DDM":
                        # in DDM all TX transmit at once
                        # so T[0] used to compute target distance
                        # does not change
                        T = Tc + (radar.t_inter_frame*frame_i) + \
                                (radar.t_inter_chirp*(chirp_i+1))
                        # T = array(T)
                        # here define the phaser from the passed
                        # configuration
                        phaser = 2*pi*TX_phase_offsets[tx_i]*chirp_i
                    else:
                        log_msg = f"MIMO mode: {mimo_mode} not valid"
                        if debug:
                            print(log_msg)
                        if raytracing_opt["logging_level"] >= 40:
                            raise ValueError(log_msg)
                    end of #4 - refactor for interferers
                    "" "


                    for rx_i in range(n_rx):
                        YIF = zeros(n_adc).astype(datatype)
                        for target in targets:
                            YIFi, fif_max = BB_IF(f0_min, slope, T,
                                                  radar.tx_antennas[tx_i],
                                                  radar.rx_antennas[rx_i],
                                                  target,
                                                  radar.medium,
                                                  TX_phase_offset=phaser,
                                                  radar_equation=radar_equation,  # noqa E501
                                                  datatype=datatype,
                                                  debug=debug)
                            # ensure Nyquist is respected
                            try:
                                assert fif_max * 2 <= radar.fs
                            except AssertionError:
                                log_msg = "Nyquist will always prevail: " +\
                                    f"fs:{radar.fs:.2g} vs f_if:{fif_max:.2g}"
                                if debug:
                                    print(f"!! Nyquist for target: {target}" +
                                          f"fif_max is: {fif_max} " +
                                          f"radar ADC fs is: {radar.fs}")
                                if raytracing_opt["logging_level"] >= 40:
                                    raise ValueError(log_msg)
                            YIF += YIFi

                            # TODO here: add the interferers contribution
                            "" "
                            for interferer in interferers:
                                # need to handle when the interferer starts
                                # the slope of TX interferer not being the same as the TX 
                                YIF_interfer_idx, fif_max = BB_IF(f0_min, slope, T, ...
                                    # BB_IF needs to now handle TX slope, RX slope and Interfer TX slope and
                                    # have a condition when f_IF > f_max f ==0 and then return the interference in the IF band only (0s elsewhere)
                                YIF += YIF_interfer_idx
                            "" "

                        if mimo_mode == "TDM":
                            adc_cube[frame_i, chirp_i, tx_i, rx_i, :] = YIF
                            times[frame_i, chirp_i, tx_i, rx_i, :] = T
                            YIF, YIFi = None, None
                        elif mimo_mode == "DDM":
                            # nth RX receives all the TXs at once
                            # TODO: optimise here to avoid multiple re-calculations of the same YIF
                            adc_cube[frame_i, chirp_i, 0, rx_i, :] += YIF
                        else:
                            log_msg = f"un supported mimo_mode: :{mimo_mode}"
                            if debug:
                                print(log_msg)
                            if raytracing_opt["logging_level"] >= 40:
                                raise ValueError(log_msg)
                        """

        baseband["adc_cube"] = adc_cube
        # T_fin = ((Tc +t_inter_chirp * NC) + t_inter_frame)*n_frames+ Tc
        baseband["times"] = times
        baseband["T_fin"] = T[-1]

    if debug:  # pragma: no cover
        Tc = bw/slope
        print("Generic observations about the simulation")
        print(f"Compute: {raytracing_opt['compute']}")
        print(f"Radar freq: {radar.tx_antennas[0].f_min_GHz} GHz")
        print("ADC samples #", n_adc)
        range_resolution = radar.medium.v/(2*radar.transmitter.bw)
        print("range resolution", range_resolution)

        if "Dres_min" in raytracing_opt:
            print("Range resolution target vs actual",
                  raytracing_opt["Dres_min"], range_resolution)
        else:
            print("Chirp Range resolution", range_resolution)
            print("FFT Range resolution", v*Tc/2/n_adc)

        if "mimo_mode" in radar.tx_conf:
            if radar.tx_conf == "DDM":
                try:
                    assert "TX_phase_offsets" in radar.tx_conf
                except AssertionError:
                    ValueError("In DDM , TX_phase_offsets must be provided")
                else:
                    for phi0 in radar.tx_conf["TX_phase_offsets"]:
                        try:
                            phi0 = float(phi0)  # force type to float
                            assert -1.0 < phi0 < 1.0
                        except AssertionError:
                            ValueError("TX_phase_offsets must be in [-1, 1]")
        print("Tc", Tc)
        print("T[-1]", T[-1])
        print("ts", ts)
        print("N adc per chirp", n_adc)
        print("t_interchirp", radar.t_inter_chirp)
        frame_time = n_adc*ts + radar.t_inter_chirp
        print("frame timing:", frame_time)
        print("simulation time", frame_time * n_frames)

        print("Dmax (ToF horizon)", v*Tc/2)
        print("Dmax as function fs", radar.fs*v/2/slope)
        print("Dres as function fs and NA", radar.fs*v/2/slope/n_adc)
        radar_lambda = radar.medium.v/radar.tx_antennas[0].f_min_GHz/1e9
        print(f"radar lambda: {radar_lambda}")
        vmax = 0
        vmax_ddm = 0
        if radar.t_inter_chirp > 0 and radar.chirps_count > 0:
            vmax = radar_lambda/4/radar.t_inter_chirp
            print(f"vmax :{vmax}")
            vref_IF = radar_lambda/2/radar.chirps_count/Tc
            print(f"speed resolution (within a frame of N chirps): {vref_IF}")

            if mimo_mode == "DDM":
                vmax_ddm = vmax / n_tx
        else:
            print("no speed info as only one chirp transmitted")
        # vres = lambda / 2 / N / Tc
        # vres_intrachirp =
        # vres_interframe =
        # vres_intraframe = radar_lambda/2/Tc
        # print(f"speed resolution intra-frame: {vres_intraframe}")

        print("---- TARGETS ---")
        for idx, target in enumerate(targets):
            x0, y0, z0 = target.pos_t()
            x1, y1, z1 = target.pos_t(t=T[-1])
            d0 = sqrt(x0**2 + y0**2 + z0**2)
            d1 = sqrt(x1**2+y1**2+z1**2)

            distance_covered = sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
            target_if = 2*slope*target.distance()/radar.medium.v

            from numpy import gradient
            vxt = gradient(target.xt(T), T[1]-T[0])
            vyt = array(gradient(target.yt(T), T[1]-T[0]))
            vzt = array(gradient(target.zt(T), T[1]-T[0]))
            vt = sqrt(vxt**2+vyt**2+vzt**2)
            vt_max = max(vt)
            vt_min = min(vt)
            vt_mean = mean(vt)

            if vt_max > 0 and radar.t_inter_chirp > 0 and radar.chirps_count > 0:
                try:
                    assert vt_max < vmax
                except AssertionError:
                    log_msg = "!!! Vmax exceeds unambiguous speed"
                    if debug:
                        print(log_msg)
                    if raytracing_opt["logging_level"] >= 40:
                        raise ValueError(log_msg)

            print(f"IF frequency for target[{idx}] is {target_if}, "
                  f"which is {target_if/radar.fs:.2g} of fs")

            if distance_covered > range_resolution:
                print("!!!!!! target[{idx}] covers more than one range: "
                      f"{distance_covered} vs {range_resolution}")
                print(f"initial position: {d0} and final position: {d1}")
            else:
                print(f"----- target[{idx}] covers less than one range: " +
                      f"{distance_covered} < {range_resolution} range res.")
            print(f"Range index: from {d0//range_resolution} "
                  f"to {d1//range_resolution}")

            if vt_max > vmax and radar.t_inter_chirp > 0 and radar.chirps_count > 0:
                print(f"!!!! vmax of target is: {vt_max} > " +
                      f"unambiguous speed: {vmax}")
            else:
                print(f"vmax of target is: {vt_max} < unambiguous" +
                      f" speed: {vmax}")
            print(f"vt_min: {vt_min}, vt_mean: {vt_mean}, vt_max:{vt_max}")
            if mimo_mode == "DDM":
                if vt_max > vmax_ddm:
                    print(f"!!!! vmax of target is: {vt_max} > DDM" +
                          f"unambiguous speed: {vmax_ddm}")

            print(f"End of simulation time: {T[-1]}")
    return baseband


def rt_points(radars, targets, receiver_radar,
              radar_equation=False,
              datatype=float32, debug=False,
              **raytracing_opt):
    """ raytracing with points

    Parameters
    ----------
    receiver_radar: Radar
        instance of Radar for which the BB cube is computed. One of the radars in the scene.
        (renamed in 0.0.10 from radar)
    targets: List[Target]
        list of targets in the Scene
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

    Raises
    ------
    ValueError
        if Nyquist rule is not upheld
    """
    n_frames = receiver_radar.frames_count
    # n_chirps is the # chirps each TX antenna sends per frame
    n_chirps = receiver_radar.chirps_count
    n_tx = len(receiver_radar.tx_antennas)
    n_rx = len(receiver_radar.rx_antennas)
    adc_samples_per_chirp = receiver_radar.receiver.adc_samples_per_chirp
    adc_sample_rate = receiver_radar.receiver.adc_sample_rate
    adc_sample_time = 1/adc_sample_rate
    bw = receiver_radar.bw
    adc_cube = zeros((n_frames, n_chirps, n_rx, adc_samples_per_chirp)).astype(datatype)
    f0_min, slope, Tc, T = 0, 0, 0, 0

    if radars is None:
        self._log.warning("future versions will not allow radars to be None")


    baseband = {"adc_cube": adc_cube,
                "frames_count": n_frames,
                "chirps_count": receiver_radar.chirps_count,
                "t_inter_chirp": receiver_radar.t_inter_chirp,
                "n_tx": n_tx,
                "n_rx": n_rx,
                "n_adc": adc_samples_per_chirp,
                "datatype": datatype,
                "f0_min": f0_min,
                "slope": slope,
                "bw": bw,
                "Tc": Tc,
                "TFFT": adc_samples_per_chirp*adc_sample_time,
                "T": T,
                "fs": receiver_radar.fs, "v": receiver_radar.v}
    if "compute" not in raytracing_opt:
        raytracing_opt["compute"] = False

    for frame_idx in range(n_frames):
        for chirp_idx in range(n_chirps):
            # TODO: need to make this code more flexibile to handle
            # cases where the chirp_start_time, chirp_slope and chirp_end_time
            # vary on chirp per chirp
            start_of_chirp = frame_idx * receiver_radar.t_inter_frame + \
                chirp_idx * receiver_radar.t_inter_chirp
            adc_times = arange(0, adc_samples_per_chirp*adc_sample_time,
                               adc_sample_time) + start_of_chirp
            adc_values = sample_all_rays(adc_times,
                                         radars,
                                         targets,
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
