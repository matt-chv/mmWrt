""" This module defines the main classes used to define a radar """

from numpy import all, array, exp, log2, ndarray, pi, random, real, select, stack, \
    sum, sqrt, where, zeros
from numpy import abs as np_abs, any as np_any, max as np_max
from numpy.typing import NDArray
from numpy import float32, float64, complex64, float16  

ERR_TARGET_T0 = "xyz<>xyzt(0)"
ERR_TFFT_lte_TC = "TFFT should be shorter than TC"
ERR_TOO_MANY_CHIRPS = "Too many chirps, not yet considered"

def scene_distance(targets_positions, antennas_pos):
    """ computes the distance between targets and antennas for each time point, using broadcasting

    Parameters
    ----------
    targets_positions: NDArray
        shape (n_targets, n_time_points, 3)
    antennas_pos: NDArray
        shape (n_antennas, 3)

    Returns
    -------
    distance_tx_target: NDArray
        shape (n_targets, n_time_points, n_antennas)
    """
    # Compute the distance from tx to target for each time point
    # targets_positions has shape (n_targets, n_time_points, 3)
    # tx_antennas_pos has shape (n_antennas, 3)
    # we want to compute the distance between each target and each antenna for each time point
    # this can be done using broadcasting by reshaping the arrays appropriately
    # we can reshape targets_positions to (n_targets, n_time_points, 1, 3) and tx_antennas_pos to (1, 1, n_antennas, 3)
    # then we can compute the difference and the distance using broadcasting
    # antennas_pos = antennas_pos.reshape(1, 1, antennas_pos.shape[0], 3)
    # targets_positions = targets_positions.reshape(targets_positions.shape[0],
    #                                              targets_positions.shape[1], 1,
    #                                              targets_positions.shape[2])
    diff = targets_positions - antennas_pos  # 2000 targets * 1024 samples  operations
    distance = sqrt(sum(diff * diff, axis=-1))
    return distance

class Target():
    def __init__(self, x=0.0, y=0.0, z=0.0,
                 xt=None, yt=None, zt=None,
                 rcs_f=lambda f: 1,
                 target_type="point"):
        """ Initializes a target, ease of use vs simplicity at definition

        Parameters
        ----------
        x: float
            x-coordinate
        y: float
            y-coordinate
        z: float
            z-coordindate
        xt: lambda
            x-coordinate in time
        yt: lamda
            y-coordinate in time
        zt: lambda
            z-coordinate in time
        rcs_f: lambda
            lambda of rcs as function of frequency
        target_type: str
            point or volume

        Raises
        ------
        ValueError
            when definition of xyz(0) is not xyz and xyz is different than 0

        Examples
        --------
        define a target at (x,y,z)=(0,0,0)
        > target = Target()
        define a target at (x,y,z)=(10,0,0)
        > target = Target(10)
        define a target with a position in time x(t) = 10 + 10*t
        > target = Target(xt= lambda t: 10 + 10*t)
        """
        self.x = x
        self.y = y
        self.z = z

        if xt is not None:
            if x != 0:
                try:
                    assert x == xt(0)
                except AssertionError:
                    raise ValueError(ERR_TARGET_T0)
            else:  # pragma: no cover
                self.x = xt(0)
            self.xt = xt
        else:
            # seems that
            # self.xt = lambdat t: x
            # does not always work to return an array when fed an array
            # so writting it lambda t: 0*t + x
            self.xt = lambda t: 0*t + x

        if yt is not None:
            if y != 0:
                try:
                    assert y == yt(0)
                except AssertionError:
                    raise ValueError(ERR_TARGET_T0)
            else:  # pragma: no cover
                self.y = yt(0)
            self.yt = yt
        else:
            self.yt = lambda t: 0*t + y

        if zt is not None:
            if z != 0:
                try:
                    assert z == zt(0)
                except AssertionError:
                    raise ValueError(ERR_TARGET_T0)

            else:  # pragma: no cover
                self.z = zt(0)
            self.zt = zt
        else:
            self.zt = lambda t: 0*t + z

        try:
            assert isinstance(self.xt(array([0, 1, 2])), ndarray)
            assert isinstance(self.yt(array([0, 1, 2])), ndarray)
            assert isinstance(self.zt(array([0, 1, 2])), ndarray)
        except Exception as ex:
            print(type(self.xt(array([0, 1, 2]))))
            print("ERRR", str(ex))
            raise ValueError("position functions need to be defined to return list not scalar, likely something like xt = lambda t: 0 instead of xt = lambda t: 0*t was used in definition")
        self.rcs_f = rcs_f
        self.target_type = target_type

    def distance(self, target=None, t=0):
        x0, y0, z0 = self.pos_t(t)
        if target is None:
            x1, y1, z1 = 0, 0, 0
        else:
            x1, y1, z1 = target.pos_t(t)
        dist = sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
        return dist

    def pos_t1(self, t: NDArray) -> NDArray:
        # x0, y0, z0 = self.x, self.y, self.z
        x_positions = self.xt(t)
        y_positions = self.yt(t)
        z_positions = self.zt(t)

        position_t = stack((x_positions, y_positions, z_positions), axis=1)

        return position_t


    def pos_t(self, t: NDArray) -> NDArray:
        # x0, y0, z0 = self.x, self.y, self.z
        x_positions = array(self.xt(t))
        y_positions = array(self.yt(t))
        z_positions = array(self.zt(t))
        position_t = stack((x_positions, y_positions, z_positions), axis=0)
        return position_t

    def __str__(self):
        return f"x0:{self.x}, y0:{self.y}, z0:{self.z}"

    def rcs(self, f):
        return self.rcs_f(f)


class Antenna:
    def __init__(self, x=0.0, y=0, z=0, angle_gains_db10=zeros((360, 360)),
                 f_min_GHz=60, f_max_GHz=64, freq_gains_db10=zeros(4)):
        """ initialize antenna position and gains.
        Defaults to isotropic radiation pattern

        Parameters
        ----------
        x: float
            x coordinate
        y: float
            y coordinate
        z: float
            z coordinate
        angle_gains_db10: numpy array
            2D array of (azimuth, elevation) gains in dB
        f_min_GHz: float
            min frequency in GHz for which antennas is characterised
        f_max_GHz: float
            min frequency in GHz for which antennas is characterised
        freq_gains_db10: numpy array
            linearly spaced antenna gains between f_min and f_max
        """
        self.x = x
        self.y = y
        self.z = z
        self.xt = lambda t: 0*t + x
        self.yt = lambda t: 0*t + y
        self.zt = lambda t: 0*t + z
        self.xyz = (x, y, z)
        self.angle_gains_db10 = angle_gains_db10
        self.f_min_GHz = f_min_GHz
        self.f_max_GHz = f_max_GHz
        self.freq_gains_db10 = freq_gains_db10
        self.look_up = (f_max_GHz-f_min_GHz)/freq_gains_db10.shape[0]

    def freq_gain_db10(self, freq):
        """ antenna gain at given frequency

        Parameters
        ----------
        freq: float
            frequency in Hertz

        Returns
        -------
        gain_dB: float
            gain in dB

        Raises
        ------
        ValueError
            if freq is too low
        """
        freq_GHz = freq / 1e9
        try:
            assert freq_GHz > self.f_min_GHz
        except Exception as ex:  # pragma: no cover
            print(f"{str(ex)}freq_GHz, self.f_min_GHz",
                  freq_GHz, self.f_min_GHz)
            raise ValueError("freq")
        assert freq_GHz < self.f_max_GHz
        idx = int((freq_GHz-self.f_min_GHz)*self.look_up)
        gain_db10 = self.freq_gains_db10[idx]
        return gain_db10

    def gain(self, azimuth, elevation, freq):
        """ computes total antenna gain over elevation, aziumth and frequency

        Parameters
        ----------
        azimuth: float
            between -pi and pi value
        elevation: float
            between -pi and pi value
        freq: float
            frequency at which antenna gain needs to be calculated

        Returns
        -------
        overall_gain: float
            antenna gain at freq and given direction
        """
        azimuth_deg = int((azimuth+pi)*180/pi) % 360
        elevation_deg = int((elevation+pi)*180/pi) % 360
        gain_angle_db = self.angle_gains_db10[azimuth_deg, elevation_deg]
        gain_freq = self.freq_gain_db10(freq)
        overall_gain = 10**gain_angle_db * 10**gain_freq
        return overall_gain

    def position_in_time(self, t: NDArray) -> NDArray:

        x_positions = array(self.xt(t))
        y_positions = array(self.yt(t))
        z_positions = array(self.zt(t))

        position_t = stack((x_positions, y_positions, z_positions), axis=1)

        return position_t
    
    def __str__(self):
        return f"x,y,z: ({self.x}, {self.y}, {self.z})"


class Receiver():
    """ Need to split this into RX RF (antennas locations)
    MIXER for RX, TX to IF
    IF filter (HPF for DC and LPF for aliasing + removal of the MIXER high freq components)"""
    def __init__(self,
                 adc_sample_rate=4e2,
                 antennas=(Antenna(),),
                 max_adc_buffer_size=1024,
                 max_fs=25e6,
                 adc_samples_per_chirp=0,
                 config=None,
                 debug=False):
        fs = adc_sample_rate
        self.fs = fs
        self.adc_sampling_frequency = fs
        self.adc_sample_rate = adc_sample_rate
        self.antennas = antennas
        self.max_adc_buffer_size = max_adc_buffer_size
        self.adc_samples_per_chirp = adc_samples_per_chirp
        self.n_adc = adc_samples_per_chirp
        n_adc = adc_samples_per_chirp
        self.number_adc_samples = n_adc
        self.adc_samples_per_chirp = n_adc
        self.rx_high_pass_freq = 1e2
        self.rx_low_pass_freq = 1e8

        try:
            assert fs < max_fs
        except AssertionError:
            if debug:
                print(f"fs:{fs} > max_fs: {max_fs}")
            raise ValueError("ADC sampling value must stay below max_fs")
        return


class Transmitter():
    """
    Attributes:
    -----------
    tx_start_time: float
        time offset for the start of the first chirp
    tx_on_times: List[float]
        list of 2-uples of start/stop timess for each chirp transmitted
    tx_slopes: List[float]
        list of slopes for each chirp transmitted
    """
    chirps_count = 1
    frames_count = 1
    tx_start_time = 0.0
    tx_on_times = []
    def __init__(self,
                 chirp_start_freq: float = 60e9,
                 chirp_slope: float = 1e12,
                 chirp_end_time: float = 1e-6,
                 antennas=[Antenna()],
                 t_inter_chirp=0.0,
                 chirps_count=1,
                 t_inter_frame=0.0,
                 frames_count=1,
                 conf=None):
        """Transmitter class models a radar transmitter

        Parameters
        ----------
        chirp_start_freq: float
            start frequency of the chirp
        slope: float
            the slope of the linearly growing chirp frequency in Hz/s
        chirp_end_time: float
            time when the ramp ends (i.e. when f=f0_min + bw) in s
        antennas: List[Antenna]
            transmitter Antennas instances
        t_inter_chirp: float
            time increment between two TX antennas sending a chirp
        chirps_count: int
            The # chirps each TX antenna sends per frame
        t_inter_frame: float
            time increment between end of last chirp in frame N-1 and
            first chirp in frame N (offset on top of
            t_inter_chirp). If t_interframe==0, then there will be a
            single t_inter_chirp offset.
        frames_count: int
            The number of iterations where each TX antennas send chirps_count
        conf: dict
            additional optional parameters (reserved for future usage)
            includes mimo_mode = [TDM, DDM]
        """
        self.chirp_slope = chirp_slope
        self.slope = chirp_slope
        slope = chirp_slope

        if chirp_slope < 1e8:
            print("chirp_slope is low", chirp_slope)
        bw = chirp_slope * chirp_end_time
        self.f0_min = chirp_start_freq
        self.chirp_start_freq = chirp_start_freq
        # self.slope = slope
        
        self.t_inter_chirp = t_inter_chirp
        self.chirps_count = chirps_count
        self.antennas = antennas
        if t_inter_frame == 0:
            self.t_inter_frame = t_inter_chirp
        else:
            assert t_inter_frame >= t_inter_chirp
            self.t_inter_frame = t_inter_frame
        self.frames_count = frames_count
        self.bw = bw
        if bw is not None and chirp_end_time is None:
            log_msg = "DeprecationWarning: chirp defined with bw instead of ramp_end_time about to be removed !!!!"
            print(log_msg)
            ramp_end_time = bw/slope
            self.ramp_end_time = ramp_end_time
        else:
            self.ramp_end_time = chirp_end_time
            self.chirp_end_time  =chirp_end_time
        self.conf = conf
        self.mimo_mode = "TDM"
        if conf is None:
            self.conf = {"mimo_mode": "TDM"}
        else:
            if "mimo_mode" in self.conf:
                assert self.conf["mimo_mode"] in ["TDM", "DDM"]
                self.mimo_mode = self.conf["mimo_mode"]
            else:
                self.conf["mimo_mode"] = "TDM"
            if "tx_start_time=" in conf:
                self.tx_start_time = conf["tx_start_time"]
        """if self.mimo_mode == "TDM":
            # compute the tx_on_times for each chirp
            for frame_idx in range(frames_count):
                for chirp_idx in range(chirps_count):
                    chirp_start = self.chirp_t_start(frame_idx, chirp_idx)
                    chirp_end = chirp_start + ramp_end_time
                    self.tx_on_times.append((chirp_start, chirp_end))
                return"""

    def frame_t_start(self, frame_idx,
                      dither_std=0.0, dither_seed=42,
                      dithering=False):
        """ returns the time offset at which frame frame_idx starts
        Default function for TDM mode with some dithering, needs to be overridden
        for more complex modes

        Parameters
        ----------
        frame_idx: int
            index of the frame

        Returns
        -------
        t_start_frame: float
            time at which the frame starts relative to first start of first frame
        """
        t_start_dithered = 0
        if dithering:
            random.seed(dither_seed + frame_idx)
            t_start_dithered += random.normal(0, dither_std)
        t_start_frame = self.tx_start_time + frame_idx * self.t_inter_frame + t_start_dithered
        return t_start_frame

    def chirp_t_start(self, frame_idx, chirp_idx, dither_std=0.0,
                      dither_seed=42.0, dithering=False) -> float:
        """ returns the time offset at which frame frame_idx starts
        Default function for TDM mode with some dithering,
        needs to be overridden for more complex modes

        Parameters
        ----------
        frame_idx: int
            index of the frame
        chirp_idx: int
            index of the chirp
        dither_std: float
            std of the chirp start time dithering
        dither_seed: float
            see for the random number generator
        dithering: bool
            if True dithering is enabled

        Returns
        -------
        t_start: float
            time offset in seconds
        """
        t_start_dithered = 0
        if dithering:
            random.seed(dither_seed + frame_idx*self.chirps_count + chirp_idx)
            t_start_dithered += random.normal(0, dither_std)
        t_start = self.frame_t_start(frame_idx) + \
            chirp_idx * self.t_inter_chirp + t_start_dithered
        return t_start

    def TX_freqs(self, times: NDArray, tx_idx: int = -1) -> NDArray[float64]:
        """ Default transmitter freq over time interval, intended to be overridden
        by more complex models like with subframes or complex stepped FMCW cases
        especially used when dealing with multiple simultaneous transmitters for interferene modelling
        this implementation assumes a simple linear FMCW chirp with constant interchirp time (no dithering)
        and a single frame (i.e. no change of slope on per frame/ subframe/ chirp basis).

        Parameters
        ----------
        times: NDArray
            time stamps in seconds

        Returns
        -------
        freq_o_times: NDArray[float64]
            2D of TX frequency slopes over time, will have size 0 if TX is not `ON` at anytime between t_start and t_end
            axis 1 is chirp, axis 0 is absolute time (i.e. not referred to start of chirp).
        """

        """#freq_o_times = zeros((2, times.shape[0]))
        freq_o_times = zeros((1, times.shape[0]))
        # freq_o_times[0, :] = times

        def piecewise_chirp(t):
            conditions = []
            functions = []
            for frame_idx in range(self.frames_count):
                for chirp_idx in range(self.chirps_count):
                    t_start_chirp = self.t_inter_frame*frame_idx + \
                        self.t_inter_chirp*chirp_idx
                    end_chirp = t_start_chirp + self.ramp_end_time
                    conditions.append(((t >= t_start_chirp) & (t <= end_chirp)))
                    " "" IMPORTANT NOTE for future (self?-)reader(s)
                    due to late binding in Python when using lambda functions inside a loop. 
                    Inside the loop, the lambda function captures the variable t_start_chirp by reference, not by value. 
                    So, when calling freq function later , the lambda function only refers to the final value of t_start_chirp, 
                    which is the chirp start time of the last chirp of the last frame (after the loop ends).

                    To ensure the value is used keep the following 
                        tx_freq = lambda t_adc, t_start_chirp=t_start_chirp: self.f0_min + (t_adc-t_start_chirp)*self.slope
                    which passes t_start_chirp as a default argument to the lambda, ensuring the value of t_start_chirp is captured at the time the lambda is created.

                    failling to do so and writting something like below will miserably fail and take forever to debug
                    tx_freq = lambda t_adc: self.f0_min + (t_adc-t_start_chirp)*self.slope
                    " ""  # noqa E501
                    tx_freq = lambda t_adc, t_start_chirp=t_start_chirp: \
                        self.f0_min + (t_adc-t_start_chirp)*self.slope

                    functions.append(tx_freq)
                    """
            # numpy select used to define piecewise chirp TX frequencies
            # could be extended for different chirps slopes / non linear slopes
            # FIXME: move the creation of conditions and functions to __init__
            # then in this function just do the select
            # chirp_frequencies = select(conditions,
            #                           [f(t) for f in functions],
            #                           default=0)
            # return chirp_frequencies
        self.__tx_freq_t()
        freq_o_times = self.chirp_frequencies(times)
        # freq_o_times = piecewise_chirp(times)

        return freq_o_times

    def TX_phases(self, times: NDArray, tx_idx: int = -1) -> NDArray[float64]:
        """ computes the TX phase at given times, default TDM: 0

        Parameters
        ----------
        times: NDArray
            time stamps in seconds
        tx_idx: int
            if tx_idx == -1 then gives the VCO phase, used for RX 
            if tx_idx > 0 then gives the VCO phase for specific PA

        Returns
        -------
        phase_o_times: NDArray[float64]
            the phase, by default in TDM always 0
        """
        tx_phases = zeros(times.shape)
        return tx_phases

    def __tx_freq_t(self):
        """ should be called every time self.tx_start_time is updated"""
        from numpy import arange, tile, repeat, piecewise
        chirp_idx = tile(arange(0, self.chirps_count), self.frames_count)
        frame_idx = repeat(arange(0, self.frames_count), self.chirps_count)
        start_time = self.t_inter_frame*frame_idx + \
                        self.t_inter_chirp*chirp_idx + self.tx_start_time
        end_time = start_time + self.ramp_end_time
        freqs = [lambda t_adc, t_start_chirp=t_start_chirp:
                 self.f0_min + (t_adc-t_start_chirp)*self.slope for t_start_chirp in start_time]
        def __temp2(t_adc):
            conditions = [(t_adc>=start_t) & (t_adc<=end_t) for start_t, end_t in zip(start_time, end_time)]
            return piecewise(t_adc, conditions, freqs)

        self.chirp_frequencies = __temp2


class TransmitterDDM(Transmitter):
    def __init__(self,
                 chirp_start_freq=60e9,
                 chirp_slope=None,
                 chirp_end_time=None,
                 slope_MHz_us=None,
                 bw=4e9,
                 antennas=[Antenna()],
                 t_inter_chirp=0.0,
                 chirps_count=1,
                 t_inter_frame=0.0,
                 frames_count=1,
                 conf=None):
        super().__init__(chirp_start_freq,
                         chirp_slope,
                         chirp_end_time,
                         antennas,
                         t_inter_chirp,
                         chirps_count,
                         t_inter_frame,
                         frames_count,
                         conf)
        if "TX_phase_offset" in conf:
            raise ValueError("TX_phase_offset deprecated replaced by TX_phaser_slopes")
        assert "TX_phaser_slopes" in conf
        assert len(conf["TX_phaser_slopes"]) == len(antennas)
        self.TX_phaser_slopes = conf["TX_phaser_slopes"]
        
    def TX_freqs(self, times: NDArray, tx_idx=-1) -> NDArray[float64]:
        """ Default implementation for DDM"""
        # if tx_idx==-1: return VCO chirp (used for any RX )
        # if tx_idx >= 0: return the TX antenna one (used for specific TX antenna phase)
        tx_phases = zeros(times.shape)
        return tx_phases

    def TX_phases(self, times: NDArray, tx_idx: int = -1) -> NDArray[float64]:
        """ Default implementation for DDM

        Parameters
        ----------
        times: NDArray
            the time stamps at which the TX phaser has to be evaluated
        tx_idx: int
            the index of the antenna
        Returns
        -------
        Example
        -------
        """
        tx_phases = zeros(len(times))
        def _piecewise_chirp(t, tx_idx):
            conditions = []
            functions = []
            for frame_idx in range(self.frames_count):
                for chirp_idx in range(self.chirps_count):
                    t_start_chirp = self.t_inter_frame*frame_idx + self.t_inter_chirp*chirp_idx
                    end_chirp = t_start_chirp + self.ramp_end_time
                    conditions.append(((t >= t_start_chirp) & (t <= end_chirp)))
                    tx_phase = lambda t_adc, chirp_idx=chirp_idx: pi*self.TX_phaser_slopes[tx_idx]*chirp_idx
                    functions.append(tx_phase)
            chirp_phases = select(conditions,
                                  [f(t) for f in functions],
                                  default=0)
            return chirp_phases
        if (tx_idx >= 0) and (tx_idx < len(self.antennas)):
            tx_phases = _piecewise_chirp(times, tx_idx)
        elif tx_idx == -1:
            # provision for the VCO phase to be modelled in more refined models
            pass
            #return tx_phases
        return tx_phases

class Medium:
    def __init__(self, v=3e8, L=0, name="void"):
        """ initialises the medium where demo runs

        Parameters
        ----------
        v: float
            speed of light in the given medium, defaults to 3e8 for void
        L: float
            attenuation in dB/m in given medium, defaults to 0 for void
        name: str
            name of the given medium, defaults to void
        """
        self.v = v
        self.L = L
        self.name = name
        if name == "void":
            # Ensuring consistency with physics
            assert v == 3e8
            assert L == 0


class Radar:
    def __init__(self, transmitter=Transmitter(), receiver=Receiver(),
                 medium=Medium(), adc_po2=False, debug=False):
        """ Defines a Radar instance from Transmitter class, Receiver class,
        Medium class
        and allows overriding the number of adc samples.

        Parameters
        ----------
        transmitter: Transmitter()
            definition of the transmitter chain used
        receiver: Receiver()
            definition of the receiver chain used
        medium: Medium()
            definition of the medium used (currently only uniform medium)
        adc_po2: bool
            if true sets number of ADC to next power of 2 from current value
        debug: bool
            if True: prints error message
            if False: exception

        Raises
        ------
        ValueError
            if ADC buffer exceeds maximum buffer size
        """
        self.transmitter = transmitter
        self.receiver = receiver
        self.rx_antennas = receiver.antennas
        self.tx_antennas = transmitter.antennas

        self.frames_count = transmitter.frames_count
        # issue #5 - frames_count renamed to total_number_frames
        # chirps_count to total_number_chirps
        self.total_number_frames = self.frames_count
        self.total_chirps_per_frame = transmitter.chirps_count
        self.n_adc = receiver.n_adc
        self.number_adc_samples = self.n_adc
        self.fs = receiver.fs
        self.bw = transmitter.bw
        self.tx_conf = transmitter.conf
        self.mimo_mode = transmitter.conf["mimo_mode"]
        self.TX_freqs = transmitter.TX_freqs
        self.TX_phases = transmitter.TX_phases
        self.chirp_t_start = transmitter.chirp_t_start
        if self.n_adc == 0:
            # self.n_adc = int(transmitter.bw * receiver.fs / transmitter.slope)
            if self.n_adc == 0:  # pragma: no cover
                log_msg = f"nadc updated to 0: {transmitter.bw:.2g}" +\
                    f"{receiver.fs:.2g}= {transmitter.bw*receiver.fs:.2g}" +\
                    f" /  {transmitter.slope:.2g}"
                # , transmitter.bw, receiver.fs, transmitter.slope
                
            if debug:  # pragma: no cover
                print("updating NADC from 0 to:", self.n_adc)
                raise ValueError(log_msg)
        t_fft = receiver.n_adc / receiver.fs
        t_chirp = transmitter.ramp_end_time  # transmitter.bw / transmitter.slope

        bw_adc = self.n_adc*transmitter.slope/receiver.fs

        if debug:  # pragma: no cover
            if bw_adc < 0.8 * transmitter.bw:
                print(f"! BW ADC: {bw_adc:.2g} << chirp: {transmitter.bw:.2g}")
            print(f"Bandwidth in chirp: {transmitter.bw:.2g}")
            print(f"Bandwidth in ADC buffers: {bw_adc:.2g}")
        if self.n_adc < 8:  # pragma: no cover
            print("!!!! ADC # low", self.n_adc)
            print("BW", transmitter.bw)
            print("BW GHz", transmitter.bw/1e9)
            print("K", transmitter.slope)
            print("K/1e12", transmitter.slope/1e12)
            print("TC", transmitter.bw / transmitter.slope)
            print("N_ADC", transmitter.bw / transmitter.slope * receiver.fs)

        if adc_po2:
            self.n_adc = 2 ** int(log2(self.n_adc))
            n_adc = self.n_adc
            assert n_adc / receiver.fs * transmitter.slope < transmitter.bw
        self.f0_min = transmitter.f0_min
        self.slope = transmitter.slope
        self.t_inter_chirp = transmitter.t_inter_chirp
        self.chirps_count = transmitter.chirps_count
        self.t_inter_frame = transmitter.t_inter_frame
        self.frames_count = transmitter.frames_count
        self.v = medium.v
        self.medium = medium
        self.bw = transmitter.bw
        # FIXME: moves this to simulation level
        # __range_bin: deprecated as relies on c for compute
        # __c = 3e8
        # self.range_bin_deprec = receiver.fs*__c/2/self.slope/self.n_adc

        if all(self.rx_antennas[0].angle_gains_db10 == 0):
            for idx, _ in enumerate(self.rx_antennas):
                self.rx_antennas[idx].f_min_GHz = self.f0_min/1e9
                self.rx_antennas[idx].f_max_GHz = (self.f0_min + self.bw)/1e9
            if debug:  # pragma: no cover
                print("rx fmin (GHz)", self.rx_antennas[idx].f_min_GHz)
                print("rx fmax (GHz)", self.rx_antennas[idx].f_max_GHz)

        if all(self.tx_antennas[0].angle_gains_db10 == 0):
            for idx, _ in enumerate(self.tx_antennas):
                self.tx_antennas[idx].f_min_GHz = self.f0_min/1e9
                self.tx_antennas[idx].f_max_GHz = (self.f0_min + self.bw)/1e9
            if debug:  # pragma: no cover
                print("tx fmin", self.tx_antennas[idx].f_min_GHz)
                print("tx fmax", self.tx_antennas[idx].f_max_GHz)
        try:
            assert t_fft <= t_chirp
        except AssertionError:
            print(f"T_FFT: {t_fft:.2g}")
            print(f"T_C: {t_chirp:.2g}")
            if debug:  # pragma: no cover
                pass
            else:
                raise ValueError(ERR_TFFT_lte_TC)

        try:
            assert self.n_adc < receiver.max_adc_buffer_size
        except AssertionError:
            if debug:  # pragma: no cover
                print(f"buffer size: {self.n_adc} > " +
                      f"vs max buffer size: {receiver.max_adc_buffer_size}" +
                      f"ratio: {self.n_adc/receiver.max_adc_buffer_size}")
            raise ValueError("ADC buffer overflow")
        return

    def BB_IF(self, adc_times, f_rx,
              debug=False) -> NDArray:
        """ Simplified mixer and IF filtering to model 
        intermediate frequency function ADC sampling.
        provisions to account for interferer radars.

        Parameters
        ----------
        Tc: NDArray[float32]
            the relative time to start of chirp in (s)
        f_rx: NDArray[float32]
            the local TX chirp transmit frequency in (Hz)
        f_tx: NDArray[float32]
            the frequency at which chirp was trasmitted in (Hz)
        rx_hpf: float
            high-pass filter - cutting off DC component. brikwall so far. (Hz)
        rx_lpf: float
            low-pass filter - cutting off beyond Nyquist. (Hz)
        tx_phase_offset: float
            used especially for DDMA modulation in radian
        debug: bool
            if True displays debug information
        Returns
        -------
        YIF:
            the ADC values in complex, shape is (1,adc_times.shape)
        Example
        -------
        >> BB_IF(array([0,3e-7,6.6e-7,1.e-6]),
                array([6e10,6.0003e10,6.0006e+10,6.001e10]),
                array([6.1e10,6.1003e10,6.1006e10,6.101e10]),
                rx_hpf=1e3, rx_lpf=1e8)
        << [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
        >> BB_IF(array([0, 1.3e-7, 2.6e-7, 4e-7,
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
                        6.08e9,6.086e9,6.093e9,6.1e9]),
                    rx_hpf=1e3, rx_lpf=1e8)
        << [ 1. +0.00000000e+00j,  0.68454711+7.28968627e-01j,
            -0.06279052+9.98026728e-01j, -0.80901699+5.87785252e-01j,
            -0.98228725-1.87381315e-01j, -0.53582679-8.44327926e-01j,
            0.30901699-9.51056516e-01j,  0.90482705-4.25779292e-01j,
            1.        -1.13310778e-15j,  0.30901699+9.51056516e-01j,
            -0.30901699+9.51056516e-01j, -0.96858316+2.48689887e-01j,
            -0.80901699-5.87785252e-01j, -0.12533323-9.92114701e-01j,
            0.63742399-7.70513243e-01j,  1.        -2.26621556e-15j]
        """

        f_tx = self.TX_freqs(adc_times)

        f_if = f_tx-f_rx
        return f_if

    def adc_sampling(self, f_if,
                     adc_times,
                     ph_rx,
                     time_of_flight,
                     radar_equation=False,
                     datatype=complex64,
                     debug=False):
        ph_tx = self.TX_phases(adc_times)
        rx_high_pass_freq = self.receiver.rx_high_pass_freq
        rx_low_pass_freq = self.receiver.rx_low_pass_freq

        if_filter = (rx_high_pass_freq < abs(f_if)) & \
            (abs(f_if) < rx_low_pass_freq)
        if debug:
            print("f_mix before if_filter", f_if)
            print(rx_high_pass_freq)
            print(rx_high_pass_freq < f_if)
            print(rx_low_pass_freq)
            print(f_if < rx_low_pass_freq)

        f_if[~(if_filter)] = 0
        if debug:
            print("f_if after if_filter", f_if)
        YIF = zeros(f_if.shape)
        if debug:
            print("if_filter !!!!!", if_filter)
            print("f_if", f_if)

        if np_any(if_filter):
            # skip computing the IF if all the frequencies are too low or too high
            # YIF += BB_IF(chirp_rx, chirp_tx, T, antenna_tx, antenna_rx, target, medium)

            # adc_samples = BB_IF_v2(tr_chirp, fif, rx_high_pass_freq, rx_low_pass_freq,
            #                       ph_tx, debug=debug)
            Tc = adc_times - adc_times[0]
            # print("Tc = adc_times-adc_times[0]", Tc)
            #IF_filter = ((rx_high_pass_freq <= np_abs(f_if)) &
            #            (np_abs(f_if) <= rx_low_pass_freq))

            YIF = where(if_filter,
                        exp(2 * pi * 1j * (f_if) * Tc + 1j*(ph_tx-ph_rx) +
                            2 * pi * 1j * time_of_flight*self.transmitter.chirp_start_freq -  # this is the important term for speed measure
                            2 * pi * 1j * self.transmitter.chirp_slope*time_of_flight**2),
                        YIF)
            # print("YIF B4 summing", YIF)
            print("radar equation", radar_equation)
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

            YIF = sum(YIF, axis=1)

            if datatype in [float64, float32, float16]:
                YIF = real(YIF)
            elif datatype in [complex64, complex]:
                pass
            elif datatype in [int64, int32, int16]:
                YIF = int(real(YIF)/max(real(YIF)) * (2**(8*datatype().nbytes-1)-1))

            # print("YIF A8 summing and datatype", YIF)
            fif_max = np_max(f_if)
            try:
                assert fif_max * 2 <= self.fs
            except AssertionError:
                log_msg = "Nyquist will always prevail: " +\
                    f"fs:{self.fs:.2g} vs f_if:{fif_max:.2g}"
                if debug:
                    raise ValueError(log_msg)

            """YIF = zeros(Tc.shape)
            IF_filter = ((rx_hpf <= abs(f_rx-f_tx)) &
                        (abs(f_rx-f_tx) <= rx_lpf))
            YIF = where(IF_filter,
                        exp(2 * pi * 1j * (f_rx-f_tx) * Tc + 1j*tx_phase_offset),
                        YIF)
            return YIF"""
        return YIF
