""" This module defines the main classes used to define a radar """

import logging
import numpy as np
from numpy import all, array, exp, log2, ndarray, pi, random, real, select, stack, \
    sum, sqrt, where, zeros
from numpy import abs as np_abs, any as np_any, max as np_max
from numpy.typing import NDArray
from numpy import float32, float64, complex64, float16, int16, int32, int64
import numpy as np
from typing import Literal, Optional, TypeVar

ERR_TARGET_T0 = "xyz<>xyzt(0)"
ERR_TFFT_lte_TC = "TFFT should be shorter than TC"
ERR_TOO_MANY_CHIRPS = "Too many chirps, not yet considered"

ADCType = TypeVar("ADCType", complex64, float32, float64)


def two_way_range(tx_antennas_positions: NDArray,
                  scatterer_positions: NDArray,
                  rx_antennas_positions: NDArray) -> NDArray:
    """ Computes the two way distance from TX antenna to scatterer back to RX antenna

    Parameters:
    ----------
    rx_antennas_positions:
        [T, RX, 3]
    scatterer_positions:
        [T, S, 3]
    tx_antennas_positions:
        [T, TX, 3]

    Returns:
    --------
    two_way_distance:
        [T, TX, S, RX]
    """
    from numpy.linalg import norm

    # Broadcast to [T, TX, S, RX, 3]
    rx = rx_antennas_positions.shape[1]
    s = scatterer_positions.shape[1]
    tx = tx_antennas_positions.shape[1]
    t = rx_antennas_positions.shape[0]
    rx_broadcast = rx_antennas_positions[:, None, None, :, :]   # [T, 1, 1, RX, 3]
    scatterer_broadcast = scatterer_positions[:, None, :, None, :]   # [T, 1, S, 1, 3]
    tx_broadcast = tx_antennas_positions[:, :, None, None, :]   # [T, TX, 1, 1, 3]

    # Leg distances — each [T, TX, S, RX]
    r_tx = norm(scatterer_broadcast - tx_broadcast, axis=-1)
    r_rx = norm(scatterer_broadcast - rx_broadcast, axis=-1)

    # Two-way range, then flatten → [T, I*J*K]
    two_way_distance = (r_tx + r_rx)  # .reshape(t, rx*s*tx)
    return two_way_distance


class Scatterer():
    def __init__(self, x=0.0, y=0.0, z=0.0,
                 xt=None, yt=None, zt=None,
                 rcs_f=lambda f: 1,
                 scatterer_type="point"):
        """ Initializes a scatterer, ease of use vs simplicity at definition

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
        scatterer_type: str
            point or volume

        Raises
        ------
        ValueError
            when definition of xyz(0) is not xyz and xyz is different than 0

        Examples
        --------
        define a scatterer at (x,y,z)=(0,0,0)
        > scatterer = Scatterer()
        define a scatterer at (x,y,z)=(10,0,0)
        > scatterer = Scatterer(10)
        define a scatterer with a position in time x(t) = 10 + 10*t
        > scatterer = Scatterer(xt= lambda t: 10 + 10*t)
        """
        self._log = logging.getLogger(self.__class__.__qualname__)
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
        self.scatterer_type = scatterer_type

    def distance(self, scatterer=None, t=0):
        x0, y0, z0 = self.pos_t(t)
        if scatterer is None:
            x1, y1, z1 = 0, 0, 0
        else:
            x1, y1, z1 = scatterer.pos_t(t)
        dist = sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
        return dist

    def pos_t1(self, t: NDArray) -> NDArray:
        # x0, y0, z0 = self.x, self.y, self.z
        x_positions = self.xt(t)
        y_positions = self.yt(t)
        z_positions = self.zt(t)

        x_positions = array(self.xt(t))
        y_positions = array(self.yt(t))
        z_positions = array(self.zt(t))
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
    def __init__(self, x=0.0, y=0.0, z=0.0, angle_gains_db10=zeros((360, 360)),
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
        self._log = logging.getLogger(self.__class__.__qualname__)
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
        """
        Returns
        -------
        position_t
            (timestamps, 3)

        Usage
        -----
        for compute of distance need to add an axis, which is done by staking over axis =1 (0 is time and 2 is 3D coordinate) 
        positions_t = stack([ant.position_in_time(timestamps) for ant in self.tx_antennas], axis=1)  # [T, N_ant, 3]
        """
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
                 adc_sample_count_max=1024,
                 adc_sample_rate_max=25e6,
                 adc_sample_count=0,
                 config=None,
                 debug=False):
        self._log = logging.getLogger(self.__class__.__qualname__)
        adc_sample_rate = adc_sample_rate
        self.adc_sample_rate = adc_sample_rate
        # self.adc_sampling_frequency = adc_sample_rate
        self.adc_sample_rate = adc_sample_rate
        self.antennas = antennas
        self.adc_sample_count_max = adc_sample_count_max
        # self.adc_sample_count = adc_sample_count
        # self.adc_sample_count = adc_sample_count
        adc_sample_count = adc_sample_count
        # self.adc_sample_count = adc_sample_count
        self.adc_sample_count = adc_sample_count
        self.rx_high_pass_freq = 1e2
        self.rx_low_pass_freq = 1e8

        try:
            assert adc_sample_rate < adc_sample_rate_max
        except AssertionError:
            if debug:
                print(f"adc_sample_rate:{adc_sample_rate} > adc_sample_rate_max: {adc_sample_rate_max}")
            raise ValueError("ADC sampling value must stay below adc_sample_rate_max")
        return


class Transmitter():
    """
    Attributes:
    -----------
    tx_start_time: float
        time offset for the start of the first chirp
    tx_on_times: List[float]
        list of 2-uples of start/stop timess for each chirp transmitted
        list of slopes for each chirp transmitted
    """
    chirp_count = 1
    frame_count = 1
    tx_start_time = 0.0
    tx_on_times = []
    conf = {"multiplexing": "TDM"}

    def __init__(self,
                 chirp_start_freq: float = 60e9,
                 chirp_slope: float = 1e12,
                 chirp_end_time: float = 1e-6,
                 antennas=[Antenna()],
                 chirp_period=1e-6,
                 chirp_count=1,
                 frame_period=50e-3,
                 frame_count=1,
                 **kwargs):
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
        chirp_period: float
            time increment between two TX antennas sending a chirp
        chirp_count: int
            The # chirps each TX antenna sends per frame
        frame_period: float
            time increment between end of last chirp in frame N-1 and
            first chirp in frame N (offset on top of
            chirp_period). If t_interframe==0, then there will be a
            single chirp_period offset.
        frame_count: int
            The number of iterations where each TX antennas send chirp_count
        conf: dict
            additional optional parameters (reserved for future usage)
            includes mimo_mode = [TDM, DDM]
        """
        self._log = logging.getLogger(self.__class__.__qualname__)
        self.chirp_slope = chirp_slope
        self.slope = chirp_slope
        slope = chirp_slope

        if chirp_slope < 1e8:
            # print("chirp_slope is low", )
            self._log.debug(f"chirp_slope is low (WARN): {chirp_slope}")
        bw = chirp_slope * chirp_end_time
        self.f0_min = chirp_start_freq
        self.chirp_start_freq = chirp_start_freq
        # self.slope = slope
        
        self.chirp_period = chirp_period
        self.chirp_period = chirp_period
        self.chirp_count = chirp_count
        self.antennas = antennas
        if frame_period == 0:
            self.frame_period = chirp_period
        else:
            assert frame_period >= chirp_period
            self.frame_period = frame_period
        self.frame_count = frame_count
        self.bw = bw
        if bw is not None and chirp_end_time is None:
            log_msg = "DeprecationWarning: chirp defined with bw instead of ramp_end_time about to be removed !!!!"
            print(log_msg)
            ramp_end_time = bw/slope
            self.ramp_end_time = ramp_end_time
        else:
            self.ramp_end_time = chirp_end_time
            self.chirp_end_time = chirp_end_time

        self.multiplexing = kwargs.get("multiplexing", "TDM")
        self.TX_phaser_slopes = kwargs.get("TX_phaser_slopes", array([0]*len(antennas)))
        self.tx_start_time = kwargs.get("tx_start_time", 0)

        self._log.debug(f"multiplexing: {self.multiplexing}")
        self._log.debug(f"self.TX_phaser_slopes: {self.TX_phaser_slopes}")

            
        """if self.mimo_mode == "TDM":
            # compute the tx_on_times for each chirp
            for frame_idx in range(frame_count):
                for chirp_idx in range(chirp_count):
                    chirp_start = self.chirp_t_start(frame_idx, chirp_idx)
                    chirp_end = chirp_start + ramp_end_time
                    self.tx_on_times.append((chirp_start, chirp_end))
                return"""

    def TX_freq(self, timestamps: NDArray) -> NDArray:
        """FIXME: this functions' description only describes the ToF use case, not LO use case
        
        Returns for each TX->Scatterer->RX path the TX frequency
        at which the chirps was sent when it is received by the mixer

        Parameters
        ----------
        timestamps
            (timestamps, TX antenna count, Scatterer count, RX antenna count)
            the timestamp at which ADC are sampling, the TX freq is then
            computed as timestamp-time_of_flight
        Returns
        --------
        tx_frequencies
            (timestamps, TX antenna count, Scatterer count, RX antenna count)
            values at each timestamp of the TX freq for antenna
            which can then be used to compute the tones on each RX antenna
            before mixing with LO to generate all the IF tones
        """

        chirp_start_freq = self.chirp_start_freq
        chirp_slope = self.chirp_slope
        chirp_end_time = self.chirp_end_time
        chirp_period = self.chirp_period
        # FIXME: remove this
        if chirp_period == 0:
            chirp_period = chirp_end_time
        else:
            chirp_period = chirp_period
        frame_period = self.frame_period
        chirp_count = self.chirp_count
        frame_count = self.frame_count
        antenna_count = len(self.antennas)

        if ((chirp_count > 1) or (antenna_count > 1)) and chirp_period == 0:
            self._log.error("self.chirp_period = 0")

        assert antenna_count == timestamps.shape[1], (
            f"timestamps shape {timestamps.shape} does not match antenna count {antenna_count}"
        )

        self._log.debug(f"antenna_count: {antenna_count}")
        self._log.debug(f"chirp_count:  {chirp_count}")

        multiplexing = self.multiplexing

        # timestamps: (ts, TX, S, RX)
        frame_index = timestamps // frame_period
        frame_index = np.clip(frame_index, 0, frame_count - 1)
        transmit_timestamps = timestamps - frame_index * frame_period

        if multiplexing == "TDM":
            # chirp_start for antenna `a`, chirp `k`:
            #   chirp_start[a, k] = (a + k * antenna_count) * chirp_period
            #
            # Invert: given timestamp t and antenna index a (broadcast from axis 1),
            #   global_chirp = t / chirp_period          (continuous)
            #   k            = floor((global_chirp - a) / antenna_count)
            #
            # Then chirp_start = (a + k * antenna_count) * chirp_period

            # antenna index broadcast to (1, TX, 1, 1)
            ant = np.arange(antenna_count)[None, :, None, None]
            global_chirp = transmit_timestamps / chirp_period            # (ts, TX, S, RX)
            k = np.floor((global_chirp - ant) / antenna_count).astype(np.int64)
            k = np.clip(k, 0, chirp_count - 1)
            cs = (ant + k * antenna_count) * chirp_period  # chirp_start per element

        elif multiplexing == "DDM":
            # All antennas share the same chirp timeline:
            #   chirp_start[k] = k * chirp_period
            k  = np.floor(transmit_timestamps / chirp_period).astype(np.int64)
            k  = np.clip(k, 0, chirp_count - 1)
            cs = k * chirp_period                               # (ts, TX, S, RX)
        else:
            raise ValueError(f"Unsupported multiplexing scheme: {multiplexing!r}")

        self._log.debug(f"multiplexing: {multiplexing}")

        # Check timestamp falls within the active chirp window (not in dead-time)
        # chirp_end = cs + chirp_end_time  (chirp_end_time is the chirp duration scalar)
        active = (transmit_timestamps >= cs) & (transmit_timestamps <= cs + chirp_end_time)

        # Frequency at this timestamp within the active chirp; zeroed outside
        # All arrays are (ts, TX, S, RX) — no large intermediates
        tx_frequencies = np.where(active, chirp_start_freq + chirp_slope * (transmit_timestamps - cs), 0.0)

        return tx_frequencies

    def TX_phases_old(self, timestamps: NDArray) -> NDArray:
        """Returns for each TX->Scatterer->RX path the TX phase
        at which the chirps was sent when it is received by the mixer.
        For code logic and documentation refer to TX_freqs

        Parameters
        ----------
        timestamps
            (T, TX, Scatterer, RX)
        Returns
        --------
        tx_phases
            (T, TX, Scatterer, RX)
        """

        chirp_start_freq = self.chirp_start_freq
        chirp_slope = self.chirp_slope
        chirp_end_time = self.chirp_end_time
        chirp_period = self.chirp_period
        chirp_count = self.chirp_count  # number chirps per antenna
        antenna_count = len(self.antennas)  # number antennas
        if ((chirp_count > 1) or (antenna_count > 1)) and self.chirp_period == 0:
            self._log.error("self.chirp_period = 0")
        chirp_indexes = np.arange(chirp_count)
        antenna_indexes = np.arange(antenna_count)
        assert antenna_count == timestamps.shape[1]

        # Absolute chirp index for transmitter k on its nth chirp
        # is a function of the multiplexing
        multiplexing = self.multiplexing
        if multiplexing == "TDM":
            """
            [[ 0  3  6  9 12 15 18 21 24 27]
            [ 1  4  7 10 13 16 19 22 25 28]
            ...
            """
            # [antenna_count, chirp_count]
            chirp_index = antenna_indexes[:, None] + chirp_indexes[None, :] * antenna_count
            phase_slope = zeros(antenna_count)
        elif multiplexing == "DDM":
            """ DDM allows 10 chirps per antenna to be sent over only 10 chirps 
            in total being in effect antenna_count faster than TDM
            (at some compromises)
            [[0 1 2 3 4 5 6 7 8 9]
            [0 1 2 3 4 5 6 7 8 9]
            ...
            """
            # [antenna_count, chirp_count]
            chirp_index = antenna_indexes[:, None]*0 + chirp_indexes[None, :]
            phase_slope = self.conf["TX_phaser_slopes"]

        # [1, antenna_count, 1, 1, chirp_count]
        chirp_start = chirp_index[None, :, None, None, :] * chirp_period
        chirp_end = chirp_start + chirp_end_time

        timestamps = timestamps[..., None]  #[:, None, None, None]  # [timestamp_count, 1, 1] T/TX/S/RX
        active = (timestamps >= chirp_start) & (timestamps <= chirp_end)

        # tx_phase = lambda t_adc, chirp_idx=chirp_idx: pi*self.TX_phaser_slopes[tx_idx]*chirp_idx
        phase = chirp_index[None, :, None, None, :] * phase_slope[None, :, None, None, None]

        tx_phases = (active * phase).sum(axis=4)
        return tx_phases

    def TX_phases(self, timestamps: NDArray) -> NDArray:
        """Returns for each TX->Scatterer->RX path the TX phase
        at which the chirps was sent when it is received by the mixer.
        For code logic and documentation refer to TX_freqs

        Parameters
        ----------
        timestamps
            (T, TX, Scatterer, RX)
        Returns
        --------
        tx_phases
            (T, TX, Scatterer, RX)
        """
        chirp_end_time = self.chirp_end_time
        chirp_period  = self.chirp_period
        chirp_count    = self.chirp_count
        antenna_count  = len(self.antennas)

        if ((chirp_count > 1) or (antenna_count > 1)) and chirp_period == 0:
            self._log.error("self.chirp_period = 0")

        assert antenna_count == timestamps.shape[1]

        multiplexing = self.multiplexing

        if multiplexing == "TDM":
            # phase_slope is zero for all antennas in TDM — result is always 0
            # but we still gate on active to preserve the zero-outside-chirp contract
            ant = np.arange(antenna_count)[None, :, None, None]   # (1, TX, 1, 1)
            k   = np.floor((timestamps / chirp_period - ant) / antenna_count).astype(np.int64)
            k   = np.clip(k, 0, chirp_count - 1)
            cs  = (ant + k * antenna_count) * chirp_period

            # phase_slope == 0 for all antennas → tx_phases is identically 0
            # keep the np.where for correctness in case TDM ever gets non-zero slopes
            phase_slope = np.zeros(antenna_count)                  # (TX,)
            ps  = phase_slope[None, :, None, None]                 # (1, TX, 1, 1)
            phase = k * ps                                         # (ts, TX, S, RX)

        elif multiplexing == "DDM":
            k  = np.floor(timestamps / chirp_period).astype(np.int64)
            k  = np.clip(k, 0, chirp_count - 1)
            cs = k * chirp_period

            phase_slope = np.asarray(self.conf["TX_phaser_slopes"])  # (TX,)
            ps  = phase_slope[None, :, None, None]                   # (1, TX, 1, 1)
            phase = k * ps                                           # (ts, TX, S, RX)

        else:
            raise ValueError(f"Unsupported multiplexing scheme: {multiplexing!r}")

        active     = (timestamps >= cs) & (timestamps <= cs + chirp_end_time)
        tx_phases  = np.where(active, phase, 0.0)

        return tx_phases


class TransmitterDDM(Transmitter):
    def __init__(self,
                 chirp_start_freq=60e9,
                 chirp_slope=None,
                 chirp_end_time=None,
                 slope_MHz_us=None,
                 bw=4e9,
                 antennas=[Antenna()],
                 chirp_period=0.0,
                 chirp_count=1,
                 frame_period=0.0,
                 frame_count=1,
                 **kwargs):
        conf={}
        conf["multiplexing"] = kwargs["conf"].get("multiplexing", "DDM")
        conf["TX_phaser_slopes"] = kwargs["conf"].get("TX_phaser_slopes", array([0]*len(antennas)))
        super().__init__(chirp_start_freq,
                         chirp_slope,
                         chirp_end_time,
                         antennas,
                         chirp_period,
                         chirp_count,
                         frame_period,
                         frame_count,
                         **conf)
        if "TX_phase_offset" in conf:
            raise ValueError("TX_phase_offset deprecated replaced by TX_phaser_slopes")
        assert "TX_phaser_slopes" in conf
        assert len(conf["TX_phaser_slopes"]) == len(antennas)
        self.TX_phaser_slopes = conf["TX_phaser_slopes"]
        
    def TX_freq(self, times: NDArray, tx_idx=-1) -> NDArray[float64]:
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
            for frame_idx in range(self.frame_count):
                for chirp_idx in range(self.chirp_count):
                    t_start_chirp = self.frame_period*frame_idx + self.chirp_period*chirp_idx
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
        self._log = logging.getLogger(self.__class__.__qualname__)
        self.transmitter = transmitter
        self.receiver = receiver
        self.rx_antennas = receiver.antennas
        self.tx_antennas = transmitter.antennas
        self.chirp_start_freq = transmitter.chirp_start_freq

        self.frame_count = transmitter.frame_count
        # issue #5 - frame_count renamed to total_number_frames
        self.chirp_count = transmitter.chirp_count
        self.adc_sample_count = receiver.adc_sample_count
        # self.adc_sample_rate = receiver.adc_sample_rate
        self.adc_sample_rate = self.receiver.adc_sample_rate
        self.chirp_slope = transmitter.chirp_slope
        self.chirp_period = transmitter.chirp_period
        self.bw = transmitter.bw
        self.tx_conf = transmitter.conf
        # self.mimo_mode = transmitter.conf["mimo_mode"]
        self.TX_freq = transmitter.TX_freq
        self.TX_phases = transmitter.TX_phases
        # self.chirp_t_start = transmitter.chirp_t_start
        if self.adc_sample_count == 0:
            # self.adc_sample_count = int(transmitter.bw * receiver.adc_sample_rate / transmitter.slope)
            if self.adc_sample_count == 0:  # pragma: no cover
                log_msg = f"nadc updated to 0: {transmitter.bw:.2g}" +\
                    f"{receiver.adc_sample_rate:.2g}= {transmitter.bw*receiver.adc_sample_rate:.2g}" +\
                    f" /  {transmitter.slope:.2g}"
                # , transmitter.bw, receiver.adc_sample_rate, transmitter.slope

            if debug:  # pragma: no cover
                print("updating NADC from 0 to:", self.adc_sample_count)
                raise ValueError(log_msg)
        t_fft = receiver.adc_sample_count / receiver.adc_sample_rate
        t_chirp = transmitter.ramp_end_time  # transmitter.bw / transmitter.slope

        bw_adc = self.adc_sample_count*transmitter.slope/receiver.adc_sample_rate

        self._log.debug(f"Bandwidth in chirp: {transmitter.bw:.2g}")
        if bw_adc < 0.8 * transmitter.bw:  # pragma: no cover
            self._log.debug(f"! BW ADC: {bw_adc:.2g} << chirp: {transmitter.bw:.2g}")
        self._log.debug(f"Bandwidth in ADC buffers: {bw_adc:.2g}")
        if self.adc_sample_count < 8:  # pragma: no cover
            print("!!!! ADC # low", self.adc_sample_count)
            print("BW", transmitter.bw)
            print("BW GHz", transmitter.bw/1e9)
            print("K", transmitter.slope)
            print("K/1e12", transmitter.slope/1e12)
            print("TC", transmitter.bw / transmitter.slope)
            print("adc_sample_count", transmitter.bw / transmitter.slope * receiver.adc_sample_rate)

        if adc_po2:
            self.adc_sample_count = 2 ** int(log2(self.adc_sample_count))
            adc_sample_count = self.adc_sample_count
            assert adc_sample_count / receiver.adc_sample_rate * transmitter.slope < transmitter.bw
        self.f0_min = transmitter.f0_min
        self.slope = transmitter.slope
        self.chirp_period = transmitter.chirp_period
        self.chirp_count = transmitter.chirp_count
        self.frame_period = transmitter.frame_period
        self.frame_count = transmitter.frame_count
        self.v = medium.v
        self.medium = medium
        self.bw = transmitter.bw
        # FIXME: moves this to simulation level
        # __range_bin: deprecated as relies on c for compute
        # __c = 3e8
        # self.range_bin_deprec = receiver.adc_sample_rate*__c/2/self.slope/self.adc_sample_count

        if all(self.rx_antennas[0].angle_gains_db10 == 0):
            for idx, _ in enumerate(self.rx_antennas):
                self.rx_antennas[idx].f_min_GHz = self.f0_min/1e9
                self.rx_antennas[idx].f_max_GHz = (self.f0_min + self.bw)/1e9

        self._log.debug(f"rx fmin (GHz): {self.rx_antennas[idx].f_min_GHz}")
        self._log.debug(f"rx fmax (GHz): {self.rx_antennas[idx].f_max_GHz}")

        if all(self.tx_antennas[0].angle_gains_db10 == 0):
            for idx, _ in enumerate(self.tx_antennas):
                self.tx_antennas[idx].f_min_GHz = self.f0_min/1e9
                self.tx_antennas[idx].f_max_GHz = (self.f0_min + self.bw)/1e9

        self._log.debug(f"tx fmin:{self.tx_antennas[idx].f_min_GHz}")
        self._log.debug(f"tx fmax:{self.tx_antennas[idx].f_max_GHz}")
        if t_fft > t_chirp:
            if debug:  # pragma: no cover
                self._log.warning(f"T_FFT: {t_fft:.2g} > T_C: {t_chirp:.2g}")
            else:
                self._log.error(f"T_FFT: {t_fft:.2g} > T_C: {t_chirp:.2g}")
                raise ValueError(ERR_TFFT_lte_TC)

        try:
            assert self.adc_sample_count < receiver.adc_sample_count_max
        except AssertionError:
            if debug:  # pragma: no cover
                print(f"buffer size: {self.adc_sample_count} > " +
                      f"vs max buffer size: {receiver.adc_sample_count_max}" +
                      f"ratio: {self.adc_sample_count/receiver.adc_sample_count_max}")
            raise ValueError("ADC buffer overflow")
        return

    def mixer(self, timestamps: NDArray,
              f_rx: NDArray) -> NDArray:
        """ RF to baseband conversion, emulates the mixing of RX and TX
        which is multiplication and low pass filter, the result is
        that the `itermediate frequency` the if is the substraction of the
        two rf frequencies. This function is a stub for other possible down
        conversion in future versions.
        Note: no low pass filtering here. Currently done at IF stage

        Parameters
        ----------
        timestamps
            (T)
        f_rx
            (T, TX, S, RX)
        Return
        ------
        f_if
            (T, TX, S, RX)
        """
        scatterer_shape = (timestamps.shape[0], f_rx.shape[1], f_rx.shape[2], f_rx.shape[3])
        timestamps_rx = np.broadcast_to(timestamps[:, None, None, None], scatterer_shape)

        f_tx = self.TX_freq(timestamps_rx)
        f_if = f_tx - f_rx

        return f_if

    def adc_sampling(self, f_if,
                     adc_times,
                     ph_rx,
                     time_of_flight: NDArray[float64],
                     radar_equation=False,
                     datatype=complex64,
                     debug=False) -> NDArray[ADCType]:
        """ sampling the f_if signals at adc_times time stamp
        Parameters
        ----------
        f_if
            (timestamps,tx, scatterers, rx)
        adc_times
            (timestamp, tx, scatters, rx)
        time_of_flight
            (timestamps,tx, scatterers, rx)
        Returns
        -------
        YIF
            (timestamps, rx_antenna_count)
        """
        ph_tx = self.TX_phases(adc_times)
        rx_high_pass_freq = self.receiver.rx_high_pass_freq
        rx_low_pass_freq = self.receiver.rx_low_pass_freq

        if_filter = (rx_high_pass_freq < abs(f_if)) & \
            (abs(f_if) < rx_low_pass_freq)
        self._log.debug(f"f_if before ADC filters: {f_if}")
        self._log.debug(f"if_filter: {if_filter[:8]}")
        self._log.debug(f"if_filter: {if_filter[:8]}")

        f_if[~(if_filter)] = 0
        fif_max = np_max(f_if)
        adc_sampling_frequency = self.adc_sample_rate
        if (fif_max * 2 > adc_sampling_frequency) and (fif_max <1e9):
            self._log.critical("some scatterers seem to be above Nyquist, they'll be filtered out")
            print(f_if)
            print("adc_saml",adc_sampling_frequency)
            print(fif_max*2)
            exit()

        self._log.debug(f"f_if: {f_if[:8]}")
        if debug:
            print("f_if after if_filter", f_if)
        YIF = zeros(f_if.shape)
        if debug:
            print("if_filter !!!!!", if_filter)
            print("f_if", f_if)

        if np_any(if_filter):
            # skip computing the IF if all the frequencies are too low or too high
            # YIF += BB_IF(chirp_rx, chirp_tx, T, antenna_tx, antenna_rx, scatterer, medium)

            # adc_samples = BB_IF_v2(tr_chirp, fif, rx_high_pass_freq, rx_low_pass_freq,
            #                       ph_tx, debug=debug)
            Tc = adc_times - adc_times[0]
            # print("Tc = adc_times-adc_times[0]", Tc)
            #IF_filter = ((rx_high_pass_freq <= np_abs(f_if)) &
            #            (np_abs(f_if) <= rx_low_pass_freq))
            YIF = where(if_filter,
                        exp(2 * pi * 1j * (f_if) * Tc +
                            1j*(ph_tx-ph_rx) +
                            2 * pi * 1j * time_of_flight*self.transmitter.chirp_start_freq -  # this is the important term for speed measure
                            1 * pi * 1j * self.transmitter.chirp_slope*time_of_flight**2),
                        YIF)
            self._log.debug(f"radar equation:{radar_equation}")
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

                YIF = YIF * scatterer.rcs(f0)
                if scatterer.scatterer_type == "corner_reflector":
                    YIF = YIF / distance**2
                else:
                    YIF = YIF / distance**4
                YIF = YIF * 10**(L*distance)
            # adc values are the sum of signals arriving on the same RX antenna
            # so summing over axis=1 (TX) and axis=2 (scatterers)
            self._log.debug(f"YIF.shape: {YIF.shape}")
            self._log.debug(f"YIF t=0: {YIF[0:10,0,0,0]}")
            YIF = sum(YIF, axis=(1, 2))
            self._log.debug(f"YIF.shape after sum on axis: {YIF.shape}")
            self._log.debug(f"YIF t=0: {YIF[0:10,0]}")

            if datatype in [float64, float32, float16]:
                YIF = real(YIF)
            elif datatype in [complex64, complex]:
                pass
            elif datatype in [int64, int32, int16]:
                YIF = int(real(YIF)/max(real(YIF)) * (2**(8*datatype().nbytes-1)-1))
        else:
            YIF = sum(YIF, axis=(1, 2))
        return YIF

    def position_tx_antennas(self, timestamps) -> NDArray:
        """
        Returns
        -------
        positions_t:
            (timestamps, antenna_count, 3)
        """
        positions_t = stack([ant.position_in_time(timestamps) for ant in self.tx_antennas], axis=1)  # [T, N_ant, 3]
        return positions_t

    def position_rx_antennas(self, timestamps) -> NDArray:
        """
        Returns
        -------
        positions_t:
            (timestamps, antenna_count, 3)
        """
        positions_t = stack([ant.position_in_time(timestamps) for ant in self.rx_antennas], axis=1)  # [T, N_ant, 3]
        return positions_t
