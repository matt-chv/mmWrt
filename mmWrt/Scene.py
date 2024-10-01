# from .Transmitter import Transmitter as Transmitter
# from .Receiver import Receiver as Receiver
# from .Radar import Radar as Radar

from numpy import all, log2, pi, sqrt, zeros

ERR_TARGET_T0 = "xyz<>xyzt(0)"
ERR_TFFT_lte_TC = "TFFT should be shorter than TC"


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
            self.xt = lambda t: x

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
            self.yt = lambda t: y

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
            self.zt = lambda t: z

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

    def pos_t(self, t=0):
        # x0, y0, z0 = self.x, self.y, self.z
        xt, yt, zt = self.xt(t), self.yt(t), self.zt(t)
        position_t = (xt, yt, zt)
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


class Receiver():
    def __init__(self,
                 fs=4e2,
                 antennas=(Antenna(),),
                 max_adc_buffer_size=1024,
                 max_fs=25e6,
                 n_adc=0,
                 config=None,
                 debug=False):
        self.fs = fs
        self.antennas = antennas
        self.max_adc_buffer_size = max_adc_buffer_size
        self.n_adc = n_adc
        try:
            assert fs < max_fs
        except AssertionError:
            if debug:
                print(f"fs:{fs} > max_fs: {max_fs}")
            raise ValueError("ADC sampling value must stay below max_fs")
        return


class Transmitter():
    def __init__(self,
                 f0_min=60e9,
                 slope=None,
                 slope_MHz_us=None,
                 bw=4e9,
                 antennas=[Antenna()],
                 t_inter_chirp=0.0,
                 chirps_count=1,
                 t_inter_frame=0.0,
                 frames_count=1,
                 conf=None):
        """Transmitter class models a radar transmitter

        Parameters
        ----------
        f0_min: float
            start frequency of the chirp
        slope: Optional[float]
            the slope of the linearly growing chirp frequency
        slope_MHz_us: Optional[float]
            mutually exclusive with slope being slope parameter
            slope in MHz/us: a 4 GHz in 16 us is 250 MHz/us.
        bw: float
            bandwidth of the chirp (i.e. fmax-fmin)
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
        """
        if slope is None and slope_MHz_us is None:
            slope_MHz_us = 250
        if slope is not None and slope_MHz_us is not None:  # pragma: no cover
            assert ValueError("only slope or slope_MHz_us can be specified")
        if slope is None:
            slope = slope_MHz_us * 1e12  # type: ignore
        assert slope > 1e8
        self.f0_min = f0_min
        self.slope = slope
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
        return


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
        self.rx_antennas = receiver.antennas
        self.tx_antennas = transmitter.antennas

        self.frames_count = transmitter.frames_count
        self.n_adc = receiver.n_adc
        self.fs = receiver.fs
        self.bw = transmitter.bw
        if self.n_adc == 0:
            self.n_adc = int(transmitter.bw * receiver.fs / transmitter.slope)
            if self.n_adc == 0:  # pragma: no cover
                log_msg = f"nadc updated to 0: {transmitter.bw:.2g}" +\
                    f"{receiver.fs:.2g}= {transmitter.bw*receiver.fs:.2g}" +\
                    f" /  {transmitter.slope:.2g}"
                # , transmitter.bw, receiver.fs, transmitter.slope
                raise ValueError(log_msg)
            if debug:  # pragma: no cover
                print("updating NADC from 0 to:", self.n_adc)
        t_fft = receiver.n_adc / receiver.fs
        t_chirp = transmitter.bw / transmitter.slope

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
                print("rx fmin", self.rx_antennas[idx].f_min_GHz)
                print("rx fmax", self.rx_antennas[idx].f_max_GHz)

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
            if debug:  # pragma: no cover
                print(f"T_FFT: {t_fft:.2g}")
                print(f"T_C: {t_chirp:.2g}")
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
