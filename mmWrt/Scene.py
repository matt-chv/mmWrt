# from .Transmitter import Transmitter as Transmitter
# from .Receiver import Receiver as Receiver
# from .Radar import Radar as Radar

from numpy import log2, pi, sqrt, zeros


class Target():
    def __init__(self, x=0, y=0, z=0,
                 vx=lambda t: 0, vy=lambda t: 0, vz=lambda t: 0,
                 rcs_f=lambda f: 1,
                 target_type="point"):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.rcs_f = rcs_f
        self.target_type = target_type

    def speed(self):
        v = (self.vx, self.vy, self.vz)
        return v

    def distance(self, target=None, t=0):
        x0, y0, z0 = self.pos(t)
        if target is None:
            x1, y1, z1 = 0, 0, 0
        else:
            x1, y1, z1 = target.pos(t)
        dist = sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
        return dist

    def pos(self, t=0):
        x0, y0, z0 = self.x, self.y, self.z
        vx, vy, vz = self.speed()
        position_t = (x0+vx(t), y0+vy(t), z0+vz(t))
        return position_t

    def __str__(self):
        return f"x0:{self.x}, y0:{self.y}, z0:{self.z}"

    def rcs(self, f):
        return self.rcs_f(f)


class Antenna:
    def __init__(self, x=0, y=0, z=0, angle_gains_db10=zeros((360, 360)),
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

        """
        freq_GHz = freq / 1e9
        assert freq_GHz > self.f_min_GHz
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
        azimuth_deg = int((azimuth+pi)*180/pi)
        elevation_deg = int((elevation+pi)*180/pi)
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
                 config=None,
                 debug=False):
        self.fs = fs
        self.antennas = antennas
        self.max_adc_buffer_size = max_adc_buffer_size
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
                 slope=250e6,
                 bw=4e9,
                 antennas=(Antenna(),),
                 t_interchirp=0,
                 chirps_count=1,
                 frames_count=1,
                 conf=None):
        """Transmitter class models a radar transmitter

        Parameters
        ----------
        f0_min: float
            start frequency of the chirp
        slope: float
            the slope of the linearly growing chirp frequency
        bw: float
            bandwidth of the chirp (i.e. fmax-fmin)
        antennas: List[Antenna]
            transmitter Antennas instances
        t_interchirp: float
            time increment between two TX antennas sending a chirp
        chirps_count: int
            The # chirps each TX antenna sends per frame
        frames_count: int
            The number of iterations where each TX antennas send chirps_count
        conf: dict
            additional optional parameters (reserved for future usage)
        """
        self.f0_min = f0_min
        self.slope = slope
        self.t_interchirp = t_interchirp
        self.chirps_count = chirps_count
        self.antennas = antennas
        self.frames_count = frames_count
        self.bw = bw
        return


class Medium:
    def __init__(self, v=3e8, L=0):
        # v default to c=3e8 speed of light in void
        # L defaults to 0 dB/m losses in medium
        self.v = v
        self.L = L


class Radar:
    def __init__(self, transmitter=Transmitter(), receiver=Receiver(),
                 medium=Medium(), adc_po2=False, debug=False):
        self.transmitter = transmitter
        self.rx_antennas = receiver.antennas
        self.tx_antennas = transmitter.antennas
        self.frames_count = transmitter.frames_count
        self.fs = receiver.fs
        self.n_adc = int(transmitter.bw / transmitter.slope * receiver.fs)
        if adc_po2:
            self.n_adc = 2 ** int(log2(self.n_adc))
            n_adc = self.n_adc
            assert n_adc / receiver.fs * transmitter.slope < transmitter.bw
        self.f0_min = transmitter.f0_min
        self.slope = transmitter.slope
        self.t_interchirp = transmitter.t_interchirp
        self.chirps_count = transmitter.chirps_count
        self.v = medium.v
        self.medium = medium
        self.range_bin = self.fs * self.v / 2 / self.slope / self.n_adc
        # f*c/2/k
        self.f2d = medium.v * receiver.fs / 2 / transmitter.slope / self.n_adc
        try:
            assert self.n_adc < receiver.max_adc_buffer_size
        except AssertionError:
            if debug:
                print(f"buffer size: {self.n_adc} > " +
                      f"vs max buffer size: {receiver.max_adc_buffer_size}" +
                      f"ratio: {self.n_adc/receiver.max_adc_buffer_size}")
            raise ValueError("ADC buffer overflow")
        return
