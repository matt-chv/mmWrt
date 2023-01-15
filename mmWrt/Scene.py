# from .Transmitter import Transmitter as Transmitter
# from .Receiver import Receiver as Receiver
# from .Radar import Radar as Radar

from numpy import sqrt


class Target():
    def __init__(self, x=0, y=0, z=0,
                 vx=lambda t: 0, vy=lambda t: 0, vz=lambda t: 0):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

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


class Receiver():
    def __init__(self,
                 fs=4e2,
                 antennas=[(0, 0, 0)],
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
                 antennas=[(0, 0, 0)],
                 t_interchirp=0,
                 chirps_count=1,
                 frames_count=1,
                 conf=None):
        self.f0_min = f0_min
        self.slope = slope
        self.t_interchirp = t_interchirp
        self.chirps_count = chirps_count
        self.antennas = antennas
        self.frames_count = frames_count
        self.bw = bw
        return


class Medium():
    def __init__(self, v=3e8, L=0):
        # v default to c=3e8 speed of light in void
        # L defaults to 0 losses in medium
        self.v = v
        self.L = L


class Radar():
    def __init__(self, transmitter=Transmitter(), receiver=Receiver(),
                 medium=Medium(), debug=False):
        self.transmitter = transmitter
        self.rx_antennas = receiver.antennas
        self.tx_antennas = transmitter.antennas
        self.frames_count = transmitter.frames_count
        self.fs = receiver.fs
        self.n_adc = int(transmitter.bw / transmitter.slope * receiver.fs)
        self.f0_min = transmitter.f0_min
        self.slope = transmitter.slope
        self.t_interchirp = transmitter.t_interchirp
        self.v = medium.v
        try:
            assert self.n_adc < receiver.max_adc_buffer_size
        except AssertionError:
            if debug:
                print(f"buffer size: {self.n_adc} > " +
                      f"vs max buffer size: {receiver.max_adc_buffer_size}" +
                      f"ratio: {self.n_adc/receiver.max_adc_buffer_size}")
            raise ValueError("ADC buffer overflow")
        return
