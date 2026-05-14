""" Place holder for testing the RX side of the system.
at the moment it is mostly done in the Raytracing module
will need to be moved into the scene modeling, place holder for later
"""

from os.path import abspath, join, pardir
import sys
from numpy import arange
from scipy.fft import fft
from scipy.signal import find_peaks


dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Raytracing import adc_samples
