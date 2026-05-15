""" Testing the Scene Engine for distance computation.
"""
import numpy as np
from os.path import abspath, join, pardir
import pytest
import sys
dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Scene import scene_distance
from test_assets import (d_0m, d_5p1m, d_10p1m, target_static_5p1m, antenna_origin_static)  # noqa E402


def test_distance_5p1m_t_0():
    d0 = target_static_5p1m.distance()
    assert abs(d0-d_5p1m) < 1e-3


@pytest.mark.parametrize("time", [(0),(1000)])
def test_distance_5p1m_t_100(time):
    """ Ensure that the distance between origin and
    the target_static_5p1m is always 5.1 meters at anytime"""
    d0 = target_static_5p1m.distance(t=time)
    assert abs(d0-d_5p1m) < 1e-3


@pytest.mark.parametrize("time", [(0),(1000)])
def test_distance_5p1m_t_1000(time):
    """ Ensure that the distance between the target target_static_5p1m
    and the antenna at the origin is always 5.1 meters, for any time."""
    tpos = target_static_5p1m.pos_t1(t=np.array([time]))
    apos = antenna_origin_static.position_in_time(t=np.array([time]))
    d0 = scene_distance(apos, tpos)
    assert abs(d0-d_5p1m) < 1e-3


def test_distance_5p1m_times():
    """ Ensure that the target target_static_5p1m
    is always 5.1 meters away from the antenna, for all times."""
    times = np.arange(0, 1000, 100)
    targets_positions = np.empty((1, 10, 3))

    targets_positions[0,:,:] = target_static_5p1m.pos_t1(t=times)

    antenna_positions = np.empty((1, 10, 3))
    antenna_positions[0,:,:] = antenna_origin_static.position_in_time(t=times)

    d0 = scene_distance(antenna_positions, targets_positions)
    # d0 = [[5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1]]
    tmp = abs(d0-d_5p1m)
    # tmp = [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    assert np.allclose(tmp, np.zeros(10), atol=1e-3)