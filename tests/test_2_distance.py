""" Testing the two_way_range
v0.0.11: 7 passed
"""
import numpy as np
from os.path import abspath, join, pardir
import pytest
import sys
dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt.Scene import two_way_range  # noqa E402
from test_assets import (d_0m, d_5p1m, d_10p1m, scatterer_static_5p1m, antenna_origin_static)  # noqa E402


def test_distance_5p1m_t_0():
    d0 = scatterer_static_5p1m.distance()
    assert abs(d0-d_5p1m) < 1e-3


@pytest.mark.parametrize("time", [(0), (1000)])
def test_distance_5p1m_t_100(time):
    # Ensure that the distance between origin and
    # the scatterer_static_5p1m is always 5.1 meters at anytime
    d0 = scatterer_static_5p1m.distance(t=time)
    assert abs(d0-d_5p1m) < 1e-3


@pytest.mark.parametrize("time", [(0), (1000)])
def test_distance_5p1m_t_1000(time):
    # Ensure that the distance between the scatterer scatterer_static_5p1m
    # and the antenna at the origin is always 5.1 meters away, for any time.
    # note since we do compute 2 ways
    # (from antenna to scatterer and back to antenna), 5.1*2
    tpos = scatterer_static_5p1m.pos_t1(t=np.array([time]))
    tpos = tpos[:, None, :]
    apos = antenna_origin_static.position_in_time(timestamp=np.array([time]))
    apos = apos[:, None, :]
    print(tpos.shape)
    print(apos.shape)
    d0 = two_way_range(apos, tpos, apos)
    print(d0)
    # d0 = scene_distance(apos, tpos)
    assert abs(d0-d_5p1m*2) < 1e-3
    assert d0.shape == (1, 1, 1, 1), \
        f"Expected shape (T={1}, TX={1}, S={1}, RX={1}), got {d0.shape}"


def test_distance_5p1m_times():
    # Ensure that the scatterer scatterer_static_5p1m
    # is always 5.1 meters away from the antenna, for all times.
    times = np.arange(0, 1000, 100)

    scatterers_positions = np.empty((10, 1, 3))

    scatterers_positions[:, 0, :] = scatterer_static_5p1m.pos_t1(t=times)

    antenna_positions = np.empty((10, 1, 3))
    antenna_positions[:, 0, :] = \
        antenna_origin_static.position_in_time(timestamp=times)

    d0 = two_way_range(antenna_positions, scatterers_positions,
                       antenna_positions)
    # d0 = [[10.2], [10.2], [10.2] ...
    # [10.2] [10.2] [10.2] [10.2] [10.2] [10.2] [10.2]]
    tmp = abs(d0-d_5p1m*2)
    expected = np.zeros((10, 1, 1, 1))
    assert np.allclose(tmp, expected, atol=1e-3)
    assert tmp.shape == expected.shape


def test_distance_multiple_antennas():
    from test_assets import antennas_ULA_64_60G
    times = np.array([0])
    antenna_positions = np.empty((8, 1, 3))
    antenna_positions[:, :, :] = [a.position_in_time(timestamp=times) for
                                  a in antennas_ULA_64_60G[0:8]]
    scatterers_positions = np.empty((1, 1, 3))
    scatterers_positions[:, 0, :] = scatterer_static_5p1m.pos_t1(t=times)
    d0 = two_way_range(antenna_positions, scatterers_positions,
                       antenna_positions)
    print(d0)
