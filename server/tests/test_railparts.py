"""roads.rail_parts: rail ways as track parts — chain profile, third rail by class, level crossings only at grade."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import roads  # noqa: E402
from test_roads import FRAME, LON, M_LON, features  # noqa: E402


def flat(z=10.0):
    return lambda la, lo: np.full(np.shape(la), z, np.float64)


def dipped(ends_z=10.0, bottom_z=4.0, at_m=30.0, width_m=6.0):
    """Ground that dips `bottom_z` at `at_m` along a way leaving (LAT, LON) eastwards, `ends_z` at its ends."""
    def elev(la, lo):
        x = (np.asarray(lo, np.float64) - LON) * M_LON
        return bottom_z + (ends_z - bottom_z) * np.clip(np.abs(x - at_m) / width_m, 0.0, 1.0)
    return elev


class RailPartsTest(unittest.TestCase):
    def test_a_rail_way_becomes_one_track_part_on_the_ground(self):
        f = features([[(0.0, 0.0), (60.0, 0.0)]], ["rail"], subtypes=["rail"])
        parts = roads.rail_parts(FRAME, 0, 0, [f], flat(10.0))
        self.assertEqual(len(parts), 1)
        z = parts[0]["pts"][:, 2]
        self.assertTrue(np.allclose(z, 10.0), "at grade the profile is the ground")
        self.assertFalse(parts[0]["third"], "a mainline class is overhead, not third rail")
        self.assertEqual(parts[0]["crossings"], [])

    def test_a_metro_way_gets_the_third_rail(self):
        f = features([[(0.0, 0.0), (60.0, 0.0)]], ["subway"], subtypes=["rail"])
        self.assertTrue(roads.rail_parts(FRAME, 0, 0, [f], flat(10.0))[0]["third"])

    def test_a_road_across_an_at_grade_track_is_a_level_crossing(self):
        f = features([[(0.0, 0.0), (60.0, 0.0)], [(30.0, -20.0), (30.0, 20.0)]],
                     ["rail", "residential"], subtypes=["rail", "road"])
        cross = roads.rail_parts(FRAME, 0, 0, [f], flat(10.0))[0]["crossings"]
        self.assertEqual(len(cross), 1)
        a, b = cross[0]
        self.assertLess(a, 30.0)
        self.assertGreater(b, 30.0)
        self.assertLess(b - a, 12.0, "the road's width, not the whole way")

    def test_a_road_under_a_rail_bridge_is_not_flushed(self):
        f = features([[(0.0, 0.0), (60.0, 0.0)], [(30.0, -20.0), (30.0, 20.0)]],
                     ["rail", "residential"], subtypes=["rail", "road"], levels=[(0, 1.0)])
        parts = roads.rail_parts(FRAME, 0, 0, [f], flat(10.0))
        self.assertEqual(parts[0]["crossings"], [], "a bridge over the road is the structure's business")

    def test_a_rail_bridge_stands_over_the_valley_and_meets_its_approaches(self):
        f = features([[(0.0, 0.0), (60.0, 0.0)]], ["rail"], subtypes=["rail"], levels=[(0, 1.0)])
        elev = dipped()
        parts = roads.rail_parts(FRAME, 0, 0, [f], elev)
        pts = parts[0]["pts"]
        z = pts[:, 2]
        self.assertAlmostEqual(float(z[0]), 10.0, places=6, msg="the abutment meets the approach")
        self.assertAlmostEqual(float(z[-1]), 10.0, places=6)
        ground_mid = float(np.min(elev(np.zeros(1), np.array([LON + 30.0 / M_LON]))))
        mid = z[len(z) // 2]
        self.assertGreater(mid - ground_mid, 3.0, "the deck stands over the valley")


if __name__ == "__main__":
    unittest.main()
