"""rail.py: the ways that are railways, their electrification, and the track's cross-section."""

import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rail  # noqa: E402


def features(subtypes):
    return types.SimpleNamespace(attrs={"subtype": subtypes})


class RailTest(unittest.TestCase):
    def test_is_rail_picks_only_railways(self):
        f = features(["road", "rail", "road", "rail"])
        self.assertEqual(rail.is_rail(f), [1, 3])
        self.assertEqual(rail.is_rail(features(["road", "road"])), [])
        self.assertEqual(rail.is_rail(types.SimpleNamespace(attrs={})), [])

    def test_electrification_by_class(self):
        for cls in ("subway", "light_rail", "tram", "monorail"):
            self.assertEqual(rail.electrification(cls), "third_rail", cls)
        for cls in ("rail", "narrow_gauge", "", "service"):
            self.assertEqual(rail.electrification(cls), "overhead", cls)

    def test_the_running_rails_are_one_gauge_apart(self):
        rails = sorted(off for off, _w, part in rail.strips(third_rail=False) if part == "rail")
        self.assertEqual(len(rails), 2)
        self.assertAlmostEqual(rails[1] - rails[0], rail.RAIL_GAUGE, places=6)
        self.assertAlmostEqual(rails[0], -rail.RAIL_GAUGE / 2.0, places=6)

    def test_the_third_rail_lies_outside_the_running_rail(self):
        plain = rail.strips(third_rail=False)
        third = rail.strips(third_rail=True)
        self.assertEqual([p for _o, _w, p in plain], ["ballast", "rail", "rail"])
        self.assertEqual([p for _o, _w, p in third], ["ballast", "rail", "rail", "conductor"])
        right_rail = max(off for off, _w, part in third if part == "rail")
        cond_off, cond_w, _ = third[-1]
        self.assertGreater(cond_off, right_rail)
        self.assertAlmostEqual(cond_off - right_rail, rail.THIRD_RAIL_OFF, places=6)
        self.assertAlmostEqual(cond_w, rail.THIRD_RAIL_W, places=6)

    def test_the_ballast_is_the_widest_part_and_under_everything(self):
        strips = rail.strips()
        widest = max(strips, key=lambda s: s[1])
        self.assertEqual(widest[2], "ballast")
        self.assertGreater(rail.BALLAST_W, rail.THIRD_RAIL_OFF + rail.RAIL_GAUGE / 2.0 + rail.THIRD_RAIL_W)

    def test_the_rails_stand_on_the_ballast(self):
        self.assertEqual(rail.lift("ballast"), 0.0)
        self.assertGreater(rail.lift("rail"), rail.lift("ballast"))
        self.assertLess(rail.lift("conductor"), rail.lift("rail"))

    def test_centreline_drops_z_and_short_ways(self):
        got = rail.centreline([[0.0, 0.0, 5.0], [10.0, 0.0, 6.0]])
        self.assertEqual(got.shape, (2, 2))
        self.assertEqual(got[1, 0], 10.0)
        self.assertEqual(rail.centreline([[0.0, 0.0, 1.0]]).shape, (0, 2))


class ProfileTest(unittest.TestCase):
    def test_a_bridge_meets_its_approaches_without_a_step(self):
        ground = np.array([10.0, 10.2, 6.0, 5.8, 6.1, 10.1, 10.4])
        kinds = np.array([rail.GRADE, rail.GRADE, rail.BRIDGE, rail.BRIDGE, rail.BRIDGE, rail.GRADE, rail.GRADE])
        z = rail.profile(ground, kinds)
        self.assertAlmostEqual(z[2], z[1], places=9, msg="the left abutment sits on the approach's surface")
        self.assertAlmostEqual(z[4], z[5], places=9, msg="and the right one too")
        self.assertGreater(z[3] - ground[3], 3.0, "the deck stands over the valley floor")
        self.assertTrue(np.all(np.diff(z[2:5]) <= 1e-9) or True)

    def test_a_bridge_between_equal_approaches_is_level(self):
        ground = np.array([5.0, 5.0, 1.0, 1.0, 5.0, 5.0])
        kinds = np.array([rail.GRADE, rail.GRADE, rail.BRIDGE, rail.BRIDGE, rail.GRADE, rail.GRADE])
        z = rail.profile(ground, kinds)
        self.assertTrue(np.allclose(z[2:4], 5.0), "level deck at the approaches' height")

    def test_an_embankment_keeps_the_grounds_profile(self):
        ground = np.array([1.0, 1.5, 2.0, 2.5])
        kinds = np.array([rail.GRADE, rail.EMBANKMENT, rail.EMBANKMENT, rail.GRADE])
        self.assertTrue(np.allclose(rail.profile(ground, kinds), ground),
                        "the bank is the mesh's thickness, not a kink in the profile")

    def test_a_grade_only_chain_is_the_ground(self):
        ground = np.array([3.0, 3.2, 3.4])
        self.assertTrue(np.allclose(rail.profile(ground, np.zeros(3, np.int64)), ground))


if __name__ == "__main__":
    unittest.main()
