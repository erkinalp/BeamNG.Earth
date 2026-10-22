"""Railways: the map's rail ways as track scenery (docs/ROADMAP.md: no trains, switches or drivable rail yet).

The map's roads layer carries railways too (subtype "rail"; mapdata draws them on the world map in RAIL_STYLE),
but roads.road_specs leaves everything that is not a road out, so nothing in the level drew a railway at all: a rail
overpass over a street did not exist, and DECK_ROAD_CLEAR had nothing to hold it to (2026-10-22). This module reads
the same features as track: the ballast the rails sit in, the two running rails at gauge, a third rail where the
line is that kind, and the bridges as structure (the road tiles lift them like a road deck).

The map has no electrification rule (a roads feature's rules are surface, flags, width, level, max_speed_kmh, and
Overture's rail columns carry none), so it comes from the class the way road widths come from theirs: metro and
light rail run a third rail, mainline rail overhead — which is later (masts and wires).

Dimensions: 1435 mm standard gauge; a single track's ballast shoulder 3.2-3.6 m, 3.4 m here; a 70 mm rail head
standing 170 mm over the sleepers (UIC54 / 60E1 class); the third rail 90 mm wide, 550 mm outside the running rail
(750 V DC, on insulators under a cover board).
"""
import numpy as np

RAIL_GAUGE = 1.435          # m, standard gauge: between the running rails' inner faces' centres
RAIL_W = 0.07               # m, a rail's head
RAIL_TOP = 0.17             # m, the rail's head over the sleepers (its foot is wider: 0.20 m either side)
BALLAST_W = 3.4             # m, the ballast bed's width at its crown, one track
BALLAST_TOP = 0.30          # m, the ballast's crown over the ground beside the track (a low prism)
THIRD_RAIL_W = 0.09         # m
THIRD_RAIL_OFF = 0.55       # m, the third rail's centre outside the running rail's centre
THIRD_RAIL_CLASSES = frozenset({"subway", "light_rail", "tram", "funicular", "monorail"})


def is_rail(feats):
    """The indices of a map cell's roads layer that are railways, in feature order."""
    sub = feats.attrs.get("subtype") if feats is not None else None
    if sub is None:
        return []
    return [i for i, s in enumerate(sub) if str(s) == "rail"]


def electrification(cls):
    """'third_rail' or 'overhead' from a rail way's class (the map carries no electrified rule)."""
    return "third_rail" if str(cls).strip().lower() in THIRD_RAIL_CLASSES else "overhead"


def strips(third_rail=True):
    """The track's cross-section as (lateral offset m from the centreline, width m, part), roadwards out:
    the ballast bed, the two running rails at gauge, and the third rail where the line has one. A part is
    'ballast', 'rail' or 'conductor'; roads.offset_polyline lays each one out along the centreline."""
    out = [(0.0, BALLAST_W, "ballast"),
           (-RAIL_GAUGE / 2.0, RAIL_W, "rail"),
           (RAIL_GAUGE / 2.0, RAIL_W, "rail")]
    if third_rail:
        out.append((RAIL_GAUGE / 2.0 + THIRD_RAIL_OFF, THIRD_RAIL_W, "conductor"))
    return out


def lift(part):
    """m a part's strip stands over the ground the track is laid on: the ballast's crown, and the rails on it
    (the third rail just under the running rails' head, as a 750 V conductor rail is)."""
    if part == "ballast":
        return 0.0
    if part == "rail":
        return BALLAST_TOP
    return BALLAST_TOP - 0.02


def ranges(s, hit):
    """(s0, s1) runs of the samples where hit is True, in arc length s (monotonic), merging runs closer than
    MERGE_M: the stretches of a way that are one thing. Pure."""
    s = np.asarray(s, np.float64)
    hit = np.asarray(hit, bool)
    if len(s) < 2 or not hit.any():
        return []
    idx = np.flatnonzero(hit)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    out = []
    for run in runs:
        a, b = float(s[run[0]]), float(s[run[-1]])
        if out and a - out[-1][1] <= MERGE_M:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


MERGE_M = 0.5              # m: hit runs closer than this are one crossing


def _point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from each point (px, py) to the segment a-b (arrays), vectorised. Pure."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    ln = vx * vx + vy * vy
    t = np.where(ln > 1e-12, (wx * vx + wy * vy) / np.maximum(ln, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    dx, dy = wx - t * vx, wy - t * vy
    return np.hypot(dx, dy)


def crossing_ranges(xy, s, roads, margin=0.0):
    """The stretches of a rail way at grade that a road surface crosses: s ranges where the way is within a road's
    half width (+ margin) of its centreline. xy (n, 2) the way's samples; s (n,) their arc lengths; roads:
    [(points (k, 2), half width m)]. Pure — the caller decides which parts may be flushed (a bridge or an
    embankment over a road is not a level crossing)."""
    hit = np.zeros(len(xy), bool)
    for pts, half in roads:
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2 or half <= 0:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            d = _point_segment_distance(xy[:, 0], xy[:, 1], a[0], a[1], b[0], b[1])
            hit |= d <= float(half) + margin
    return ranges(s, hit)


GRADE, BRIDGE, TUNNEL, EMBANKMENT = 0, 1, 2, 3


def profile(ground, kinds):
    """One height profile along a rail chain, so its parts join without a step: grade and tunnel follow the ground,
    a bridge runs straight between the heights of the ground at its two ends (which are the neighbours' own heights,
    at the abutments), and an embankment keeps the ground's height under its earthwork — the bank is a mesh
    thickness, not a kink in the profile. ground: (n,) DEM z at the samples; kinds: (n,) GRADE/BRIDGE/TUNNEL/
    EMBANKMENT. Returns z (n,), continuous at every boundary by construction. Pure.

    The abutment is the whole point: a bridge resolved apart from its approaches agrees with them only by luck, and
    a step there is what a player sees first (2026-10-22)."""
    z = np.asarray(ground, np.float64).copy()
    k = np.asarray(kinds, np.int64)
    if len(z) < 2:
        return z
    i = 0
    while i < len(k):
        if k[i] != BRIDGE:
            i += 1
            continue
        j = i
        while j + 1 < len(k) and k[j + 1] == BRIDGE:
            j += 1
        # the ends take the neighbours' heights: the abutments meet the approach surfaces exactly
        left = z[i - 1] if i > 0 else z[i]
        right = z[j + 1] if j + 1 < len(z) else z[j]
        z[i:j + 1] = np.linspace(left, right, j - i + 1)
        i = j + 1
    return z


def centreline(xy):
    """A rail way's centreline as float64 (n, 2) in metres; empty for fewer than two points."""
    a = np.asarray(xy, np.float64)[:, :2]
    return a if len(a) >= 2 else np.zeros((0, 2))
