"""
The `fade`/`build` branch in `choose_transition_style` used to be decided by
the raw sign of `eo - ei`. Mining real DJ mixes found that comparison carries
almost no signal near a tie (median margin 0.055; see CHANGELOG 2026-08-08) —
`_match_entry` picks the incoming cue by *minimizing* |eo - ei|, so the branch
was effectively a coin flip on measurement noise. These tests pin the
deliberate, reproducible replacement: a seeded draw inside TIE_MARGIN, an
undisturbed sign comparison outside it, and a per-pair salt so a repeated pair
doesn't draw the same coin every time.
"""
from infinite_dj.mixer import choose_transition_style, TIE_MARGIN, plan_transition
from infinite_dj.models import CuePoint


def _style(eo, ei, **kw):
    return choose_transition_style(
        CuePoint(0.0, "out", True, eo, 1.0),
        CuePoint(0.0, "in", True, ei, 1.0),
        beatmatched=True, **kw,
    )


def test_decisive_margins_are_unaffected():
    """Outside TIE_MARGIN, the sign of eo - ei still decides, as before."""
    assert _style(0.8, 0.5).name == "fade"     # margin +0.3
    assert _style(0.5, 0.8).name == "build"    # margin -0.3


def test_same_inputs_always_draw_the_same_style():
    """plan_transition must stay replayable: same cues in, same style out."""
    eo, ei = 0.52, 0.50   # margin 0.02, inside TIE_MARGIN
    first = _style(eo, ei).name
    for _ in range(20):
        assert _style(eo, ei).name == first


def test_tie_zone_is_not_pinned_to_the_raw_sign():
    """
    Somewhere inside the tie zone the seeded draw must pick 'build' even though
    eo >= ei — otherwise this is just the old sign comparison relabeled.
    """
    names = set()
    for ei in (0.500, 0.501, 0.502, 0.503, 0.504, 0.505, 0.506, 0.507, 0.508,
               0.509, 0.510, 0.511, 0.512, 0.513, 0.514, 0.515, 0.516, 0.517):
        names.add(_style(0.52, ei).name)
        if len(names) == 2:
            break
    assert names == {"fade", "build"}


def test_exact_tie_is_roughly_balanced_over_many_draws():
    """
    At a true tie (margin 0) the draw should not be systematically biased
    toward one style — sweep the cue timestamp as the only varying seed input.
    """
    fades = 0
    n = 200
    for i in range(n):
        style = choose_transition_style(
            CuePoint(float(i) * 0.01, "out", True, 0.5, 1.0),
            CuePoint(float(i) * 0.017, "in", True, 0.5, 1.0),
            beatmatched=True,
        )
        fades += style.name == "fade"
    assert 0.30 * n < fades < 0.70 * n


def test_seed_extra_changes_the_draw_for_otherwise_identical_cues():
    """occurrence/track-identity salt must be able to flip a tied draw."""
    eo, ei = 0.5, 0.5
    outcomes = {
        _style(eo, ei, seed_extra=(f"a{i}", "b", 0)).name for i in range(10)
    }
    assert outcomes == {"fade", "build"}


def _track(path, bpm, cue_energy):
    """Minimal TrackMeta for plan_transition — no audio I/O required."""
    from infinite_dj.models import TrackMeta

    beat = 60.0 / bpm
    downbeats = [round(k * beat * 4, 3) for k in range(40)]
    cues = [CuePoint(downbeats[10], "out", True, cue_energy, 0.9),
            CuePoint(downbeats[2], "in", True, cue_energy, 0.9)]
    return TrackMeta(
        file_path=path, title=path, duration=downbeats[-1] + 4 * beat,
        bpm=bpm, bpm_confidence=0.9, beats=[], downbeats=downbeats, phrases=[],
        key="8A", key_name="A minor", key_confidence=0.8,
        energy_curve=[cue_energy] * 40, sections=[], cue_points=cues,
        analyzed_at=0.0,
    )


def test_plan_transition_salts_repeated_pairs_by_occurrence():
    """
    plan_transition must forward (track paths, occurrence) so a pair repeating
    later in a set (radio mode) isn't guaranteed to repeat its style choice.
    Both tracks sit at the same tied energy, so occurrence is the only thing
    that can move the draw.
    """
    a, b = _track("a.wav", 124.0, 0.5), _track("b.wav", 124.0, 0.5)
    names = {
        plan_transition(a, b, 0.0, a.duration, min_solo_bars=1,
                        occurrence=occ).style.name
        for occ in range(15)
    }
    assert names == {"fade", "build"}
