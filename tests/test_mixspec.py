import pytest

from infinite_dj.mixspec import (
    PACE_PRESETS, library_groups, resolve_params, select_tracks,
)
from infinite_dj.models import TrackMeta
from infinite_dj.timeline import _track_id


def _track(path, title, bpm=128.0, key="8A"):
    return TrackMeta(
        file_path=path, title=title, duration=300.0, bpm=bpm, bpm_confidence=1.0,
        beats=[], downbeats=[], phrases=[], key=key, key_name="A minor",
        key_confidence=1.0, energy_curve=[0.5] * 300, sections=[], cue_points=[],
        analyzed_at=0.0,
    )


# ── library_groups ───────────────────────────────────────────────────────────

def test_groups_by_artist_album_folder():
    tracks = [
        _track("/m/Tom Toms - As Above So Below EP/01_a.mp3", "01 A"),
        _track("/m/Tom Toms - As Above So Below EP/02_b.mp3", "02 B"),
        _track("/m/Vlad Antonenko - Groove/01_deep.mp3", "01 Deep"),
    ]
    groups = library_groups(tracks)
    assert [g["artist"] for g in groups] == ["Tom Toms", "Vlad Antonenko"]
    assert groups[0]["albums"][0]["album"] == "As Above So Below EP"
    assert groups[0]["count"] == 2


def test_falls_back_to_filename_when_folder_is_a_slug():
    # Slug folder, but the filename carries "Artist - Album - NN Title".
    tracks = [_track("/m/aphex-ssw/Aphex Twin - Surfing On Sine Waves - 01 Polygon Window.mp3",
                     "Aphex Twin - Surfing On Sine Waves - 01 Polygon Window")]
    groups = library_groups(tracks)
    assert groups[0]["artist"] == "Aphex Twin"
    assert groups[0]["albums"][0]["album"] == "Surfing On Sine Waves"


def test_track_ids_match_timeline_ids():
    path = "/m/A - B/01.mp3"
    groups = library_groups([_track(path, "01 X")])
    assert groups[0]["albums"][0]["tracks"][0]["id"] == _track_id(path)


# ── resolve_params: serendipity → renderer ───────────────────────────────────

@pytest.mark.parametrize("level,renderer", [
    ("low", "set"), ("medium", "set"), ("high", "collage"), ("insane", "collage"),
])
def test_serendipity_selects_renderer(level, renderer):
    assert resolve_params({"serendipity": level})["renderer"] == renderer


def test_low_is_full_set_without_segment_bounds():
    kwargs = resolve_params({"serendipity": "low"})["kwargs"]
    assert "max_seg_sec" not in kwargs      # whole tracks, not splices


@pytest.mark.parametrize("level", ["low", "medium", "high", "insane"])
def test_every_level_honours_the_requested_length(level):
    # Regression: LOW used to ignore length and render the entire library.
    kwargs = resolve_params({"serendipity": level, "length_min": 3})["kwargs"]
    assert kwargs["target_length_sec"] == 180.0


def test_medium_passes_pace_bounds_through():
    p = resolve_params({"serendipity": "medium", "pace": "choppy", "length_min": 5})
    lo, hi = PACE_PRESETS["choppy"]
    assert p["kwargs"]["min_seg_sec"] == lo and p["kwargs"]["max_seg_sec"] == hi
    assert p["kwargs"]["target_length_sec"] == 300.0


def test_raw_override_beats_the_preset():
    p = resolve_params({"serendipity": "medium", "pace": "choppy",
                        "min_sec": 60, "max_sec": 90})
    assert (p["kwargs"]["min_seg_sec"], p["kwargs"]["max_seg_sec"]) == (60.0, 90.0)


def test_insane_ignores_pace_and_raw_bounds():
    p = resolve_params({"serendipity": "insane", "pace": "long-form",
                        "min_sec": 120, "max_sec": 240})
    assert p["seg"] is None                       # Pace deliberately dropped
    assert p["kwargs"]["min_seg_bars"] == 2       # free to sub-segment
    assert p["kwargs"]["max_seg_bars"] <= 12      # splices stay short
    assert p["kwargs"]["layers"] >= 5             # heavy overlap
    assert p["kwargs"]["chaos"] == 1.0            # full wildness
    assert p["kwargs"]["seed"] is not None        # fresh every render


def test_chaos_rises_with_serendipity():
    high = resolve_params({"serendipity": "high"})["kwargs"]["chaos"]
    insane = resolve_params({"serendipity": "insane"})["kwargs"]["chaos"]
    assert 0 < high < insane == 1.0


def test_high_derives_bars_from_pace():
    tight = resolve_params({"serendipity": "high", "pace": "choppy"})["kwargs"]
    loose = resolve_params({"serendipity": "high", "pace": "long-form"})["kwargs"]
    assert tight["max_seg_bars"] < loose["max_seg_bars"]


def test_unknown_level_and_pace_fall_back():
    p = resolve_params({"serendipity": "bogus", "pace": "nope"})
    assert p["renderer"] == "set" and p["seg"] == PACE_PRESETS["flowing"]


def test_bounds_are_sane_when_inverted():
    p = resolve_params({"serendipity": "medium", "min_sec": 100, "max_sec": 10})
    lo, hi = p["kwargs"]["min_seg_sec"], p["kwargs"]["max_seg_sec"]
    assert hi > lo


# ── select_tracks ────────────────────────────────────────────────────────────

def test_select_tracks_filters_by_id_and_defaults_to_all():
    tracks = [_track("/m/A - B/1.mp3", "1"), _track("/m/A - B/2.mp3", "2")]
    only = select_tracks({"track_ids": [_track_id("/m/A - B/2.mp3")]}, tracks)
    assert [t.file_path for t in only] == ["/m/A - B/2.mp3"]
    assert len(select_tracks({}, tracks)) == 2


def test_low_scales_solo_length_for_short_mixes():
    # A short request shouldn't be forced to the 32-bar default solo per track.
    short = resolve_params({"serendipity": "low", "length_min": 3})["kwargs"]
    long_ = resolve_params({"serendipity": "low", "length_min": 30})["kwargs"]
    assert short["min_solo_bars"] < long_["min_solo_bars"] == 32
