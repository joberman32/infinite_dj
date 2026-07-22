from infinite_dj.models import TrackMeta
from infinite_dj.timeline import build_timeline, _track_id


def _track(path, title, bpm, key):
    return TrackMeta(
        file_path=path, title=title, duration=300.0, bpm=bpm, bpm_confidence=1.0,
        beats=[], downbeats=[], phrases=[], key=key, key_name="A minor",
        key_confidence=1.0, energy_curve=[0.5] * 300, sections=[], cue_points=[],
        analyzed_at=0.0,
    )


def _clip(path, start, end, mode="blend", fo=4.0):
    return {"track": path, "title": "x", "out_start": start, "out_end": end,
            "fade_in": 0.0, "fade_out": fo, "mode": mode, "section": "peak",
            "bpm": 128.0, "key": "8A"}


def test_build_timeline_shape_and_join():
    tracks = [_track("a.mp3", "Art - Al - 01 Alpha", 128.0, "8A"),
              _track("b.mp3", "Art - Al - 02 Beta", 130.0, "9A")]
    clips = [_clip("a.mp3", 0.0, 40.0), _clip("b.mp3", 36.0, 90.0)]
    tl = build_timeline(clips, tracks, duration=90.0, sr=44100)

    assert tl["version"] == 1 and tl["duration"] == 90.0
    assert set(tl["tracks"].keys()) == {_track_id("a.mp3"), _track_id("b.mp3")}
    # clips reference tracks by short id and are time-sorted
    assert [c["start"] for c in tl["clips"]] == [0.0, 36.0]
    assert tl["clips"][0]["track"] == _track_id("a.mp3")
    # per-track metadata is joined in (title stripped to track name)
    assert tl["tracks"][_track_id("a.mp3")]["title"] == "01 Alpha"
    assert tl["tracks"][_track_id("a.mp3")]["key"] == "8A"
    # embeddings are never leaked into the export
    assert "embedding" not in str(tl)


def test_build_timeline_handles_missing_track_meta():
    # clip references a path not in `tracks` — should still export using inline data
    clips = [_clip("ghost.mp3", 0.0, 10.0)]
    tl = build_timeline(clips, [], duration=10.0)
    assert len(tl["clips"]) == 1
    assert _track_id("ghost.mp3") in tl["tracks"]
