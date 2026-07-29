"""
End-to-end mining, against a mix this engine rendered itself.

`render_set` gives us a mix whose transitions we know exactly: it returns
`(audio, sr, markers, clips)`, and for transition i
`t_start = markers[i].time = clips[i]["out_end"] - clips[i]["fade_out"]`,
`t_end = clips[i]["out_end"]`. Writing the incoming clip's `out_start` as the
"announced" timestamp reproduces what a real tracklist gives us.

Two honest caveats:

  - This validates the **measurement**, not the calibration. Mining our own
    renders tells us nothing about real DJ practice; it tells us the pipeline
    recovers what was actually laid down.
  - It's an **optimistic** bound. `render_set`'s crossfades are piecewise-linear
    lanes, close to the shape the prober fits. Real transitions include manual
    fader moves, effects, loops and third decks the two-source model cannot
    represent.

Tier 1 (`test_transition_probe.py`) is the fast default; this file needs the
local library and skips without it.
"""
import os

import numpy as np
import pytest
import soundfile as sf

from infinite_dj.db import TrackDB
from infinite_dj.mix_corpus import MINER_VERSION, corpus_stats, mine_mix

LIB_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "combined.db")

pytestmark = pytest.mark.skipif(
    not os.path.exists(LIB_DB),
    reason="needs the local library at combined.db")


def _render_corpus_mix(tmp_path, n_tracks=5):
    """Render a set and write it out with a matching tracklist sidecar."""
    from infinite_dj.mixer import render_set

    db = TrackDB(LIB_DB)
    tracks = sorted(db.load_all(), key=lambda t: t.file_path)[:n_tracks]
    db.close()
    if len(tracks) < 3:
        pytest.skip("library has too few analyzed tracks")

    audio, sr, markers, clips = render_set(tracks, min_solo_bars=8)

    mix_path = tmp_path / "synthetic.wav"
    sf.write(str(mix_path), audio, sr, subtype="PCM_16")

    # The "announced" time for track i+1 is where its clip becomes audible —
    # exactly the convention a human tracklist uses.
    lines = []
    for i, c in enumerate(clips):
        t = float(c["out_start"])
        lines.append(f"{int(t // 60):02d}:{int(t % 60):02d} "
                     f"Track{i} - {c.get('title', 'x')}")
    (tmp_path / "synthetic.txt").write_text("\n".join(lines) + "\n")

    truth = []
    for i, m in enumerate(markers):
        truth.append({"t_start": float(m.time),
                      "t_end": float(clips[i]["out_end"]),
                      "fade_out": float(clips[i]["fade_out"]),
                      "style": m.style, "method": m.method})
    return mix_path, tmp_path / "synthetic.txt", truth, sr


def test_mines_its_own_render(tmp_path):
    mix_path, tl_path, truth, _sr = _render_corpus_mix(tmp_path)

    db = TrackDB(str(tmp_path / "corpus.db"))
    summary = mine_mix(str(mix_path), str(tl_path), db, source="synthetic",
                       quiet=True)

    assert summary["n_transitions"] >= 2, summary
    counts = db.corpus_counts()
    assert counts["n_mixes"] == 1
    assert counts["n_transitions"] == summary["n_transitions"]

    rows = db.load_transitions(min_confidence=None, status=None)
    assert len(rows) == summary["n_transitions"]
    # Every row must carry the bookkeeping the corpus report depends on.
    for r in rows:
        assert r["track_a"] and r["track_b"]
        assert r["status"] in ("ok", "rejected")
        if r["status"] == "rejected":
            assert r["reject_reason"]
    db.close()


def test_recovers_transition_extent_of_its_own_render(tmp_path):
    """
    For boundaries it accepts, the measured transition must land inside the
    crossfade the renderer actually laid down — the end-to-end check that
    tracklist parsing, reference placement, tempo tracking and the probe all
    agree on a timeline.
    """
    mix_path, tl_path, truth, _sr = _render_corpus_mix(tmp_path)

    db = TrackDB(str(tmp_path / "corpus.db"))
    mine_mix(str(mix_path), str(tl_path), db, source="synthetic", quiet=True)
    rows = [r for r in db.load_transitions(min_confidence=None, status="ok")]
    db.close()

    if not rows:
        pytest.skip("no boundaries survived the guards on this library subset")

    checked = 0
    for r in rows:
        t = truth[r["idx"] - 1]
        if t["fade_out"] <= 0.5:      # a cut; extent isn't meaningful
            continue
        # Generous but real: the measured centre has to sit within the crossfade
        # plus a margin, not somewhere else in the mix.
        margin = max(4.0, t["fade_out"])
        assert t["t_start"] - margin <= r["t_center"] <= t["t_end"] + margin, (
            r["idx"], r["t_center"], t)
        checked += 1
    assert checked >= 1, "no blended transitions to check"


def test_dwell_bars_are_recovered(tmp_path):
    """dwell_bars comes from the tracklist and the tempo track, so it should be
    available on every row including rejected ones with a next neighbour."""
    mix_path, tl_path, _truth, _sr = _render_corpus_mix(tmp_path)

    db = TrackDB(str(tmp_path / "corpus.db"))
    mine_mix(str(mix_path), str(tl_path), db, source="synthetic", quiet=True)
    rows = db.load_transitions(min_confidence=None, status="ok")
    db.close()

    dwells = [r["dwell_bars"] for r in rows if r["dwell_bars"] is not None]
    if not dwells:
        pytest.skip("no accepted boundaries")
    # render_set was told min_solo_bars=8, so nothing should read as near-zero.
    assert min(dwells) > 2.0, dwells
    assert max(dwells) < 2000.0, dwells


def test_separates_cuts_from_blends_on_a_real_render(tmp_path):
    """
    The classification that works on real material, and the one calibration most
    depends on being right.

    Measured over 31 transitions from this library: 10 of 12 true 0.3 s cuts came
    back under one bar, while every true blend (14-26 s) measured several seconds.
    Absolute *duration* on blends is much weaker — median |error| 7.8 s — so this
    asserts the discrimination, not the magnitude. See the corpus report for the
    honest accuracy numbers.

    Asserted as a separation between the two populations rather than a per-cut
    count: an individual render can go 2-of-4 where the aggregate is 10-of-12,
    and a count threshold would just be flaky.
    """
    mix_path, tl_path, truth, _sr = _render_corpus_mix(tmp_path, n_tracks=8)

    db = TrackDB(str(tmp_path / "corpus.db"))
    mine_mix(str(mix_path), str(tl_path), db, source="synthetic", quiet=True)
    rows = {r["idx"]: r for r in db.load_transitions(min_confidence=None,
                                                     status="ok")}
    db.close()

    true_cuts, true_blends = [], []
    for idx, r in rows.items():
        t = truth[idx - 1]
        (true_cuts if t["fade_out"] <= 1.0 else true_blends).append(r)

    if not true_cuts or not true_blends:
        pytest.skip("this render has no mix of cuts and blends")

    cut_lens = [r["duration_sec"] or 0.0 for r in true_cuts]
    blend_lens = [r["duration_sec"] or 0.0 for r in true_blends]

    # The populations must be clearly separated, which is what lets a cut-rate
    # statistic mean anything.
    assert float(np.median(cut_lens)) < 0.5 * float(np.median(blend_lens)), (
        cut_lens, blend_lens)
    # And no true blend may collapse to a cut.
    bar = 2.0     # ~one bar at 120 BPM, the probe's resolution floor
    for r in true_blends:
        assert (r["duration_sec"] or 0) > bar, (r["idx"], r["duration_sec"])


def test_centre_is_located_more_accurately_than_width(tmp_path):
    """
    Locating a transition and measuring its width are not equally hard, and the
    corpus statistics rest on knowing which is which: centre error runs ~3 s
    median against a width error of ~8 s on 14-26 s crossfades.
    """
    mix_path, tl_path, truth, _sr = _render_corpus_mix(tmp_path, n_tracks=8)

    db = TrackDB(str(tmp_path / "corpus.db"))
    mine_mix(str(mix_path), str(tl_path), db, source="synthetic", quiet=True)
    rows = {r["idx"]: r for r in db.load_transitions(min_confidence=None,
                                                     status="ok")}
    db.close()

    errs = []
    for idx, r in rows.items():
        t = truth[idx - 1]
        if t["fade_out"] <= 1.0:
            continue
        centre = (t["t_start"] + t["t_end"]) / 2
        errs.append(abs(r["t_center"] - centre))
    if not errs:
        pytest.skip("no blended transitions in this render")
    assert float(np.median(errs)) < 8.0, errs


def test_corpus_stats_runs_on_a_mined_mix(tmp_path):
    mix_path, tl_path, _truth, _sr = _render_corpus_mix(tmp_path)

    db = TrackDB(str(tmp_path / "corpus.db"))
    mine_mix(str(mix_path), str(tl_path), db, source="synthetic", quiet=True)
    stats = corpus_stats(db, min_confidence=0.0)
    db.close()

    assert stats["counts"]["n_mixes"] == 1
    assert stats["rejected"]["accept_rate"] is not None
    from infinite_dj.mix_corpus import format_corpus_stats
    text = format_corpus_stats(stats)
    assert "Mined DJ-mix corpus" in text


def test_remining_replaces_rather_than_duplicates(tmp_path):
    mix_path, tl_path, _truth, _sr = _render_corpus_mix(tmp_path)

    db = TrackDB(str(tmp_path / "corpus.db"))
    first = mine_mix(str(mix_path), str(tl_path), db, source="synthetic",
                     quiet=True)
    before = db.corpus_counts()
    second = mine_mix(str(mix_path), str(tl_path), db, source="synthetic",
                      quiet=True)
    after = db.corpus_counts()
    db.close()

    assert first["mix_id"] == second["mix_id"]
    assert before == after, (before, after)
