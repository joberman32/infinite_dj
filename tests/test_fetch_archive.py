"""
Tests for the Internet Archive fetcher.

Everything here runs offline: the two HTTP helpers (`_get_json`, `_get_gzip_json`)
are the only doors to the network, so monkeypatching them covers the module.
The screening tests care most about *order* — a reject that costs a request when
it could have been free is the difference between screening 24 hours of music in
minutes and in hours.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infinite_dj import fetch_archive as fa  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _cand(**kw):
    base = dict(identifier="lbl001", filename="lbl001-01-track.mp3",
                format="VBR MP3", size=6_000_000, creator="Some Artist",
                album="An EP", track_title="Track One",
                license="http://creativecommons.org/licenses/by-nc-sa/3.0/",
                duration=300.0)
    base.update(kw)
    return fa.Candidate(**base)


def _esshigh(*, electronic=0.95, house=0.6, techno=0.2, ambient=0.1,
             danceable=0.8):
    return {"highlevel": {
        "genre_dortmund": {"all": {"electronic": electronic, "rock": 1 - electronic}},
        "genre_electronic": {"all": {"ambient": ambient, "dnb": 0.0,
                                     "house": house, "techno": techno,
                                     "trance": 0.0}},
        "danceability": {"all": {"danceable": danceable,
                                 "not_danceable": 1 - danceable}},
    }}


def _esslow(*, bpm=126.0, length=300.0):
    return {"rhythm": {"bpm": bpm},
            "tonal": {"key_key": "A", "key_scale": "minor"},
            "metadata": {"audio_properties": {"length": length}}}


@pytest.fixture
def offline(monkeypatch):
    """Serve canned sidecars and count every request that escapes."""
    calls = []

    def fake_gzip_json(url):
        calls.append(url)
        if url.endswith("_esshigh.json.gz"):
            return _esshigh()
        if url.endswith("_esslow.json.gz"):
            return _esslow()
        raise fa.FetchError(url)

    def fake_json(url, params=None):
        calls.append(url)
        raise AssertionError(f"unexpected metadata call: {url}")

    monkeypatch.setattr(fa, "_get_gzip_json", fake_gzip_json)
    monkeypatch.setattr(fa, "_get_json", fake_json)
    return calls


# ── Pure helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,folded", [
    (126.0, 126.0),
    (63.0, 126.0),      # doubled into range
    (252.0, 126.0),     # halved into range
    (95.96, 95.96),
    (180.0, 90.0),      # the top of the window is exclusive
    (0.0, 0.0),
])
def test_fold_bpm_matches_the_engines_window(raw, folded):
    assert fa.fold_bpm(raw) == pytest.approx(folded)


@pytest.mark.parametrize("url,code", [
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "by-nc-sa"),
    ("https://creativecommons.org/licenses/by/4.0/", "by"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("", ""),
])
def test_license_code(url, code):
    assert fa._license_code(url) == code


@pytest.mark.parametrize("raw,seconds", [
    ("272.12", 272.12), (272, 272.0), ("4:32", 272.0),
    ("1:00:30", 3630.0), (None, 0.0), ("n/a", 0.0),
])
def test_parse_length_handles_both_archive_forms(raw, seconds):
    assert fa._parse_length(raw) == pytest.approx(seconds)


def test_first_unwraps_archive_list_fields():
    assert fa._first(["A", "B"]) == "A"
    assert fa._first("A") == "A"
    assert fa._first([]) == ""
    assert fa._first(None) == ""


def test_dest_path_is_safe_and_names_the_library_title():
    """`analyzer` titles a track from its filename stem, so the stem is UI."""
    c = _cand(creator="AC/DC?", track_title='Track: "One"')
    dest = c.dest_path("/lib")
    assert "/" not in Path(dest).name and ":" not in Path(dest).name
    assert Path(dest).name.startswith("AC DC")
    assert dest.endswith(".mp3")


def test_dest_path_does_not_repeat_the_artist():
    c = _cand(creator="Aphex", artist="Aphex", track_title="Aphex")
    assert Path(c.dest_path("/lib")).name == "Aphex.mp3"


def test_compilations_are_filed_under_the_label_but_named_by_the_artist():
    """
    On a compilation the item-level creator reads "AA.VV." and the real name is
    only on the file. The folder groups the release; the filename — which
    becomes the library title — has to carry the artist.
    """
    c = _cand(creator="AA.VV.", artist="Emanuele Fais", track_title="Bara")
    dest = Path(c.dest_path("/lib"))
    assert dest.parent.name == "AA.VV"
    assert dest.name == "Emanuele Fais - Bara.mp3"


def test_artist_falls_back_to_the_item_creator(monkeypatch):
    meta = _item_metadata()
    meta["files"][1]["artist"] = "Real Name"     # a.flac has one, b.mp3 doesn't
    monkeypatch.setattr(fa, "_get_json", lambda url, params=None: meta)
    cands = {c.track_title: c for c in fa.item_candidates("lbl001")}
    assert cands["A"].artist == "Real Name"
    assert cands["B"].artist == "Some Artist"


def test_metadata_calls_get_a_longer_timeout_than_the_default():
    """
    Measured: at the 30s default, ~1 in 3 releases timed out on `/metadata/`
    and was discarded whole. Large releases return large JSON slowly, so this
    endpoint gets its own budget. Regressing it silently costs a third of the
    corpus.
    """
    import inspect
    sig = inspect.signature(fa._get_json)
    assert sig.parameters["timeout"].default >= 60.0
    assert sig.parameters["timeout"].default > \
        inspect.signature(fa._get).parameters["timeout"].default


def test_reason_keys_collapse_to_categories():
    assert fa._reason_key("off-tempo (135 vs 126±8)") == "off-tempo"
    assert fa._reason_key("already have") == "already have"
    assert fa._reason_key("no Essentia data") == "no Essentia data"


# ── Screening ────────────────────────────────────────────────────────────────

def test_a_matching_track_is_kept(offline):
    assert fa.screen(_cand(), fa.Screen(bpm_center=126.0)) == ""


def test_license_and_duration_rejects_cost_no_requests(offline):
    """
    The whole point of the module is cheap rejection. A missing license or an
    obviously wrong duration is known from the search result, so neither may
    touch the network.
    """
    assert fa.screen(_cand(license=""), fa.Screen()) == "no declared license"
    assert fa._reason_key(fa.screen(_cand(duration=45.0), fa.Screen())) == "duration"
    assert fa._reason_key(fa.screen(_cand(duration=4000.0), fa.Screen())) == "duration"
    assert offline == [], "cheap screens must not hit the network"


def test_license_allowlist_filters(offline):
    spec = fa.Screen(licenses=("by", "by-sa"))
    assert fa._reason_key(fa.screen(_cand(), spec)) == "license"
    ok = _cand(license="http://creativecommons.org/licenses/by-sa/4.0/")
    assert fa.screen(ok, spec) == ""


def test_genre_reject_costs_only_the_small_sidecar(monkeypatch):
    """A wrong-genre track must die on the 2 KB file, never reach the 24 KB one."""
    seen = []

    def fake(url):
        seen.append(url)
        assert "_esslow" not in url, "rejected on genre but still fetched BPM"
        return _esshigh(house=0.02, techno=0.01, ambient=0.9)

    monkeypatch.setattr(fa, "_get_gzip_json", fake)
    reason = fa.screen(_cand(), fa.Screen(bpm_center=126.0))
    assert fa._reason_key(reason) == "wrong subgenre"
    assert len(seen) == 1


def test_non_electronic_is_rejected(monkeypatch):
    monkeypatch.setattr(fa, "_get_gzip_json",
                        lambda url: _esshigh(electronic=0.1))
    assert fa._reason_key(fa.screen(_cand(), fa.Screen())) == "not electronic"


def test_danceability_floor_is_applied(monkeypatch):
    monkeypatch.setattr(fa, "_get_gzip_json",
                        lambda url: _esshigh(danceable=0.2))
    spec = fa.Screen(min_danceability=0.5)
    assert fa._reason_key(fa.screen(_cand(), spec)) == "danceability"


def test_bpm_is_only_fetched_when_a_tempo_target_is_set(monkeypatch):
    seen = []

    def fake(url):
        seen.append(url)
        return _esshigh() if "_esshigh" in url else _esslow()

    monkeypatch.setattr(fa, "_get_gzip_json", fake)
    fa.screen(_cand(), fa.Screen())                     # no bpm_center
    assert not any("_esslow" in u for u in seen)
    fa.screen(_cand(), fa.Screen(bpm_center=126.0))
    assert any("_esslow" in u for u in seen)


def test_off_tempo_is_rejected_and_octaves_are_matched(monkeypatch):
    def make(bpm):
        return lambda url: _esshigh() if "_esshigh" in url else _esslow(bpm=bpm)

    monkeypatch.setattr(fa, "_get_gzip_json", make(150.0))
    assert fa._reason_key(fa.screen(_cand(), fa.Screen(bpm_center=126.0))) == "off-tempo"

    # 63 BPM folds to 126 — the engine would see them as the same tempo.
    monkeypatch.setattr(fa, "_get_gzip_json", make(63.0))
    assert fa.screen(_cand(), fa.Screen(bpm_center=126.0)) == ""


def test_a_track_with_no_essentia_data_is_skipped_by_default(monkeypatch):
    def missing(url):
        raise fa.FetchError(url)

    monkeypatch.setattr(fa, "_get_gzip_json", missing)
    assert fa.screen(_cand(), fa.Screen()) == "no Essentia data"
    assert fa.screen(_cand(), fa.Screen(allow_unscreened=True)) == ""


def test_lowlevel_duration_overrides_the_file_listing(monkeypatch):
    """The listing's length can be missing or wrong; Essentia measured it."""
    monkeypatch.setattr(
        fa, "_get_gzip_json",
        lambda url: _esshigh() if "_esshigh" in url else _esslow(length=20.0))
    c = _cand(duration=0.0)
    assert fa._reason_key(fa.screen(c, fa.Screen(bpm_center=126.0))) == "duration"
    assert c.duration == 20.0


# ── Item parsing ─────────────────────────────────────────────────────────────

def _item_metadata():
    return {
        "metadata": {"identifier": "lbl001", "title": "An EP",
                     "creator": ["Some Artist"],
                     "licenseurl": "http://creativecommons.org/licenses/by/4.0/"},
        "files": [
            {"name": "cover.jpg", "format": "JPEG", "size": "1000"},
            {"name": "a.flac", "format": "Flac", "size": "30000000",
             "length": "300.0", "title": "A"},
            {"name": "a.mp3", "format": "VBR MP3", "size": "6000000",
             "length": "300.0", "title": "A"},
            {"name": "b.mp3", "format": "128Kbps MP3", "size": "4000000",
             "length": "4:00", "title": "B"},
            {"name": "b.ogg", "format": "Ogg Vorbis", "size": "3000000",
             "length": "240.0", "title": "B"},
            {"name": "a_esslow.json.gz", "format": "Essentia Low GZ"},
        ],
    }


def test_item_candidates_picks_the_best_format_per_track(monkeypatch):
    monkeypatch.setattr(fa, "_get_json", lambda url, params=None: _item_metadata())
    cands = {c.track_title: c for c in fa.item_candidates("lbl001")}
    assert set(cands) == {"A", "B"}
    assert cands["A"].format == "Flac", "lossless original must win"
    assert cands["B"].format == "Ogg Vorbis", "ranked above 128 Kbps MP3"
    assert cands["A"].creator == "Some Artist"
    assert cands["A"].license.endswith("by/4.0/")
    assert cands["B"].duration == pytest.approx(240.0)


def test_item_candidates_ignores_non_audio(monkeypatch):
    monkeypatch.setattr(fa, "_get_json", lambda url, params=None: _item_metadata())
    names = {c.filename for c in fa.item_candidates("lbl001")}
    assert not any(n.endswith((".jpg", ".json.gz")) for n in names)


# ── Planning ─────────────────────────────────────────────────────────────────

def test_build_plan_stops_at_the_target(monkeypatch):
    monkeypatch.setattr(fa, "search_items",
                        lambda q, limit=500: [{"identifier": f"i{n}"}
                                              for n in range(50)])
    monkeypatch.setattr(fa, "item_candidates",
                        lambda ident, meta=None: [
                            _cand(identifier=ident, filename=f"{ident}-{k}.mp3")
                            for k in range(4)])
    monkeypatch.setattr(fa, "screen", lambda c, spec, fetch=True: "")

    plan = fa.build_plan(["q"], fa.Screen(), target_hours=1.0, workers=1)
    assert plan.hours >= 1.0
    # 300s tracks: 12 reaches 1.0h, and it must not run away past it.
    assert len(plan.keep) == 12


def test_build_plan_counts_rejections_by_category(monkeypatch):
    monkeypatch.setattr(fa, "search_items",
                        lambda q, limit=500: [{"identifier": "i0"}])
    monkeypatch.setattr(fa, "item_candidates",
                        lambda ident, meta=None: [_cand(filename=f"x{k}.mp3")
                                                  for k in range(3)])
    monkeypatch.setattr(fa, "screen",
                        lambda c, spec, fetch=True: "off-tempo (140 vs 126±8)")

    plan = fa.build_plan(["q"], fa.Screen(), target_hours=10.0, workers=1)
    assert plan.keep == []
    assert plan.rejected == {"off-tempo": 3}
    assert plan.tracks_seen == 3 and plan.items_seen == 1


def test_build_plan_skips_what_we_already_have(monkeypatch):
    monkeypatch.setattr(fa, "search_items",
                        lambda q, limit=500: [{"identifier": "lbl001"}])
    monkeypatch.setattr(fa, "item_candidates",
                        lambda ident, meta=None: [_cand()])
    plan = fa.build_plan(["q"], fa.Screen(), target_hours=10.0, workers=1,
                         have={"lbl001/lbl001-01-track.mp3"})
    assert plan.keep == []
    assert plan.rejected == {"already have": 1}


def test_build_plan_deduplicates_items_across_sources(monkeypatch):
    monkeypatch.setattr(fa, "search_items",
                        lambda q, limit=500: [{"identifier": "same"}])
    monkeypatch.setattr(fa, "item_candidates",
                        lambda ident, meta=None: [_cand(identifier=ident)])
    monkeypatch.setattr(fa, "screen", lambda c, spec, fetch=True: "")
    plan = fa.build_plan(["house", "techno"], fa.Screen(),
                         target_hours=10.0, workers=1)
    assert plan.items_seen == 1 and len(plan.keep) == 1


def test_build_plan_survives_an_unreadable_item(monkeypatch):
    def boom(ident, meta=None):
        raise fa.FetchError("HTTP 404")

    monkeypatch.setattr(fa, "search_items",
                        lambda q, limit=500: [{"identifier": "i0"}])
    monkeypatch.setattr(fa, "item_candidates", boom)
    plan = fa.build_plan(["q"], fa.Screen(), target_hours=10.0, workers=1)
    assert plan.rejected == {"item unreadable": 1}


# ── Download + provenance ────────────────────────────────────────────────────

def test_download_skips_a_file_already_present(tmp_path, monkeypatch):
    c = _cand(size=11)
    dest = Path(c.dest_path(str(tmp_path)))
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"hello world")

    def no_network(*a, **k):
        raise AssertionError("re-downloaded a file already on disk")

    monkeypatch.setattr(fa.urllib.request, "urlopen", no_network)
    path, fetched = fa.download(c, str(tmp_path))
    assert path == str(dest) and fetched is False


def test_provenance_round_trips_into_existing_sources(tmp_path):
    c = _cand(bpm=126.4, genre="house")
    dest = c.dest_path(str(tmp_path))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).touch()
    fa.record_provenance(str(tmp_path), c, dest)

    rows = [json.loads(l) for l in
            (tmp_path / "PROVENANCE.jsonl").read_text().splitlines()]
    assert rows[0]["license"].endswith("by-nc-sa/3.0/")
    assert rows[0]["archive_bpm"] == 126.4
    assert rows[0]["identifier"] == "lbl001"
    assert fa.existing_sources(str(tmp_path)) == {"lbl001/lbl001-01-track.mp3"}


def test_existing_sources_is_empty_for_a_fresh_directory(tmp_path):
    assert fa.existing_sources(str(tmp_path)) == set()


def test_existing_sources_survives_a_corrupt_line(tmp_path):
    (tmp_path / "PROVENANCE.jsonl").write_text(
        'not json\n{"source": "https://archive.org/download/x/y.mp3"}\n')
    assert fa.existing_sources(str(tmp_path)) == {"x/y.mp3"}


# ── Presets ──────────────────────────────────────────────────────────────────

def test_every_preset_is_scoped_to_the_cc_licensed_collection():
    """
    Bulk fetching is only defensible because `netlabels` is Creative Commons by
    editorial policy. A preset that escaped that scope would be a licensing bug,
    not a search bug.
    """
    for name, query in fa.PRESETS.items():
        assert "collection:netlabels" in query, name
        assert "mediatype:audio" in query, name


def test_preset_genres_are_essentia_classes():
    """`--genre` values have to be classes the classifier can actually emit."""
    assert set(fa.ELECTRONIC_GENRES) == {"ambient", "dnb", "house", "techno",
                                         "trance"}
    assert set(fa.Screen().genres) <= set(fa.ELECTRONIC_GENRES)
