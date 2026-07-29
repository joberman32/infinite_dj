"""
Mining a corpus of real DJ mixes for calibration statistics.

Reads a folder of mix audio, each with a tracklist sidecar, and measures every
announced boundary with `transition_probe`. Entirely offline — nothing here
fetches anything, by design. You supply the files.

## What comes out, and what doesn't

Recoverable from mix audio plus timestamps:

  - transition duration in beats/bars, and cut-vs-blend
  - key either side, as a distribution over `harmony.camelot_compatibility`
  - `dwell_bars` (announced start to announced start) and `solo_bars` (the
    measured gap between one overlap ending and the next beginning)
  - the mix's tempo trajectory, hence `tempo_step_pct` per boundary

Not recoverable, and no amount of care changes it:

  - **The native tempo delta between A and B.** Beatmatching erases it; the
    applied stretch is never observable in the output. What we get instead is
    how far the DJ let the *set* tempo drift, which the engine currently has no
    model of at all.
  - **Cue-point position within a source track.** That needs the source audio.
  - **Per-band automation phase** (`cp`, mid/high leads). Measurable in
    principle, ~7x compressed in practice — see `transition_probe`.

## The bias to disclose, not correct

Boundaries get rejected when the two tracks are spectrally too alike to
separate. Those are exactly the long, smooth, timbrally-matched blends — the
population we care most about. So the surviving duration distribution is
**biased short**, and `corpus_stats` reports the rejection rate and the
separability of what it dropped so the bias stays visible. Don't quote the
surviving mean as if it were unbiased.

## Free sanity check on real data

The `duration_bars` histogram *must* show peaks near 8/16/32 bars, because DJs
mix on phrase boundaries. If it comes out smooth, the miner is broken. That's how
to trust this without ground truth.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Optional

import numpy as np

# Bump when a measurement change invalidates stored rows; `mix_needs_mining`
# re-mines anything older so old and new measurements never pool.
MINER_VERSION = 1

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".m4a")
SIDECAR_EXTS = (".json", ".cue", ".txt")

# How far either side of an announced timestamp to look for the transition.
#
# Asymmetric on purpose: a tracklist timestamp usually marks "where you first
# hear B", which is already inside the overlap, so the transition more often
# starts *before* the announced time than after it. 60 s back covers a 32-bar
# blend at 128 BPM even when the announced time lands at its very end.
#
# Sized deliberately, not generously. These bounds also set how far the clean
# reference windows sit from the transition, and real tracks are not spectrally
# stationary — a mean spectrum taken 90 s away models the audio next to the
# transition poorly, which starves the decomposition of travel and fills the
# plateaus with noise. Measured against renders with known crossfades, widening
# from 60/25 to 90/45 degraded median duration error from 1.6 s to 6.0 s and
# turned 0.3 s cuts into 8-111 s "blends". Wider is worse here.
SEARCH_BACK = 60.0
SEARCH_FWD = 25.0

# Clean-reference windows. Placed relative to the *neighbouring* boundaries, not
# just this one, and never allowed closer than GUARD to a transition.
REF_LEN = 35.0
REF_GUARD = 6.0
REF_MIN = 15.0


# ── Tracklist parsing ────────────────────────────────────────────────────────

# Optional leading index, then [h:]mm:ss, then an optional separator, then the
# label. Deliberately tolerant — real tracklists are copy-pasted from forums.
_LINE = re.compile(
    r"""^\s*
    (?:\#?\s*\d{1,3}[\.\)]\s*)?          # optional "01." / "1)" index
    \[?
    (?:(?P<h>\d{1,2}):)?                 # optional hours
    (?P<m>\d{1,3}):(?P<s>\d{2})
    \]?
    \s*(?:[-–—:|]\s*)?                   # optional separator
    (?P<label>.+?)\s*$
    """,
    re.VERBOSE,
)

_CUE_INDEX = re.compile(
    r"^\s*INDEX\s+01\s+(?P<m>\d{1,3}):(?P<s>\d{2}):(?P<f>\d{2})", re.IGNORECASE)
_CUE_TITLE = re.compile(r'^\s*TITLE\s+"?(?P<v>[^"]*)"?', re.IGNORECASE)
_CUE_PERF = re.compile(r'^\s*PERFORMER\s+"?(?P<v>[^"]*)"?', re.IGNORECASE)

_SPLITS = (" - ", " – ", " — ", " -- ")


def _split_label(label: str) -> tuple:
    """`"Artist - Title"` -> `(artist, title)`; unsplittable stays as title."""
    for sep in _SPLITS:
        if sep in label:
            a, _, b = label.partition(sep)
            return a.strip(), b.strip()
    return None, label.strip()


def parse_tracklist(path: str) -> tuple:
    """
    Parse a tracklist sidecar into `(entries, problems)`.

    Each entry is `{"t": seconds, "artist": str|None, "title": str,
    "genre": str|None, "raw": str}`. `problems` collects unparsed lines and
    structural warnings — always surfaced, never silently dropped, because a
    tracklist whose timestamps are offset relative to a longer stream corrupts
    every boundary in the file and that has to be visible.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    problems: list = []
    if ext == ".json":
        entries = _parse_json(text, problems)
    elif ext == ".cue":
        entries = _parse_cue(text, problems)
    else:
        entries = _parse_text(text, problems)

    entries.sort(key=lambda e: e["t"])

    # Non-monotone timestamps mean the parse latched onto the wrong numbers.
    deduped = []
    for e in entries:
        if deduped and e["t"] <= deduped[-1]["t"]:
            problems.append(f"non-monotone or duplicate timestamp at {e['t']}s: "
                            f"{e['raw']!r}")
            continue
        deduped.append(e)

    if deduped and deduped[0]["t"] > 60.0:
        problems.append(
            f"first entry starts at {deduped[0]['t']:.0f}s — if this tracklist is "
            "offset relative to a longer stream, every boundary will be wrong")
    return deduped, problems


def _parse_json(text: str, problems: list) -> list:
    try:
        raw = json.loads(text)
    except ValueError as exc:
        problems.append(f"invalid JSON: {exc}")
        return []
    if isinstance(raw, dict):
        raw = raw.get("tracks") or raw.get("tracklist") or []
    out = []
    for item in raw:
        if not isinstance(item, dict) or "t" not in item:
            problems.append(f"skipped JSON entry without 't': {item!r}")
            continue
        title = item.get("title") or ""
        artist = item.get("artist")
        if not title and not artist:
            problems.append(f"skipped JSON entry without a label: {item!r}")
            continue
        out.append({"t": float(item["t"]), "artist": artist, "title": title,
                    "genre": item.get("genre"),
                    "raw": item.get("raw") or f"{artist or ''} - {title}".strip()})
    return out


def _parse_cue(text: str, problems: list) -> list:
    """
    Standard .cue sheets — precise, and what actually ships with DJ mixes.

    `INDEX 01 MM:SS:FF` where FF is frames at 75/sec.
    """
    out: list = []
    cur: dict = {}
    for line in text.splitlines():
        if re.match(r"^\s*TRACK\s+\d+", line, re.IGNORECASE):
            cur = {}
            continue
        m = _CUE_TITLE.match(line)
        if m and "title" not in cur:
            cur["title"] = m.group("v").strip()
            continue
        m = _CUE_PERF.match(line)
        if m and "artist" not in cur:
            cur["artist"] = m.group("v").strip()
            continue
        m = _CUE_INDEX.match(line)
        if m:
            t = int(m.group("m")) * 60 + int(m.group("s")) + int(m.group("f")) / 75.0
            title = cur.get("title", "")
            artist = cur.get("artist")
            if not title and not artist:
                problems.append(f"cue INDEX at {t:.2f}s with no TITLE/PERFORMER")
            out.append({"t": t, "artist": artist, "title": title, "genre": None,
                        "raw": f"{artist or ''} - {title}".strip(" -")})
            cur = {}
    return out


def _parse_text(text: str, problems: list) -> list:
    out = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "//")):
            continue
        m = _LINE.match(line)
        if not m:
            problems.append(f"unparsed line: {line.strip()!r}")
            continue
        h = int(m.group("h") or 0)
        t = h * 3600 + int(m.group("m")) * 60 + int(m.group("s"))
        label = m.group("label").strip()
        # A genre tag in trailing brackets/parens, if the source provides one.
        genre = None
        gm = re.search(r"[\[(]([^\])]{3,30})[\])]\s*$", label)
        if gm and not re.search(r"\d{4}", gm.group(1)):
            genre = gm.group(1).strip()
            label = label[:gm.start()].strip()
        artist, title = _split_label(label)
        out.append({"t": float(t), "artist": artist, "title": title,
                    "genre": genre, "raw": line.strip()})
    return out


def find_sidecar(audio_path: str) -> Optional[str]:
    """The tracklist next to a mix: `<mix>.json` / `.cue` / `.txt`."""
    stem = os.path.splitext(audio_path)[0]
    for ext in SIDECAR_EXTS:
        for cand in (stem + ext, audio_path + ext):
            if os.path.exists(cand):
                return cand
    return None


def find_pairs(mix_dir: str) -> tuple:
    """`([(audio, sidecar), ...], [problems])` for a corpus folder."""
    pairs, problems = [], []
    if not os.path.isdir(mix_dir):
        return pairs, [f"not a directory: {mix_dir}"]
    for name in sorted(os.listdir(mix_dir)):
        path = os.path.join(mix_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in AUDIO_EXTS:
            continue
        side = find_sidecar(path)
        if side is None:
            problems.append(f"no tracklist sidecar for {name} "
                            f"(expected {os.path.splitext(name)[0]}.txt/.cue/.json)")
            continue
        pairs.append((path, side))
    return pairs, problems


# ── Reference-window placement ───────────────────────────────────────────────

def reference_windows(entries: list, i: int, duration: float) -> Optional[tuple]:
    """
    Clean-A and clean-B windows for boundary `i`, or None if there isn't room.

    Positioned relative to the *neighbouring* announced boundaries, not just this
    one: the A window has to sit after the previous transition finished and well
    before this one starts, and the B window after this one but before the next
    begins. Getting this wrong silently contaminates every measurement, so the
    honest response to a cramped layout is to reject the boundary.
    """
    t = entries[i]["t"]
    prev_t = entries[i - 1]["t"] if i > 0 else 0.0
    next_t = entries[i + 1]["t"] if i + 1 < len(entries) else duration

    # A: between the previous boundary and this one's search window.
    a_hi = t - SEARCH_BACK - REF_GUARD
    a_lo = max(prev_t + REF_GUARD, 0.0)
    if a_hi - a_lo < REF_MIN:
        # Not enough clear room before the search window; take what's available
        # between the neighbours instead of reaching into the previous overlap.
        a_hi = max(t - REF_GUARD, a_lo)
    if a_hi - a_lo < REF_MIN:
        return None
    a_lo = max(a_lo, a_hi - REF_LEN)

    # B: after this boundary's forward search, before the next one.
    b_lo = t + SEARCH_FWD + REF_GUARD
    b_hi = min(next_t - REF_GUARD, duration)
    if b_hi - b_lo < REF_MIN:
        b_lo = min(t + REF_GUARD, max(b_hi - REF_MIN, 0.0))
    if b_hi - b_lo < REF_MIN:
        return None
    b_hi = min(b_hi, b_lo + REF_LEN)

    return (a_lo, a_hi), (b_lo, b_hi)


# ── Mining ───────────────────────────────────────────────────────────────────

def _camelot_at(y: np.ndarray, sr: int, win: tuple) -> tuple:
    """`(camelot, confidence)` over a time window, via the existing detector."""
    import librosa

    from .harmony import detect_key

    lo, hi = int(win[0] * sr), int(win[1] * sr)
    seg = y[lo:hi]
    if len(seg) < sr:
        return None, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=512)
    camelot, _name, conf = detect_key(chroma)
    return camelot, float(conf)


def mine_mix(audio_path: str, tracklist_path: str, db, *,
             source: str = "user", quiet: bool = False) -> dict:
    """
    Measure every announced boundary in one mix and persist the rows.

    Returns a summary dict. Rejected boundaries are stored too.
    """
    import librosa

    from .analyzer import BEAT_HOP, SR, _compute_energy_curve
    from .harmony import camelot_compatibility
    from .mix_grid import bars_between, bpm_at, mix_onset_envelope, track_tempo_segments
    from .transition_probe import probe_transition

    entries, problems = parse_tracklist(tracklist_path)
    if len(entries) < 2:
        return {"file": audio_path, "error": "tracklist has fewer than 2 entries",
                "problems": problems, "n_transitions": 0, "n_accepted": 0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr

    late = [e for e in entries if e["t"] >= duration]
    if late:
        problems.append(f"{len(late)} tracklist entries at or past the end of the "
                        f"audio ({duration:.0f}s) — dropped")
        entries = [e for e in entries if e["t"] < duration]
    if len(entries) < 2:
        return {"file": audio_path, "error": "no usable tracklist entries",
                "problems": problems, "n_transitions": 0, "n_accepted": 0}

    onset_env = mix_onset_envelope(y, sr, BEAT_HOP)
    segments = track_tempo_segments(onset_env, duration=duration, sr=sr,
                                    hop=BEAT_HOP)
    energy = _compute_energy_curve(y, sr)

    mix_id = db.save_mix(
        file_path=audio_path,
        title=os.path.splitext(os.path.basename(audio_path))[0],
        duration=duration, tracklist=entries,
        tempo_segments=[s.to_dict() for s in segments],
        energy_curve=energy, source=source, miner_version=MINER_VERSION,
    )
    db.delete_mix_transitions(mix_id)

    rows, results = [], []
    for i in range(1, len(entries)):
        entry = entries[i]
        t = entry["t"]
        refs = reference_windows(entries, i, duration)
        if refs is None:
            rows.append({
                "mix_id": mix_id, "idx": i, "announced_t": t,
                "status": "rejected", "reject_reason": "no_clean_reference",
                "track_a": entries[i - 1]["raw"], "track_b": entry["raw"],
                "genre": entry.get("genre"), "confidence": 0.0,
            })
            continue
        ref_a, ref_b = refs

        bpm_before, _ = bpm_at(segments, ref_a[1])
        bpm_after, _ = bpm_at(segments, ref_b[0])
        probe_bpm = bpm_before if np.isfinite(bpm_before) else 128.0

        res = probe_transition(y, sr, t, ref_a=ref_a, ref_b=ref_b, bpm=probe_bpm,
                               search_back=SEARCH_BACK, search_fwd=SEARCH_FWD)
        results.append(res)

        key_before, kcb = _camelot_at(y, sr, ref_a)
        key_after, kca = _camelot_at(y, sr, ref_b)
        cam = (camelot_compatibility(key_before, key_after)
               if key_before and key_after else None)

        step = None
        if np.isfinite(bpm_before) and np.isfinite(bpm_after) and bpm_before > 0:
            step = (bpm_after - bpm_before) / bpm_before * 100.0

        # dwell: announced start to announced start. solo: the measured gap
        # between the previous overlap ending and this one starting — the real
        # analogue of render_set's `min_solo_bars`.
        next_t = entries[i + 1]["t"] if i + 1 < len(entries) else duration
        dwell = bars_between(segments, t, next_t)
        solo = None
        prev_ok = next((r for r in reversed(results[:-1])
                        if r.status == "ok"), None)
        if res.status == "ok" and prev_ok is not None and \
                np.isfinite(prev_ok.t_end) and res.t_start > prev_ok.t_end:
            solo = bars_between(segments, prev_ok.t_end, res.t_start)

        bands = res.bands or {}

        def band(name, attr):
            f = bands.get(name)
            v = getattr(f, attr, None) if f is not None and getattr(f, "ok", False) \
                else None
            return None if v is None or not np.isfinite(v) else round(float(v), 4)

        rows.append({
            "mix_id": mix_id, "idx": i, "announced_t": t,
            "status": res.status, "reject_reason": res.reject_reason,
            "track_a": entries[i - 1]["raw"], "track_b": entry["raw"],
            "t_start": _num(res.t_start), "t_end": _num(res.t_end),
            "t_center": _num(res.t_center), "t_bass": _num(res.t_bass),
            "duration_sec": _num(res.duration_sec),
            "duration_beats": _num(res.duration_beats),
            "duration_bars": _num(res.duration_bars),
            "is_cut": int(bool(res.is_cut)),
            "bpm_before": _num(bpm_before), "bpm_after": _num(bpm_after),
            "tempo_step_pct": _num(step),
            "band_low_cp": res.band_phase.get("low"),
            "band_mid_cp": res.band_phase.get("mid"),
            "band_high_cp": res.band_phase.get("high"),
            "band_low_w": band("low", "span_1090"),
            "band_mid_w": band("mid", "span_1090"),
            "band_high_w": band("high", "span_1090"),
            "key_before": key_before, "key_after": key_after,
            "key_conf_before": round(kcb, 4), "key_conf_after": round(kca, 4),
            "camelot_score": cam,
            "dwell_bars": _num(dwell), "solo_bars": _num(solo),
            "genre": entry.get("genre"),
            "confidence": res.confidence,
            "sub_scores": json.dumps(res.sub_scores or {}),
        })

    db.save_transitions(rows)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    if not quiet:
        print(f"  {os.path.basename(audio_path)}: {n_ok}/{len(rows)} boundaries "
              f"measured ({len(entries)} tracks, {len(segments)} tempo segments)")
        for p in problems[:5]:
            print(f"    ! {p}")
        if len(problems) > 5:
            print(f"    ! ... and {len(problems) - 5} more tracklist problems")
    return {"file": audio_path, "mix_id": mix_id, "problems": problems,
            "n_transitions": len(rows), "n_accepted": n_ok,
            "n_tracks": len(entries), "duration": duration}


def _num(v):
    """None for non-finite, else a rounded float — sqlite wants real numbers."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None


def _dist(values: list) -> Optional[dict]:
    """Percentile summary of a sample, or None if it's empty."""
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    a = np.asarray(vals)
    return {
        "n": len(vals),
        "mean": round(float(a.mean()), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
    }


def phrase_multiple_histogram(bars: list, edges=(2, 4, 8, 12, 16, 24, 32, 48, 64)):
    """
    Histogram of transition lengths bucketed around phrase multiples.

    The free correctness check on real data: DJs mix on phrase boundaries, so
    this must show peaks near 8/16/32. A smooth histogram means the miner is
    measuring noise.
    """
    vals = [float(b) for b in bars if b is not None and np.isfinite(float(b))
            and float(b) > 0]
    counts = {e: 0 for e in edges}
    for v in vals:
        nearest = min(edges, key=lambda e: abs(e - v))
        counts[nearest] += 1
    return counts, len(vals)


def corpus_stats(db, min_confidence: float = 0.5) -> dict:
    """
    The distributions calibration is drawn from, plus the bias disclosure.

    `rejected` is not incidental bookkeeping: rejection correlates with exactly
    the transitions we most want (long, smooth, timbrally-matched blends), so the
    accept rate and the separability of the dropped population are part of the
    result, not a footnote.
    """
    counts = db.corpus_counts()
    rows = db.load_transitions(min_confidence=min_confidence, status="ok")
    all_ok = db.load_transitions(min_confidence=None, status="ok")

    blends = [r for r in rows if not r["is_cut"]]
    hist, n_hist = phrase_multiple_histogram([r["duration_bars"] for r in blends])

    by_genre: dict = {}
    for r in blends:
        g = (r.get("genre") or "").strip().lower()
        if g:
            by_genre.setdefault(g, []).append(r["duration_bars"])

    cam = [r["camelot_score"] for r in rows if r["camelot_score"] is not None]
    stats = {
        "counts": counts,
        "min_confidence": min_confidence,
        "n_at_confidence": len(rows),
        "n_blends": len(blends),
        "cut_rate": (round(sum(1 for r in rows if r["is_cut"]) / len(rows), 4)
                     if rows else None),
        "duration_bars": _dist([r["duration_bars"] for r in blends]),
        "duration_beats": _dist([r["duration_beats"] for r in blends]),
        "dwell_bars": _dist([r["dwell_bars"] for r in rows]),
        "solo_bars": _dist([r["solo_bars"] for r in rows]),
        "tempo_step_pct": _dist([r["tempo_step_pct"] for r in rows]),
        "camelot_score": _dist(cam),
        "camelot_zero_rate": (round(sum(1 for c in cam if c == 0.0) / len(cam), 4)
                              if cam else None),
        "confidence": _dist([r["confidence"] for r in all_ok]),
        "phrase_histogram": hist,
        "phrase_histogram_n": n_hist,
        "band_phase_low": _dist([r["band_low_cp"] for r in blends]),
        "band_phase_mid": _dist([r["band_mid_cp"] for r in blends]),
        "band_phase_high": _dist([r["band_high_cp"] for r in blends]),
        "by_genre": {g: _dist(v) for g, v in sorted(by_genre.items())},
        "rejected": {
            "reasons": db.reject_reasons(),
            "accept_rate": (round(counts["n_accepted"] / counts["n_transitions"], 4)
                            if counts["n_transitions"] else None),
        },
    }
    return stats


def format_corpus_stats(stats: dict) -> str:
    """Human-readable corpus report for `dj.py corpus`."""
    c = stats["counts"]
    out = [
        "── Mined DJ-mix corpus ──",
        f"mixes: {c['n_mixes']}   boundaries: {c['n_transitions']}   "
        f"measured: {c['n_accepted']}   "
        f"at confidence >= {stats['min_confidence']}: {stats['n_at_confidence']}",
    ]
    ar = stats["rejected"]["accept_rate"]
    if ar is not None:
        out.append(f"accept rate: {ar:.1%}")
    if stats["rejected"]["reasons"]:
        out.append("rejections: " + ", ".join(
            f"{k}={v}" for k, v in stats["rejected"]["reasons"].items()))
        out.append("  NOTE: rejection favours timbrally-similar pairs, which are")
        out.append("  the long smooth blends — the surviving durations are biased")
        out.append("  SHORT. Do not read the mean as unbiased.")

    def line(name, d, unit=""):
        if not d:
            out.append(f"{name:<16} (no data)")
            return
        out.append(f"{name:<16} n={d['n']:<5} p10={d['p10']:<8} p50={d['p50']:<8} "
                   f"p90={d['p90']:<8} mean={d['mean']}{unit}")

    out.append("")
    line("duration_bars", stats["duration_bars"])
    line("duration_beats", stats["duration_beats"])
    line("dwell_bars", stats["dwell_bars"])
    line("solo_bars", stats["solo_bars"])
    line("tempo_step_pct", stats["tempo_step_pct"])
    line("camelot_score", stats["camelot_score"])
    if stats["cut_rate"] is not None:
        out.append(f"{'cut_rate':<16} {stats['cut_rate']:.1%}")
    if stats["camelot_zero_rate"] is not None:
        out.append(f"{'camelot=0 rate':<16} {stats['camelot_zero_rate']:.1%}  "
                   "(transitions harmony.py scores as incompatible)")

    out.append("")
    out.append("phrase-multiple histogram of blend lengths (bars) —")
    out.append("  must peak near 8/16/32; a flat spread means the miner is broken")
    if stats["phrase_histogram_n"]:
        total = stats["phrase_histogram_n"]
        for k, v in stats["phrase_histogram"].items():
            bar = "#" * int(round(40 * v / max(1, total)))
            out.append(f"  {k:>3} |{bar} {v}")
    else:
        out.append("  (no data)")

    if stats["by_genre"]:
        out.append("")
        out.append("by genre (not pooled — transition style is genre-dependent):")
        for g, d in stats["by_genre"].items():
            if d:
                out.append(f"  {g:<20} n={d['n']:<4} p50={d['p50']} bars")
    return "\n".join(out)


def format_probe_detail(mix_query: str, idx: int, db) -> str:
    """
    Everything measured at one boundary, per band, with every sub-score.

    This is the view to live in when a corpus looks wrong: a rejected boundary
    tells you *which* guard fired and how close the others were, which is the
    difference between fixing a threshold and guessing at one.
    """
    mixes = db.load_mixes()
    if not mixes:
        raise FileNotFoundError("no mixes mined yet — run `dj.py mine <dir>` first")
    q = mix_query.lower()
    match = [m for m in mixes if q in (m["file_path"] or "").lower()
             or q in (m["title"] or "").lower()]
    if not match:
        names = ", ".join(os.path.basename(m["file_path"]) for m in mixes[:8])
        raise FileNotFoundError(f"no mined mix matching {mix_query!r} (have: {names})")
    mix = match[0]

    rows = [r for r in db.load_transitions(min_confidence=None, status=None)
            if r["mix_id"] == mix["id"]]
    row = next((r for r in rows if r["idx"] == idx), None)
    if row is None:
        have = ", ".join(str(r["idx"]) for r in rows)
        raise FileNotFoundError(f"mix has no boundary {idx} (have: {have})")

    out = [f"── {os.path.basename(mix['file_path'])}  boundary #{idx} ──",
           f"  A: {row['track_a']}",
           f"  B: {row['track_b']}",
           f"  announced at {_fmt_t(row['announced_t'])}",
           ""]
    if row["status"] != "ok":
        out.append(f"  STATUS: rejected — {row['reject_reason']}")
    else:
        out += [
            f"  STATUS: ok   confidence {row['confidence']}",
            f"  transition {_fmt_t(row['t_start'])} -> {_fmt_t(row['t_end'])} "
            f"(centre {_fmt_t(row['t_center'])})",
            f"  duration   {row['duration_bars']} bars / "
            f"{row['duration_beats']} beats"
            + ("   [CUT]" if row["is_cut"] else ""),
            f"  offset from announced: "
            f"{(row['t_center'] or 0) - row['announced_t']:+.1f}s",
        ]
        if row["t_bass"] is not None:
            out.append(f"  bass swap at {_fmt_t(row['t_bass'])}")
    out.append("")
    out.append("  tempo   before {} / after {}  (step {})".format(
        _fmt_num(row["bpm_before"]), _fmt_num(row["bpm_after"]),
        _fmt_num(row["tempo_step_pct"], "%")))
    out.append("  key     before {} / after {}  camelot score {}".format(
        row["key_before"] or "?", row["key_after"] or "?",
        _fmt_num(row["camelot_score"])))
    out.append("  dwell   {} bars   solo {} bars".format(
        _fmt_num(row["dwell_bars"]), _fmt_num(row["solo_bars"])))

    out.append("")
    out.append("  per-band 10-90% span (seconds) and phase within the transition:")
    for band in ("low", "mid", "high"):
        out.append("    {:<5} span={:<9} phase={}".format(
            band, _fmt_num(row[f"band_{band}_w"]),
            _fmt_num(row[f"band_{band}_cp"])))
    out.append("    (band phase is diagnostic only — it does not reliably track")
    out.append("     _make_profile's cp; see transition_probe's module notes)")

    if row["sub_scores"]:
        out.append("")
        out.append("  sub-scores (confidence is their geometric mean):")
        for k, v in row["sub_scores"].items():
            bar = "#" * int(round(20 * float(v)))
            out.append(f"    {k:<8} {v:<7} |{bar}")
    return "\n".join(out)


def _fmt_t(t) -> str:
    if t is None or not np.isfinite(float(t)):
        return "?"
    t = float(t)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _fmt_num(v, suffix: str = "") -> str:
    if v is None:
        return "?"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "?" if not np.isfinite(f) else f"{round(f, 3)}{suffix}"


def scan_corpus(mix_dir: str, db, *, force: bool = False,
                quiet: bool = False) -> dict:
    """Mine every (audio, sidecar) pair in a folder."""
    pairs, problems = find_pairs(mix_dir)
    if not quiet:
        for p in problems:
            print(f"  ! {p}")
    mined, skipped = [], 0
    for audio, side in pairs:
        if not force and not db.mix_needs_mining(audio, MINER_VERSION):
            skipped += 1
            continue
        mined.append(mine_mix(audio, side, db, quiet=quiet))
    return {"n_pairs": len(pairs), "n_mined": len(mined), "n_skipped": skipped,
            "problems": problems, "results": mined}
