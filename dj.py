#!/usr/bin/env python3
"""
Infinite DJ — CLI

Usage:
  python dj.py analyze <music_dir> [--db <path>] [--force]
  python dj.py inspect <file_or_title> [--db <path>]
  python dj.py library [--db <path>]
  python dj.py cues <file_or_title> [--db <path>]
  python dj.py compatible <file_or_title> [--db <path>] [--top <n>]
  python dj.py mix <track_a> <track_b> --out <file.wav> [--db <path>] [--bars <n>]
  python dj.py sequence [--db <path>] [--start <title>] [--n <int>] [--arc peak|steady|build|wave]
  python dj.py render-set [--db <path>] [--n <int>] [--arc <arc>] --out <file.wav>
"""

import sys
import os
import argparse
import fnmatch
import json

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infinite_dj import TrackDB, analyze_track
from infinite_dj.harmony import camelot_compatibility, bpm_compatibility
from infinite_dj.mixer import (
    TransitionPlan, render_transition, write_mix,
    best_cue_out, best_cue_in,
)
from infinite_dj.engine import StreamEngine
from infinite_dj.sequencer import sequence_energy_arc, sequence_greedy, sequence_for_mixing

SUPPORTED_FORMATS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".m4a")

# `analyze` walks a directory and scoops up any audio — including sets/transitions
# this tool renders into that same folder. These globs skip our own output so it
# doesn't pollute the library. Add more with `analyze --exclude <glob>`, or keep
# renders anyway with `analyze --include-renders`.
RENDER_IGNORE_GLOBS = (
    "infinite_dj_set*.wav",   # full-set renders
    "*_full_set*.wav",
    "*_test_transition*.wav",
    "*.djset.wav",            # explicit render marker
    "_*.wav",                 # scratch renders prefixed with underscore
)


# ── Commands ──────────────────────────────────────────────────────────────────

def _analyze_file_worker(fpath: str):
    """Top-level worker function for ProcessPoolExecutor."""
    import time
    t0 = time.time()
    try:
        meta = analyze_track(fpath, verbose=False)
        elapsed = time.time() - t0
        return (fpath, meta, None, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        return (fpath, None, str(e), elapsed)


def cmd_analyze(args):
    """Scan a directory, analyze any new or changed tracks, save to DB."""
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    db = TrackDB(args.db)
    music_dir = args.music_dir

    if not os.path.isdir(music_dir):
        print(f"Error: {music_dir} is not a directory.")
        sys.exit(1)

    ignore_globs = list(getattr(args, 'exclude', None) or [])
    if not getattr(args, 'include_renders', False):
        ignore_globs += list(RENDER_IGNORE_GLOBS)

    def _is_ignored(fname):
        return any(fnmatch.fnmatch(fname.lower(), g.lower()) for g in ignore_globs)

    files = []
    ignored = 0
    for root, _, fnames in os.walk(music_dir):
        for fname in fnames:
            if not fname.lower().endswith(SUPPORTED_FORMATS):
                continue
            if _is_ignored(fname):
                ignored += 1
                continue
            files.append(os.path.join(root, fname))

    files.sort()
    ignored_note = f" ({ignored} excluded)" if ignored else ""
    print(f"Found {len(files)} audio files in {music_dir}{ignored_note}\n")

    # Filter out cached files fast in main process
    to_analyze = []
    skipped = 0
    for fpath in files:
        if not args.force and not db.needs_analysis(fpath):
            skipped += 1
        else:
            to_analyze.append(fpath)

    if skipped > 0:
        print(f"Skipping {skipped} previously analyzed track(s) (cached).")

    if not to_analyze:
        stats = db.stats()
        print(f"\n{'─'*50}")
        print(f"Done. 0 analyzed, {skipped} skipped.")
        if stats['n']:
            print(f"Library: {stats['n']} tracks | "
                  f"Avg BPM: {stats['avg_bpm']:.1f} | "
                  f"Avg duration: {stats['avg_dur']/60:.1f}m")
        db.close()
        return

    # Determine workers count
    cpu_cores = os.cpu_count() or 4
    num_workers = args.workers if (getattr(args, 'workers', None) and args.workers > 0) else min(cpu_cores, 8)
    if len(to_analyze) == 1:
        num_workers = 1

    print(f"Analyzing {len(to_analyze)} track(s) using {num_workers} parallel worker(s)...\n")

    analyzed = 0
    t_start_batch = time.time()

    if num_workers == 1:
        for i, fpath in enumerate(to_analyze, 1):
            fname = os.path.basename(fpath)
            print(f"[{i}/{len(to_analyze)}] Analyzing: {fname}")
            try:
                meta = analyze_track(fpath, verbose=True)
                db.save(meta)
                analyzed += 1
                print()
            except Exception as e:
                print(f"  ERROR: {e}\n")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(_analyze_file_worker, fpath): fpath for fpath in to_analyze}
            for i, future in enumerate(as_completed(future_to_file), 1):
                fpath, meta, err, elapsed = future.result()
                fname = os.path.basename(fpath)
                if err:
                    print(f"[{i}/{len(to_analyze)}] ERROR ({elapsed:.1f}s): {fname} -> {err}")
                else:
                    db.save(meta)
                    analyzed += 1
                    n_emb = sum(1 for c in meta.cue_points if c.embedding is not None)
                    emb_str = f", {n_emb} CLAP" if n_emb > 0 else ""
                    print(f"[{i}/{len(to_analyze)}] Analyzed ({elapsed:.1f}s): {meta.title} [{meta.bpm:.1f} BPM, {meta.key}{emb_str}]")

    total_batch_time = time.time() - t_start_batch
    stats = db.stats()
    print(f"\n{'─'*50}")
    print(f"Done in {total_batch_time:.1f}s. {analyzed} analyzed, {skipped} skipped.")
    if stats['n']:
        print(f"Library: {stats['n']} tracks | "
              f"Avg BPM: {stats['avg_bpm']:.1f} | "
              f"Avg duration: {stats['avg_dur']/60:.1f}m")
    db.close()



def cmd_library(args):
    """Print a summary table of all analyzed tracks."""
    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if not tracks:
        print("No tracks in database. Run `analyze` first.")
        return

    print(f"\n{'#':<5} {'Title':<40} {'BPM':<7} {'Key':<6} {'Dur':<8} {'Cues':<8} {'Sections'}")
    print("─" * 90)

    for i, t in enumerate(tracks, 1):
        dur = f"{int(t.duration//60)}:{int(t.duration%60):02d}"
        n_in  = sum(1 for c in t.cue_points if c.type == "in")
        n_out = sum(1 for c in t.cue_points if c.type == "out")
        cues = f"{n_in}in/{n_out}out"
        sections = ",".join(s.label for s in t.sections)
        title = t.title[:38] + ".." if len(t.title) > 40 else t.title
        print(f"{i:<5} {title:<40} {t.bpm:<7.1f} {t.key:<6} {dur:<8} {cues:<8} {sections}")


def _find_track(db, query):
    """Find a track by partial title match or file path."""
    tracks = db.load_all()
    query_lower = query.lower()

    # Exact path match
    for t in tracks:
        if t.file_path == os.path.abspath(query):
            return t

    # Partial title match
    matches = [t for t in tracks if query_lower in t.title.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous query '{query}'. Matches:")
        for m in matches:
            print(f"  {m.title}")
        sys.exit(1)

    print(f"No track found matching '{query}'.")
    sys.exit(1)


def cmd_inspect(args):
    """Print full analysis details for a single track."""
    db = TrackDB(args.db)
    track = _find_track(db, args.query)
    db.close()

    dur = f"{int(track.duration//60)}:{int(track.duration%60):02d}"
    print(f"\n{'═'*55}")
    print(f"  {track.title}")
    print(f"{'═'*55}")
    print(f"  File:        {track.file_path}")
    print(f"  Duration:    {dur}")
    print(f"  BPM:         {track.bpm:.2f}  (confidence: {track.bpm_confidence:.2f})")
    print(f"  Key:         {track.key_name} [{track.key}]  (confidence: {track.key_confidence:.2f})")
    print(f"  Beats:       {len(track.beats)}")
    print(f"  Downbeats:   {len(track.downbeats)}")
    print(f"  Phrases:     {len(track.phrases)} boundaries")

    print(f"\n  Sections:")
    for s in track.sections:
        start = f"{int(s.start//60)}:{int(s.start%60):02d}"
        end   = f"{int(s.end//60)}:{int(s.end%60):02d}"
        bar   = "█" * int(s.energy * 20)
        print(f"    {start}-{end}  [{s.label:<10}]  energy {s.energy:.2f}  {bar}")

    print(f"\n  Cue Points:")
    for c in sorted(track.cue_points, key=lambda x: x.timestamp):
        ts  = f"{int(c.timestamp//60)}:{int(c.timestamp%60):02d}"
        tag = "←" if c.type == "in" else "→"
        ph  = "♦" if c.phrase_aligned else " "
        emb = " [CLAP]" if (c.embedding is not None and len(c.embedding) > 0) else ""
        bar = "▓" * int(c.confidence * 15)
        print(f"    {ts}  {tag} {c.type.upper():<3}  {ph}  conf {c.confidence:.2f}{emb}  {bar}")

    print(f"\n  Energy curve (normalized, 1s bins):")
    curve = track.energy_curve
    cols  = min(60, len(curve))
    step  = max(1, len(curve) // cols)
    sampled = curve[::step][:cols]
    bars = " ".join("▁▂▃▄▅▆▇█"[min(7, int(v * 8))] for v in sampled)
    print(f"    {bars}")
    print()


def cmd_cues(args):
    """Print just the cue points for a track in detail."""
    db = TrackDB(args.db)
    track = _find_track(db, args.query)
    db.close()

    print(f"\nCue points for: {track.title}")
    print(f"{'─'*50}")

    ins  = sorted([c for c in track.cue_points if c.type == "in"],  key=lambda x: -x.confidence)
    outs = sorted([c for c in track.cue_points if c.type == "out"], key=lambda x: -x.confidence)

    print("\n  IN points (enter here):")
    for c in ins:
        ts = f"{int(c.timestamp//60)}:{int(c.timestamp%60):02d}.{int((c.timestamp%1)*10)}"
        ph = " [phrase]" if c.phrase_aligned else ""
        emb = " [512D CLAP]" if (c.embedding is not None and len(c.embedding) > 0) else ""
        print(f"    {ts}  conf={c.confidence:.3f}  energy={c.energy:.3f}{ph}{emb}")

    print("\n  OUT points (exit here):")
    for c in outs:
        ts = f"{int(c.timestamp//60)}:{int(c.timestamp%60):02d}.{int((c.timestamp%1)*10)}"
        ph = " [phrase]" if c.phrase_aligned else ""
        emb = " [512D CLAP]" if (c.embedding is not None and len(c.embedding) > 0) else ""
        print(f"    {ts}  conf={c.confidence:.3f}  energy={c.energy:.3f}{ph}{emb}")


def cmd_compatible(args):
    """Find the most harmonically and rhythmically compatible tracks."""
    db = TrackDB(args.db)
    source = _find_track(db, args.query)
    all_tracks = db.load_all()
    db.close()

    top_n = args.top

    results = []
    for t in all_tracks:
        if t.file_path == source.file_path:
            continue
        harm  = camelot_compatibility(source.key, t.key)
        rhyth = bpm_compatibility(source.bpm, t.bpm)
        score = 0.6 * harm + 0.4 * rhyth
        results.append((score, harm, rhyth, t))

    results.sort(key=lambda x: -x[0])

    print(f"\nTop {top_n} matches for: {source.title} [{source.key}, {source.bpm:.1f} BPM]")
    print(f"  {'Score':<7} {'Harm':<7} {'Rhythm':<8} {'Key':<6} {'BPM':<8} Title")
    print(f"  {'─'*65}")

    for score, harm, rhyth, t in results[:top_n]:
        title = t.title[:35] + ".." if len(t.title) > 37 else t.title
        print(f"  {score:.3f}  {harm:.3f}  {rhyth:.3f}   {t.key:<6} {t.bpm:<8.1f} {title}")


def cmd_mix(args):
    """Render a single transition between two tracks."""
    db = TrackDB(args.db)
    track_a = _find_track(db, args.track_a)
    track_b = _find_track(db, args.track_b)
    db.close()

    from infinite_dj.sequencer import find_best_cue_pair, cue_cosine_similarity
    cue_out, cue_in, pair_score = find_best_cue_pair(track_a, track_b)
    sim = cue_cosine_similarity(cue_out, cue_in) if (cue_out and cue_in) else None

    if not cue_out:
        print(f"Warning: no OUT cue points found for '{track_a.title}', using mid-point.")
        from infinite_dj.models import CuePoint
        cue_out = CuePoint(
            timestamp=track_a.duration * 0.6,
            type="out", phrase_aligned=False,
            energy=0.5, confidence=0.1
        )

    if not cue_in:
        print(f"Warning: no IN cue points found for '{track_b.title}', using start.")
        from infinite_dj.models import CuePoint
        cue_in = CuePoint(
            timestamp=max(track_b.downbeats[0] if track_b.downbeats else 0.0, 0.0),
            type="in", phrase_aligned=False,
            energy=0.5, confidence=0.1
        )

    n_bars = getattr(args, 'bars', 8)
    plan = TransitionPlan(
        track_out=track_a,
        track_in=track_b,
        cue_out=cue_out,
        cue_in=cue_in,
        n_mix_bars=n_bars,
    )

    sim_str = f"  |  CLAP Similarity: {sim:.3f}" if sim is not None else ""
    print(f"\nMix plan:")
    print(f"  OUT: {track_a.title} [{track_a.key}, {track_a.bpm:.1f} BPM]")
    print(f"       cue at {cue_out.timestamp:.1f}s (confidence {cue_out.confidence:.2f})")
    print(f"  IN:  {track_b.title} [{track_b.key}, {track_b.bpm:.1f} BPM]")
    print(f"       cue at {cue_in.timestamp:.1f}s (confidence {cue_in.confidence:.2f})")
    print(f"  Cue Match Score: {pair_score:.3f}{sim_str}")
    if plan.beatmatched:
        print(f"  Method: beatmatch ({(plan.stretch_ratio-1)*100:+.1f}% stretch)  |  Mix: {n_bars} bars\n")
    else:
        print(f"  Method: cut (tempos {track_a.bpm:.0f}/{track_b.bpm:.0f} too far to beatmatch)\n")

    result = render_transition(plan)
    write_mix(result, args.out)


def cmd_sequence(args):
    """Print an optimized track sequence without rendering audio."""
    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if not tracks:
        print("No tracks in database. Run `analyze` first.")
        return

    start = None
    if hasattr(args, 'start') and args.start:
        db2 = TrackDB(args.db)
        start = _find_track(db2, args.start)
        db2.close()

    n = getattr(args, 'n', None) or len(tracks)
    arc = getattr(args, 'arc', None)

    if arc:
        seq = sequence_energy_arc(tracks, arc=arc, n_tracks=n)
    else:
        seq = sequence_greedy(tracks, start=start, n_tracks=n)

    seq.describe()


def _serve(audio_path, timeline_path, port):
    from infinite_dj.webserver import serve_player
    httpd, url = serve_player(audio_path, timeline_path, port=port)
    print(f"\n  ▶ Player running at {url}   (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Player stopped.")
        httpd.shutdown()


def _write_and_maybe_serve(args, tracks, clips, audio_path, duration, sr):
    """Write the timeline JSON (if requested) and start the player (if --serve)."""
    want_timeline = getattr(args, "timeline", None) or getattr(args, "serve", False)
    if not want_timeline:
        return
    from infinite_dj.timeline import write_timeline
    tl_path = getattr(args, "timeline", None) or \
        (os.path.splitext(audio_path)[0] + ".timeline.json")
    write_timeline(tl_path, clips, tracks, duration, sr)
    print(f"  Timeline: {tl_path}")
    if getattr(args, "serve", False):
        _serve(audio_path, tl_path, getattr(args, "port", 8765))


def cmd_studio(args):
    """
    Launch the studio: a setup pane to pick tracks and dial in the mix, which
    renders on demand and hands off to the player.
    """
    import tempfile
    from infinite_dj.webserver import serve_app

    if not os.path.isfile(args.db):
        print(f"Database not found: {args.db}\nRun `analyze` first.")
        return

    out_dir = args.out_dir or os.path.join(tempfile.gettempdir(), "infinite_dj_mixes")
    os.makedirs(out_dir, exist_ok=True)

    httpd, url = serve_app(args.db, out_dir, port=args.port)
    print(f"\n  ▶ Studio running at {url}   (Ctrl+C to stop)")
    print(f"    Renders are written to {out_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Studio stopped.")
        httpd.shutdown()


def cmd_serve(args):
    """Serve the local web player for an already-rendered audio + timeline."""
    if not os.path.isfile(args.audio):
        print(f"Audio file not found: {args.audio}")
        return
    tl = args.timeline or (os.path.splitext(args.audio)[0] + ".timeline.json")
    if not os.path.isfile(tl):
        print(f"Timeline JSON not found: {tl}\n"
              f"Render with --timeline/--serve first, or pass --timeline.")
        return
    _serve(args.audio, tl, args.port)


def cmd_render_set(args):
    """
    Build a full mixed set: sequence the tracks and render them onto one
    continuous timeline (solo sections + overlapping beat-locked crossfades).
    """
    import soundfile as sf
    from infinite_dj.mixer import render_set

    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if len(tracks) < 2:
        print("Need at least 2 analyzed tracks.")
        return

    n = getattr(args, 'n', None) or len(tracks)
    arc = getattr(args, 'arc', 'peak')

    print(f"Building mix-optimized {arc} sequence over {n} tracks...")
    seq = sequence_for_mixing(tracks, arc=arc, n_tracks=n)
    seq.describe()

    print(f"\nRendering continuous set ({len(seq.tracks)} tracks)...")
    audio, sr, markers, clips = render_set(seq.tracks, n_mix_bars=16)

    # 16-bit PCM at the source sample rate — matches the library's fidelity
    # without the bloat of 24-bit.
    sf.write(args.out, audio, sr, subtype='PCM_16')
    duration = len(audio) / sr
    mb = os.path.getsize(args.out) / 1024 / 1024

    print(f"\nSet rendered: {args.out}")
    print(f"  {sr} Hz / 16-bit | Duration: {duration/60:.1f} min | Size: {mb:.1f} MB")
    print(f"\n  Transitions:")
    for mk in markers:
        m, s = divmod(mk.time, 60)
        detail = (f"{mk.style} {mk.stretch_pct:+.1f}%"
                  if mk.method == "beatmatch" else f"cut ({mk.style})")
        print(f"    {int(m)}:{s:04.1f}  [{detail}]  {mk.label[:50]}")

    _write_and_maybe_serve(args, tracks, clips, args.out, duration, sr)


def cmd_splice(args):
    """
    Build a collage set: short segments of many tracks spliced together at
    CLAP-serendipitous cut points, filling a target length.
    """
    import soundfile as sf
    from infinite_dj.mixer import render_set, render_collage

    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if len(tracks) < 2:
        print("Need at least 2 analyzed tracks.")
        return

    target_sec = args.length * 60.0
    cooldown = min(4, len(tracks) - 1)

    if args.layers > 1:
        # Structured, variable-pace overlap-add collage (feature / weave / breathe).
        print(f"Composing a {args.layers}-layer collage ({args.length:.0f} min, "
              f"{args.min_seg_bars}-{args.max_seg_bars} bar segments)...")
        audio, sr, markers, clips = render_collage(
            tracks, target_length_sec=target_sec, layers=args.layers,
            min_seg_bars=args.min_seg_bars, max_seg_bars=args.max_seg_bars,
            seed=args.seed, chaos=args.chaos)
    else:
        min_seg, max_seg = args.min_seg, args.max_seg
        if min_seg >= max_seg:
            print("--min-seg must be less than --max-seg.")
            return
        avg_seg = (min_seg + max_seg) / 2.0
        n_seg = int(target_sec / avg_seg) + 4
        print(f"Building {args.arc} splice sequence "
              f"(~{n_seg} segments, {min_seg:.0f}-{max_seg:.0f}s each) for "
              f"{args.length:.0f} min...")
        seq = sequence_for_mixing(tracks, arc=args.arc, n_tracks=n_seg,
                                  allow_repeats=True, cooldown=cooldown)
        print(f"\nSplicing {len(seq.tracks)} segments...")
        audio, sr, markers, clips = render_set(
            seq.tracks, min_seg_sec=min_seg, max_seg_sec=max_seg,
            target_length_sec=target_sec)

    sf.write(args.out, audio, sr, subtype='PCM_16')
    duration = len(audio) / sr
    mb = os.path.getsize(args.out) / 1024 / 1024

    kind = "collage entries" if args.layers > 1 else "splices"
    print(f"\nSplice set rendered: {args.out}")
    print(f"  {sr} Hz / 16-bit | Duration: {duration/60:.1f} min | "
          f"{len(markers)} {kind} | Size: {mb:.1f} MB")
    print(f"\n  Entries (gap since previous shows the varying pace):")
    prev = 0.0
    for mk in markers:
        m, s = divmod(mk.time, 60)
        gap = mk.time - prev
        prev = mk.time
        mode = (mk.method if args.layers > 1 else mk.style)
        name = mk.label.split(" → ")[0].split(" - ")[-1][:40]
        print(f"    {int(m)}:{s:04.1f}  (+{gap:4.0f}s)  [{mode:8}]  {name}")

    _write_and_maybe_serve(args, tracks, clips, args.out, duration, sr)


def cmd_play(args):
    """
    Start the real-time infinite DJ engine.

    Plays indefinitely, mixing tracks in real-time using the
    lookahead scheduler. Press Ctrl+C to stop.

    With --out: renders to a WAV file instead of speakers (useful
    for previewing the engine without audio hardware).
    """
    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if not tracks:
        print("No tracks in database. Run `analyze` first.")
        return

    start_track = None
    if hasattr(args, 'start') and args.start:
        db2 = TrackDB(args.db)
        start_track = _find_track(db2, args.start)
        db2.close()

    arc      = getattr(args, 'arc', 'peak') or 'peak'
    out_file = getattr(args, 'out', None)
    duration = getattr(args, 'duration', None)

    engine = StreamEngine(
        library=tracks,
        arc=arc,
        output_file=out_file,
        max_duration=duration,
    )

    print(f"Starting Infinite DJ with {len(tracks)} tracks...")
    if out_file:
        print(f"Output: {out_file}")
    else:
        print("Press Ctrl+C to stop.\n")

    engine.start(first_track=start_track)

    if out_file:
        mb = os.path.getsize(out_file) / 1024 / 1024 if os.path.exists(out_file) else 0
        print(f"\nDone. {out_file} ({mb:.1f} MB)")

    gaps = engine.state.gap_events
    if gaps:
        total = sum(g["duration"] for g in gaps)
        print(f"\n⚠ {len(gaps)} audible gap(s) this session, {total:.1f}s total:")
        for g in gaps:
            src = f" [{g['source']}]" if g.get("source") else ""
            print(f"    {g['duration']:.2f}s before {g['track']!r}"
                  f" (next queued: {g['next_track']!r}){src}")
    else:
        print("\n✓ No audible gaps this session.")


# ── Mix-corpus mining ─────────────────────────────────────────────────────────

def cmd_mine(args):
    """Measure every announced boundary in a folder of mixes + tracklists."""
    from infinite_dj.mix_corpus import scan_corpus

    db = TrackDB(args.db)
    print(f"Mining mixes in {args.mine_dir} ...")
    summary = scan_corpus(args.mine_dir, db, force=args.force)

    if not summary["n_pairs"]:
        print("\nNo (audio + tracklist) pairs found.")
        print("Drop mix audio in the folder with a sidecar next to each file:")
        print("  somemix.mp3  +  somemix.txt   (lines like '12:04 Artist - Title')")
        print("  somemix.mp3  +  somemix.cue   (standard cue sheet)")
        print("  somemix.mp3  +  somemix.json  ([{\"t\": 724, \"title\": ...}])")
        db.close()
        return

    counts = db.corpus_counts()
    print(f"\nMined {summary['n_mined']} mix(es), skipped {summary['n_skipped']} "
          "already current.")
    print(f"Corpus now: {counts['n_mixes']} mixes, {counts['n_transitions']} "
          f"boundaries, {counts['n_accepted']} measured.")
    print("\nNext: python dj.py corpus")
    db.close()


def cmd_probe(args):
    """Full measurement detail for one boundary — the debugging view."""
    from infinite_dj.mix_corpus import format_probe_detail

    db = TrackDB(args.db)
    try:
        detail = format_probe_detail(args.mix, args.idx, db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        db.close()
        sys.exit(1)
    print(detail)
    db.close()


def cmd_corpus(args):
    """Distributions mined from the corpus, plus the rejection-bias report."""
    from infinite_dj.mix_corpus import corpus_stats, format_corpus_stats

    db = TrackDB(args.db)
    stats = corpus_stats(db, min_confidence=args.min_confidence)
    if not stats["counts"]["n_transitions"]:
        print("No mined transitions yet. Run: python dj.py mine <mix_dir>")
        db.close()
        return
    print(format_corpus_stats(stats))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"\nWrote {args.json}")
    db.close()


def cmd_calibrate(args):
    """Derive engine constants from the mined corpus and write them out."""
    from infinite_dj.calibration import build_from_stats, load, save
    from infinite_dj.mix_corpus import corpus_stats

    if args.show:
        cal = load(args.out)
        print(cal.describe())
        return

    db = TrackDB(args.db)
    stats = corpus_stats(db, min_confidence=args.min_confidence)
    db.close()

    if not stats["counts"]["n_transitions"]:
        print("No mined transitions. Run: python dj.py mine <mix_dir>")
        sys.exit(1)

    cal = build_from_stats(stats)
    save(cal, args.out)
    print(cal.describe())
    print(f"\nWrote {args.out}")
    print("The engine reads this file automatically; values marked (default) are")
    print("unchanged because fewer than the minimum observations back them.")


def cmd_validate(args):
    """Compare the engine's transition shape against the held-out mined corpus."""
    from infinite_dj.validation import compare_to_corpus, format_validation_report

    db = TrackDB(args.db)
    tracks = db.load_all()
    if len(tracks) < 2:
        print("Need at least 2 analyzed tracks. Run: python dj.py analyze <dir>")
        db.close()
        sys.exit(1)

    counts = db.corpus_counts()
    if not counts["n_accepted"]:
        print("No mined transitions to validate against.")
        print("Run: python dj.py mine <mix_dir>")
        print("\nThe measurement ceiling can still be established without a")
        print("corpus — it only needs the local library:")
        print("  python dj.py validate --ceiling-only")
        if not args.ceiling_only:
            db.close()
            sys.exit(1)

    print(f"Rendering and probing {args.pairs} engine transitions "
          "(a few seconds each)...")
    res = compare_to_corpus(db, tracks, min_confidence=args.min_confidence,
                            n_pairs=args.pairs, seed=args.seed)
    db.close()
    print()
    print(format_validation_report(res))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nWrote {args.json}")


def cmd_triage(args):
    """Grade every analyzed track on whether the engine can mix it well."""
    from infinite_dj.library_health import triage

    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if not tracks:
        print("No tracks in database. Run `analyze` first.")
        return

    reports = triage(tracks)
    if args.grade:
        reports = [r for r in reports if r.grade == args.grade]

    counts = {"good": 0, "usable": 0, "reject": 0}
    for r in triage(tracks):
        counts[r.grade] += 1

    print(f"\n{'Grade':<8} {'Title':<38} {'BPM':<7} {'Conf':<6} {'Why'}")
    print("─" * 100)
    for r in reports:
        title = r.title[:36] + ".." if len(r.title) > 38 else r.title
        why = "; ".join(r.reasons)[:44] if r.reasons else "—"
        print(f"{r.grade:<8} {title:<38} {r.bpm:<7.1f} {r.bpm_confidence:<6.2f} {why}")

    total = sum(counts.values())
    print("─" * 100)
    print(f"{total} tracks: {counts['good']} good, "
          f"{counts['usable']} usable, {counts['reject']} reject")
    if counts["reject"]:
        print("\nRejects can't be mixed sensibly — too short, too few cues, or "
              "a tempo the analyzer wasn't sure about.")
    print("Note: a confidently-WRONG beat grid looks fine here. Only ears catch that.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([r.__dict__ for r in reports], fh, indent=2)
        print(f"\nWrote {args.json}")


def cmd_gaps(args):
    """Report what the library is missing, against the engine's real gates."""
    from infinite_dj.library_health import library_gaps

    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if len(tracks) < 2:
        print("Need at least 2 analyzed tracks. Run: python dj.py analyze <dir>")
        return

    g = library_gaps(tracks, target_hours=args.target_hours)

    print(f"\nLIBRARY: {g.n_tracks} tracks, {g.total_hours:.2f}h "
          f"(target {g.target_hours:.0f}h)")
    print(f"  beatmatchable pairs : {g.beatmatchable_frac:.0%}")
    print(f"  Camelot coverage    : {g.camelot_coverage}/24 keys")
    print(f"  energy extremes     : {g.low_energy_tracks} low, "
          f"{g.high_energy_tracks} high")

    if g.bpm_clusters:
        print("\nBPM CLUSTERS")
        for centre, n in g.bpm_clusters:
            print(f"  {centre:6.1f} BPM   {'█' * min(n, 40)} {n}")

    print("\nTRANSITION STYLES REACHABLE")
    total = sum(g.style_counts.values()) or 1
    for name in ("blend", "swap", "fade", "build", "cut"):
        n = g.style_counts.get(name, 0)
        bar = "█" * int(40 * n / total)
        flag = "  ← UNREACHABLE" if n == 0 else ""
        print(f"  {name:<7} {bar:<40} {n / total:5.1%}{flag}")

    if g.findings:
        print("\nWHAT TO FIX")
        for f in g.findings:
            print(f"  • {f}")
    else:
        print("\nNo gaps found — this library exercises the whole engine.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(g.__dict__, fh, indent=2)
        print(f"\nWrote {args.json}")


def cmd_fetch(args):
    """Screen and download Creative Commons tracks from the Internet Archive."""
    from infinite_dj import fetch_archive as fa

    # argparse's `append` action can't carry a default without appending to it.
    args.source = args.source or ["house", "techno"]
    args.genre  = args.genre  or ["house", "techno"]

    spec = fa.Screen(
        bpm_center=args.bpm, bpm_tol=args.bpm_tol,
        min_duration=args.min_duration, max_duration=args.max_duration,
        genres=tuple(args.genre), min_genre_prob=args.min_genre_prob,
        min_danceability=args.min_danceability,
        allow_unscreened=args.allow_unscreened,
        licenses=tuple(args.license.split(",")) if args.license else None,
    )
    target_hours = args.hours

    # `--from-gaps` turns the gap report into the shopping list: aim at the
    # densest BPM cluster the library already has (that's where new tracks buy
    # the most beatmatchable pairs) and fetch only the shortfall.
    if args.from_gaps:
        from infinite_dj.library_health import library_gaps
        db = TrackDB(args.db)
        tracks = db.load_all()
        db.close()
        if len(tracks) < 2:
            print("--from-gaps needs an analyzed library. Run `analyze` first.")
            return
        g = library_gaps(tracks, target_hours=args.gap_target_hours)
        if spec.bpm_center is None and g.bpm_clusters:
            spec.bpm_center = g.bpm_clusters[0][0]
        target_hours = max(0.0, g.target_hours - g.total_hours)
        print(f"From gaps: {g.total_hours:.1f}h of {g.target_hours:.0f}h, "
              f"{g.beatmatchable_frac:.0%} beatmatchable")
        if spec.bpm_center:
            print(f"  targeting {spec.bpm_center:.0f} BPM "
                  f"(±{spec.bpm_tol:.0f}) — the library's densest cluster")

    if target_hours <= 0:
        print("Nothing to fetch — the target is already met.")
        return

    queries = [fa.PRESETS[s] for s in args.source]
    have = fa.existing_sources(args.out)
    print(f"Screening the Archive for {target_hours:.1f}h "
          f"({', '.join(args.source)}); {len(have)} files already fetched.")
    print("Reading Essentia sidecars, not audio — this costs ~26 KB per track.\n")

    def progress(plan, ident):
        print(f"\r  {plan.items_seen:4d} releases · {plan.tracks_seen:5d} tracks "
              f"seen · {len(plan.keep):4d} kept · {plan.hours:5.2f}h   ",
              end="", flush=True)

    plan = fa.build_plan(queries, spec, target_hours=target_hours,
                         max_items=args.max_items, have=have,
                         workers=args.workers,
                         max_bytes=int(args.max_gb * 1024 ** 3),
                         progress=progress)
    print()

    _print_plan(plan, spec)
    _print_yield_advice(plan, spec, target_hours, args)

    if not plan.keep:
        print("\nNothing survived screening — see the advice above.")
        return
    if args.dry_run:
        print("\n(dry run — nothing downloaded). Drop --dry-run to fetch.")
        return

    print(f"\nDownloading {len(plan.keep)} files "
          f"({plan.bytes / 1024 ** 3:.2f} GB) to {args.out}/ ...")
    ok = skipped = failed = 0
    for i, cand in enumerate(plan.keep, 1):
        try:
            dest, fetched = fa.download(cand, args.out)
        except Exception as e:
            failed += 1
            print(f"\r  [{i}/{len(plan.keep)}] FAILED {cand.filename}: {e}")
            continue
        if fetched:
            fa.record_provenance(args.out, cand, dest)
            ok += 1
        else:
            skipped += 1
        print(f"\r  [{i}/{len(plan.keep)}] {ok} new, {skipped} present, "
              f"{failed} failed", end="", flush=True)
    print(f"\n\n{ok} files written to {args.out}/")
    print(f"Attribution recorded in {os.path.join(args.out, 'PROVENANCE.jsonl')}")
    print(f"\nNext:  python dj.py --db {args.db} analyze {args.out}")
    print(f"       python dj.py --db {args.db} triage --grade reject")
    print(f"       python dj.py --db {args.db} gaps")


def _print_plan(plan, spec):
    """Show what screening kept and, just as usefully, what it threw away."""
    from infinite_dj.fetch_archive import _license_code, fold_bpm

    if plan.keep:
        print(f"\nKEEPING {len(plan.keep)} tracks · {plan.hours:.2f}h · "
              f"{plan.bytes / 1024 ** 3:.2f} GB")
        licenses, genres = {}, {}
        for c in plan.keep:
            code = _license_code(c.license)
            licenses[code] = licenses.get(code, 0) + 1
            if c.genre:
                genres[c.genre] = genres.get(c.genre, 0) + 1
        if genres:
            print("  genres  : " + ", ".join(
                f"{k} {v}" for k, v in sorted(genres.items(), key=lambda x: -x[1])))
        print("  licenses: " + ", ".join(
            f"{k or 'none'} {v}" for k, v in sorted(licenses.items(), key=lambda x: -x[1])))
        if any("nd" in (k or "").split("-") for k in licenses):
            print("  note    : NoDerivatives (-nd) tracks are fine to listen to, "
                  "but a published mix built from them is a derivative work.")
        # "top" is the classifier's argmax; "match" is the summed probability of
        # the genres asked for, which is what actually let the track through.
        print(f"\n  {'BPM':>6}  {'top genre':<10} {'match':>5}  "
              f"{'Artist':<24} Title")
        for c in plan.keep[:15]:
            bpm = f"{fold_bpm(c.bpm):.0f}" if c.bpm else "—"
            print(f"  {bpm:>6}  {c.genre[:10]:<10} {c.target_prob:>5.2f}  "
                  f"{(c.artist or c.creator)[:24]:<24} {c.track_title[:32]}")
        if len(plan.keep) > 15:
            print(f"  … and {len(plan.keep) - 15} more")

    if plan.rejected:
        print(f"\nSCREENED OUT ({plan.tracks_seen} tracks across "
              f"{plan.items_seen} releases)")
        for reason, n in sorted(plan.rejected.items(), key=lambda x: -x[1]):
            print(f"  {n:5d}  {reason}")


def _print_yield_advice(plan, spec, target_hours, args):
    """
    A run that stops short did so because the release budget ran out, not
    because the Archive is empty. Say which screen was the binding constraint
    and what to relax, since the yield is the thing that decides whether 24
    hours takes one run or twenty.
    """
    if plan.hours >= target_hours or not plan.items_seen:
        return

    print(f"\nYIELD: {plan.hours:.2f}h of the {target_hours:.1f}h asked for "
          f"— screened {args.max_items} releases per source and ran out.")
    if plan.tracks_seen:
        per = plan.hours / plan.items_seen
        need = (target_hours - plan.hours) / per if per > 0 else float("inf")
        print(f"  At this yield, the rest needs ~{need:.0f} more releases "
              f"screened. Raise --max-items.")

    top = sorted(plan.rejected.items(), key=lambda x: -x[1])
    hints = {
        "no Essentia data": "much of the collection predates the Archive's "
                            "analysis pass — --allow-unscreened keeps those, "
                            "but then --bpm can't filter them",
        "off-tempo": f"--bpm-tol {spec.bpm_tol:.0f} is tight; try a wider "
                     f"window or a --bpm the netlabel corpus has more of "
                     f"(it skews 120-130)",
        "wrong subgenre": "lower --min-genre-prob, or add --genre "
                          "trance/ambient/dnb",
        "duration": "adjust --min-duration / --max-duration",
        "item unreadable": "Archive timeouts — usually transient; re-run",
    }
    for reason, n in top[:2]:
        if reason in hints and n > plan.tracks_seen * 0.15:
            print(f"  Biggest loss: {reason} ({n}) — {hints[reason]}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infinite DJ — music library analysis & mixing engine"
    )
    parser.add_argument("--db", default="infinite_dj.db",
                        help="Path to SQLite database (default: infinite_dj.db)")
    parser.add_argument("--calibration", metavar="PATH",
                        help="Calibration file to use (default: ./calibration.json "
                             "if present; built-in constants otherwise)")

    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a music directory")
    p_analyze.add_argument("music_dir")
    p_analyze.add_argument("--force", action="store_true",
                           help="Re-analyze even if cached")
    p_analyze.add_argument("--workers", type=int, default=None,
                           help="Number of parallel worker processes (default: auto CPU count)")
    p_analyze.add_argument("--exclude", action="append", metavar="GLOB",
                           help="Skip files matching this glob (repeatable)")
    p_analyze.add_argument("--include-renders", action="store_true",
                           help="Also analyze this tool's own rendered sets/transitions")


    # library
    sub.add_parser("library", help="List all analyzed tracks")

    # triage
    p_triage = sub.add_parser(
        "triage", help="Grade each track on whether the engine can mix it well")
    p_triage.add_argument("--grade", choices=["good", "usable", "reject"],
                          help="Show only tracks with this grade")
    p_triage.add_argument("--json", help="Also write the report to this JSON path")

    # gaps
    p_gaps = sub.add_parser(
        "gaps", help="Report what the library is missing, per the engine's gates")
    p_gaps.add_argument("--target-hours", type=float, default=24.0,
                        dest="target_hours",
                        help="Library size goal in hours (default 24)")
    p_gaps.add_argument("--json", help="Also write the report to this JSON path")

    # fetch
    p_fetch = sub.add_parser(
        "fetch",
        help="Download screened Creative Commons tracks from the Internet Archive")
    p_fetch.add_argument("--out", default="music/archive",
                         help="Destination directory (default music/archive)")
    p_fetch.add_argument("--hours", type=float, default=2.0,
                         help="Hours of audio to fetch this run (default 2)")
    p_fetch.add_argument("--from-gaps", action="store_true", dest="from_gaps",
                         help="Target the library's densest BPM cluster and "
                              "fetch its shortfall, per the `gaps` report")
    p_fetch.add_argument("--gap-target-hours", type=float, default=24.0,
                         dest="gap_target_hours",
                         help="Library size goal used by --from-gaps (default 24)")
    p_fetch.add_argument("--source", action="append",
                         choices=sorted(("house", "techno", "experimental",
                                         "electronic")),
                         help="Search preset (repeatable; default house+techno)")
    p_fetch.add_argument("--bpm", type=float,
                         help="Target tempo; only tracks within --bpm-tol are kept")
    p_fetch.add_argument("--bpm-tol", type=float, default=8.0, dest="bpm_tol",
                         help="Tempo tolerance in BPM (default 8)")
    p_fetch.add_argument("--genre", action="append",
                         choices=["ambient", "dnb", "house", "techno", "trance"],
                         help="Essentia sub-genre to accept (repeatable; "
                              "default house+techno)")
    p_fetch.add_argument("--min-genre-prob", type=float, default=0.25,
                         dest="min_genre_prob",
                         help="Minimum summed probability for --genre (default 0.25)")
    p_fetch.add_argument("--min-danceability", type=float, default=0.0,
                         dest="min_danceability",
                         help="Minimum Essentia danceability, 0-1 (default 0, off)")
    p_fetch.add_argument("--min-duration", type=float, default=150.0,
                         dest="min_duration", help="Seconds (default 150)")
    p_fetch.add_argument("--max-duration", type=float, default=900.0,
                         dest="max_duration",
                         help="Seconds (default 900 — longer is usually a DJ mix)")
    p_fetch.add_argument("--license", metavar="CODES",
                         help="Comma-separated allowlist, e.g. by,by-sa,by-nc-sa. "
                              "Default: any declared CC license")
    p_fetch.add_argument("--allow-unscreened", action="store_true",
                         dest="allow_unscreened",
                         help="Keep tracks with no Essentia data (unfiltered)")
    p_fetch.add_argument("--max-items", type=int, default=400, dest="max_items",
                         help="Releases to screen per source (default 400)")
    p_fetch.add_argument("--max-gb", type=float, default=20.0, dest="max_gb",
                         help="Hard download ceiling in GB (default 20)")
    p_fetch.add_argument("--workers", type=int, default=6,
                         help="Concurrent screening requests (default 6)")
    p_fetch.add_argument("--dry-run", action="store_true", dest="dry_run",
                         help="Screen and print the plan without downloading")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Full details for one track")
    p_inspect.add_argument("query", help="Partial title or file path")

    # cues
    p_cues = sub.add_parser("cues", help="Show cue points for a track")
    p_cues.add_argument("query")

    # compatible
    p_compat = sub.add_parser("compatible", help="Find harmonically compatible tracks")
    p_compat.add_argument("query")
    p_compat.add_argument("--top", type=int, default=10)

    # mix
    p_mix = sub.add_parser("mix", help="Render a transition between two tracks")
    p_mix.add_argument("track_a", help="Outgoing track (partial title or path)")
    p_mix.add_argument("track_b", help="Incoming track (partial title or path)")
    p_mix.add_argument("--out", required=True, help="Output WAV file path")
    p_mix.add_argument("--bars", type=int, default=16, help="Mix region length in bars (default 16)")

    # sequence
    p_seq = sub.add_parser("sequence", help="Print an optimized track sequence")
    p_seq.add_argument("--start", help="Starting track (partial title)")
    p_seq.add_argument("--n", type=int, help="Number of tracks in sequence")
    p_seq.add_argument("--arc", choices=["peak", "steady", "build", "wave"],
                       help="Energy arc shape")

    # render-set
    p_set = sub.add_parser("render-set", help="Render a full mixed set to WAV")
    p_set.add_argument("--out", required=True, help="Output WAV file path")
    p_set.add_argument("--n", type=int, help="Number of tracks")
    p_set.add_argument("--arc", default="peak",
                       choices=["peak", "steady", "build", "wave"])
    p_set.add_argument("--timeline", help="Also write a timeline JSON for the web player")
    p_set.add_argument("--serve", action="store_true",
                       help="Launch the interactive web player after rendering")
    p_set.add_argument("--port", type=int, default=8765, help="Player port (default 8765)")

    # splice
    p_splice = sub.add_parser("splice",
                              help="Collage set from short segments of many tracks")
    p_splice.add_argument("--out", required=True, help="Output WAV file path")
    p_splice.add_argument("--length", type=float, default=10.0,
                          help="Target total length in minutes (default 10)")
    p_splice.add_argument("--min-seg", type=float, default=20.0, dest="min_seg",
                          help="Minimum segment length in seconds (default 20)")
    p_splice.add_argument("--max-seg", type=float, default=120.0, dest="max_seg",
                          help="Maximum segment length in seconds (default 120)")
    p_splice.add_argument("--layers", type=int, default=1,
                          help="Overlap-add collage: max tracks sounding at once "
                               "(1 = sequential splices; try 3)")
    p_splice.add_argument("--min-seg-bars", type=int, default=8, dest="min_seg_bars",
                          help="Collage: shortest segment in bars (default 8)")
    p_splice.add_argument("--max-seg-bars", type=int, default=24, dest="max_seg_bars",
                          help="Collage: longest segment in bars (default 24)")
    p_splice.add_argument("--seed", type=int, default=None,
                          help="Collage: random seed for reproducible pacing")
    p_splice.add_argument("--chaos", type=float, default=0.0,
                          help="Collage wildness 0..1: shorter splices, more "
                               "overlap, sub-segments, complementary blends")
    p_splice.add_argument("--arc", default="steady",
                          choices=["peak", "steady", "build", "wave"])
    p_splice.add_argument("--timeline", help="Also write a timeline JSON for the web player")
    p_splice.add_argument("--serve", action="store_true",
                          help="Launch the interactive web player after rendering")
    p_splice.add_argument("--port", type=int, default=8765, help="Player port (default 8765)")

    # studio
    p_studio = sub.add_parser("studio",
                              help="Launch the studio: pick tracks, set the vibe, generate a mix")
    p_studio.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    p_studio.add_argument("--out-dir", dest="out_dir",
                          help="Where rendered mixes are written (default: temp dir)")

    # serve
    p_serve = sub.add_parser("serve", help="Launch the web player for a rendered set")
    p_serve.add_argument("--audio", required=True, help="Rendered audio file")
    p_serve.add_argument("--timeline", help="Timeline JSON (defaults to <audio>.timeline.json)")
    p_serve.add_argument("--port", type=int, default=8765, help="Player port (default 8765)")

    # play
    p_play = sub.add_parser("play", help="Start real-time infinite DJ engine")
    p_play.add_argument("--start", help="Starting track (partial title)")
    p_play.add_argument("--arc", default="peak",
                        choices=["peak", "steady", "build", "wave"],
                        help="Energy arc (default: peak)")
    p_play.add_argument("--out", help="Write to WAV file instead of speakers")
    p_play.add_argument("--duration", type=float,
                        help="Stop after N seconds (useful with --out)")

    # ── Mix-corpus mining (calibration) ──────────────────────────────────────
    p_mine = sub.add_parser(
        "mine", help="Mine a folder of DJ mixes + tracklists for calibration data")
    p_mine.add_argument("mine_dir", metavar="MIX_DIR",
                        help="Folder of mix audio, each with a .txt/.cue/.json "
                             "tracklist sidecar beside it")
    p_mine.add_argument("--force", action="store_true",
                        help="Re-mine mixes even if already current")

    p_probe = sub.add_parser(
        "probe", help="Full measurement detail for one mined boundary")
    p_probe.add_argument("mix", help="Partial mix filename or title")
    p_probe.add_argument("--idx", type=int, required=True,
                         help="Boundary index within the mix (1 = first transition)")

    p_corpus = sub.add_parser(
        "corpus", help="Distributions mined from the mix corpus")
    p_corpus.add_argument("--min-confidence", type=float, default=0.5,
                          dest="min_confidence",
                          help="Drop measurements below this confidence "
                               "(default 0.5; rows are kept in the DB regardless)")
    p_corpus.add_argument("--json", help="Also write the stats to this JSON path")

    p_cal = sub.add_parser(
        "calibrate", help="Derive engine constants from the mined corpus")
    p_cal.add_argument("--out", default="calibration.json",
                       help="Calibration file to write (default calibration.json)")
    p_cal.add_argument("--min-confidence", type=float, default=0.5,
                       dest="min_confidence",
                       help="Confidence floor for measurements used (default 0.5)")
    p_cal.add_argument("--show", action="store_true",
                       help="Print the current calibration and its provenance "
                            "without rebuilding it")

    p_val = sub.add_parser(
        "validate", help="Compare the engine against the mined corpus")
    p_val.add_argument("--min-confidence", type=float, default=0.5,
                       dest="min_confidence",
                       help="Confidence floor for mined measurements (default 0.5)")
    p_val.add_argument("--pairs", type=int, default=12,
                       help="Engine transitions to render and probe (default 12)")
    p_val.add_argument("--seed", type=int, default=0,
                       help="Seed for pair selection and the holdout split")
    p_val.add_argument("--ceiling-only", action="store_true", dest="ceiling_only",
                       help="Establish the measurement ceiling without a corpus")
    p_val.add_argument("--json", help="Also write the report to this JSON path")

    args = parser.parse_args()

    # A calibration file is read automatically; this only overrides the path.
    if getattr(args, "calibration", None):
        from infinite_dj import calibration as _cal_mod
        _cal_mod.set_active(_cal_mod.load(args.calibration))

    dispatch = {
        "analyze":    cmd_analyze,
        "library":    cmd_library,
        "inspect":    cmd_inspect,
        "cues":       cmd_cues,
        "compatible": cmd_compatible,
        "mix":        cmd_mix,
        "sequence":   cmd_sequence,
        "render-set": cmd_render_set,
        "splice":     cmd_splice,
        "studio":     cmd_studio,
        "serve":      cmd_serve,
        "play":       cmd_play,
        "mine":       cmd_mine,
        "probe":      cmd_probe,
        "corpus":     cmd_corpus,
        "calibrate":  cmd_calibrate,
        "validate":   cmd_validate,
    }

    dispatch["triage"] = cmd_triage
    dispatch["gaps"]   = cmd_gaps
    dispatch["fetch"]  = cmd_fetch

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
