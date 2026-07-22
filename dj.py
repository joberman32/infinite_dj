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
    audio, sr, markers = render_set(seq.tracks, n_mix_bars=16)

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


def cmd_splice(args):
    """
    Build a collage set: short segments of many tracks spliced together at
    CLAP-serendipitous cut points, filling a target length.
    """
    import soundfile as sf
    from infinite_dj.mixer import render_set

    db = TrackDB(args.db)
    tracks = db.load_all()
    db.close()

    if len(tracks) < 2:
        print("Need at least 2 analyzed tracks.")
        return

    target_sec = args.length * 60.0
    min_seg, max_seg = args.min_seg, args.max_seg
    if min_seg >= max_seg:
        print("--min-seg must be less than --max-seg.")
        return

    # Enough segments (with repeats) to fill the target, plus a small buffer.
    avg_seg = (min_seg + max_seg) / 2.0
    n_seg = int(target_sec / avg_seg) + 4
    cooldown = min(4, len(tracks) - 1)

    print(f"Building {args.arc} splice sequence "
          f"(~{n_seg} segments, {min_seg:.0f}-{max_seg:.0f}s each) for "
          f"{args.length:.0f} min...")
    seq = sequence_for_mixing(tracks, arc=args.arc, n_tracks=n_seg,
                              allow_repeats=True, cooldown=cooldown)

    print(f"\nSplicing {len(seq.tracks)} segments...")
    audio, sr, markers = render_set(
        seq.tracks, min_seg_sec=min_seg, max_seg_sec=max_seg,
        target_length_sec=target_sec)

    sf.write(args.out, audio, sr, subtype='PCM_16')
    duration = len(audio) / sr
    mb = os.path.getsize(args.out) / 1024 / 1024

    print(f"\nSplice set rendered: {args.out}")
    print(f"  {sr} Hz / 16-bit | Duration: {duration/60:.1f} min | "
          f"{len(markers)} splices | Size: {mb:.1f} MB")
    print(f"\n  Splices:")
    prev = 0.0
    for mk in markers:
        m, s = divmod(mk.time, 60)
        seg = mk.time - prev
        prev = mk.time
        out_name = mk.label.split(" → ")[0].split(" - ")[-1][:34]
        print(f"    {int(m)}:{s:04.1f}  (+{seg:4.0f}s)  [{mk.style}]  → {out_name}")


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


# ── Argument parsing ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infinite DJ — music library analysis & mixing engine"
    )
    parser.add_argument("--db", default="infinite_dj.db",
                        help="Path to SQLite database (default: infinite_dj.db)")

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
    p_splice.add_argument("--arc", default="steady",
                          choices=["peak", "steady", "build", "wave"])

    # play
    p_play = sub.add_parser("play", help="Start real-time infinite DJ engine")
    p_play.add_argument("--start", help="Starting track (partial title)")
    p_play.add_argument("--arc", default="peak",
                        choices=["peak", "steady", "build", "wave"],
                        help="Energy arc (default: peak)")
    p_play.add_argument("--out", help="Write to WAV file instead of speakers")
    p_play.add_argument("--duration", type=float,
                        help="Stop after N seconds (useful with --out)")

    args = parser.parse_args()

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
        "play":       cmd_play,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
