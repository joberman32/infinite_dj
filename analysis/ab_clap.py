"""
A/B harness: how much does CLAP change what the mixer actually plays?

Ablation: render the same spec twice — once with the library's Section and
CuePoint embeddings intact, once with them nulled on a copy so every
CLAP-aware selector falls back to its production heuristic (energy ordering,
random-among-compatible, and the CLAP term dropped from track scoring).
Nothing in the stored library is mutated.

Two arms, because CLAP feeds two different layers and they are *supposed* to
move in opposite directions:

  1. Track ordering  — consecutive tracks should get MORE similar (continuity).
  2. Splice overlap  — simultaneous layers should get LESS similar (contrast).

Both conditions are scored with the ORIGINAL embeddings so the two arms are
judged on the same yardstick.

    python analysis/ab_clap.py            # both arms
    python analysis/ab_clap.py --order    # fast arm only (no audio rendered)
"""
import copy, os, re, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np

from infinite_dj.db import TrackDB
from infinite_dj.harmony import camelot_compatibility
from infinite_dj.mixer import render_collage, _clap_cos
from infinite_dj.sequencer import (
    cue_cosine_similarity, find_best_cue_pair, sequence_for_mixing,
)

DB = "combined.db"
START_RE = re.compile(r"@(\d+)s\]$")


def strip_embeddings(tracks):
    """Copy the library with every embedding nulled; originals untouched."""
    out = []
    for t in tracks:
        t2 = copy.copy(t)
        t2.sections = [copy.copy(s) for s in t.sections]
        for s in t2.sections:
            s.embedding = None
        t2.cue_points = [copy.copy(c) for c in t.cue_points]
        for c in t2.cue_points:
            c.embedding = None
        out.append(t2)
    return out


# ── Arm 1: track ordering ────────────────────────────────────────────────────

def score_order(order, by_path):
    sims, harms = [], []
    for a, b in zip(order, order[1:]):
        ta, tb = by_path[a.file_path], by_path[b.file_path]
        c_out, c_in, _ = find_best_cue_pair(ta, tb)
        s = cue_cosine_similarity(c_out, c_in)
        if s is not None:
            sims.append(s)
        harms.append(camelot_compatibility(ta.key, tb.key))
    return (np.mean(sims) if sims else float("nan")), np.mean(harms)


def arm_ordering(orig, off, by_path):
    print("=" * 72)
    print("1. TRACK ORDERING — sequence_for_mixing (higher sim = more continuity)")
    print("=" * 72)
    print(f"{'arc':<8} {'cond':<5} {'consec CLAP':>12} {'harmonic':>9} {'order match':>12}")
    for arc in ("peak", "steady", "build"):
        on = sequence_for_mixing(orig, arc=arc, n_tracks=len(orig)).tracks
        of = sequence_for_mixing(off, arc=arc, n_tracks=len(off)).tracks
        s_on, h_on = score_order(on, by_path)
        s_off, h_off = score_order(of, by_path)
        match = sum(1 for a, b in zip(on, of) if a.file_path == b.file_path) \
            / max(1, min(len(on), len(of)))
        print(f"{arc:<8} {'ON':<5} {s_on:>12.3f} {h_on:>9.2f}")
        print(f"{arc:<8} {'OFF':<5} {s_off:>12.3f} {h_off:>9.2f} {match:>11.0%}")
    print()


# ── Arm 2: splice placement ──────────────────────────────────────────────────

def resolve(markers, clips, by_path):
    """markers and clips are appended 1:1 in place(), and the marker carries the
    section start — so each clip maps to an exact original Section."""
    out = []
    for m, c in zip(markers, clips):
        t = by_path.get(c["track"])
        mm = START_RE.search(m.label)
        sec = None
        if t and mm and t.sections:
            want = int(mm.group(1))
            sec = min(t.sections, key=lambda s: abs(s.start - want))
        out.append((c, sec))
    return out


def overlap_sims(pairs):
    """CLAP similarity between every pair of concurrently sounding sections."""
    sims = []
    for i, (a, sa) in enumerate(pairs):
        if not sa or not sa.embedding:
            continue
        for b, sb in pairs[i + 1:]:
            if b["out_start"] >= a["out_end"]:      # clips are time-sorted
                break
            if not sb or not sb.embedding:
                continue
            sims.append(_clap_cos(sa.embedding, sb.embedding))
    return sims


def arm_placement(orig, by_path, seconds=120, seeds=(3, 17, 42, 101)):
    print("=" * 72)
    print("2. SPLICE PLACEMENT — render_collage (lower sim = more contrast)")
    print("=" * 72)
    pool = orig[:8]
    pool_off = strip_embeddings(pool)
    configs = {
        "calm  chaos=0.0 ": dict(layers=2, min_seg_bars=12, max_seg_bars=28, chaos=0.0),
        "mid   chaos=0.35": dict(layers=3, min_seg_bars=6,  max_seg_bars=18, chaos=0.35),
        "wild  chaos=0.8 ": dict(layers=4, min_seg_bars=3,  max_seg_bars=12, chaos=0.8),
    }
    print(f"{'config':<18} {'cond':<4} {'clips':>6} {'ovl':>5} {'overlapCLAP':>12} "
          f"{'identical':>10} {'pickMatch':>10}", flush=True)
    for name, prof in configs.items():
        acc = {"ON": [], "OFF": []}
        identical, matches = 0, []
        for seed in seeds:
            _, _, m_on, c_on = render_collage(pool, target_length_sec=seconds,
                                              seed=seed, **prof)
            _, _, m_of, c_of = render_collage(pool_off, target_length_sec=seconds,
                                              seed=seed, **prof)
            sig_on = [(c["track"], round(c["out_start"], 1)) for c in c_on]
            sig_of = [(c["track"], round(c["out_start"], 1)) for c in c_of]
            if sig_on == sig_of:
                identical += 1
            n = min(len(sig_on), len(sig_of))
            matches.append(sum(1 for x, y in zip(sig_on, sig_of) if x == y) / max(1, n))
            acc["ON"].append((len(c_on), overlap_sims(resolve(m_on, c_on, by_path))))
            acc["OFF"].append((len(c_of), overlap_sims(resolve(m_of, c_of, by_path))))
        for cond in ("ON", "OFF"):
            nclips = np.mean([r[0] for r in acc[cond]])
            sims = [s for r in acc[cond] for s in r[1]]
            sim = f"{np.mean(sims):.3f}" if sims else "—"
            ident = f"{identical}/{len(seeds)}" if cond == "ON" else ""
            pm = f"{np.mean(matches):.0%}" if cond == "ON" else ""
            print(f"{name:<18} {cond:<4} {nclips:>6.1f} {len(sims):>5} {sim:>12} "
                  f"{ident:>10} {pm:>10}", flush=True)
    print()


def main():
    db = TrackDB(DB)
    orig = db.load_all()
    db.close()
    by_path = {t.file_path: t for t in orig}
    n_emb = sum(1 for t in orig for s in t.sections if s.embedding)
    print(f"library: {len(orig)} tracks, {n_emb} sections carry CLAP embeddings\n")
    if n_emb == 0:
        print("No embeddings found — install requirements-clap.txt and re-analyze.")
        return
    arm_ordering(orig, strip_embeddings(orig), by_path)
    if "--order" not in sys.argv:
        arm_placement(orig, by_path)


if __name__ == "__main__":
    main()
