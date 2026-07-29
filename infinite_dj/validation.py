"""
Comparing what this engine does against what real DJs did.

## Why this can't be a per-transition hit rate

The brief asked for: hold out ~15% of mined transitions, run the engine on the
same (track A, track B) pair, compare cue points within +/-1 beat, report the hit
rate. Two things make that literal form impossible, both consequences of
mix-audio-only mining rather than choices:

  - **We don't have the corpus's source tracks.** A mined transition is a
    measurement of a mix, not a pair of files we can hand to `plan_transition`.
    So there is nothing to run the engine "on the same pair" of, and the
    comparison has to be distributional.
  - **Cue position inside a source track is not recoverable at all** from mix
    audio, so there is no DJ cue point to compare our cue point against.

What *is* comparable is the shape of the transitions: how long they are, how
often they're cuts, how long a track runs before mixing out, and how far out of
key DJs are willing to mix. This module compares those distributions.

## The measurement ceiling is the number that matters

A low agreement score could mean the engine mixes unlike a human, or it could
mean the probe can't measure well enough to tell. Those call for opposite
responses, so the report separates them:

  - **Ceiling** — render transitions whose length we set exactly, measure them
    through the identical pipeline, and see how often the measurement lands
    within a given tolerance of the truth. This is the best any engine could
    score, and it is *not* high: probe duration error runs ~8 s median on
    14-26 s crossfades, which at 128 BPM is roughly 17 beats.
  - **Agreement** — the same statistic computed against the mined corpus.

Reporting agreement without the ceiling would make a measurement limitation look
like an engine defect. Read the two together, and read the gap between them as
the only part attributable to the engine.

## Both sides go through the same estimator

The engine's transitions are **rendered and then probed**, not read analytically
off its automation lanes. The probe has known biases — a ~0.787 scale factor on
symmetric profiles, ~2x over-reporting on asymmetric ones, spectral leakage
between bands — and those cancel only if both sides pass through them. Reading
the lanes analytically would silently attribute every one of those biases to the
DJ.
"""
from __future__ import annotations

import os
import random
import tempfile

import numpy as np

# Tolerance tiers for "did the engine choose what the DJ chose", in beats.
# The +/-1 beat tier is the brief's; the wider two are reported because at this
# measurement precision the tight tier says more about the probe than the engine.
TOLERANCE_BEATS = (1.0, 4.0, 8.0)

# Rendering and probing a pair costs a few seconds, so the default sample is
# small. Raise it when the corpus is large enough for the comparison to bind.
DEFAULT_PAIRS = 12

HOLDOUT_FRAC = 0.15


def holdout_split(rows: list, frac: float = HOLDOUT_FRAC, seed: int = 0) -> tuple:
    """
    `(train, test)` split of mined transitions.

    Held out per *mix*, not per transition: transitions from one mix share a DJ,
    a tracklist and a tempo track, so splitting within a mix would leak.
    """
    by_mix: dict = {}
    for r in rows:
        by_mix.setdefault(r["mix_id"], []).append(r)
    mix_ids = sorted(by_mix)
    rng = random.Random(seed)
    rng.shuffle(mix_ids)

    n_test = max(1, int(round(len(mix_ids) * frac))) if len(mix_ids) > 1 else 0
    test_ids = set(mix_ids[:n_test])
    train = [r for m in mix_ids if m not in test_ids for r in by_mix[m]]
    test = [r for m in test_ids for r in by_mix[m]]
    return train, test


def _render_and_probe_set(tracks: list, workdir: str, tag: int,
                          quiet: bool = True) -> list:
    """
    Render a multi-track set with the real engine, then mine it back.

    A **set**, not a pair, and that matters for the ceiling: a two-track render
    leaves only one boundary with little clean audio either side, and the probe's
    reference windows have nowhere good to sit. Measured on this library, pairs
    gave a blend error of ~30 beats where 6-9 track sets gave ~17 — so scoring the
    ceiling on pairs would report a limitation of the harness as a limitation of
    the method. Real mixes look like sets.

    Deliberately routed through `mine_mix` — file, tracklist sidecar and all — so
    the engine's side traverses exactly the code path the corpus did. Returns a
    list of `(measured_row, truth)`.
    """
    import soundfile as sf

    from .db import TrackDB
    from .mix_corpus import mine_mix
    from .mixer import render_set

    audio, sr, markers, clips = render_set(tracks)
    if not markers:
        return []

    stem = os.path.join(workdir, f"set_{tag}")
    wav, tl = stem + ".wav", stem + ".txt"
    sf.write(wav, audio, sr, subtype="PCM_16")
    with open(tl, "w") as fh:
        for i, c in enumerate(clips):
            t = float(c["out_start"])
            fh.write(f"{int(t // 60):02d}:{int(t % 60):02d} T{i} - x\n")

    truths = {}
    for i, (m, c) in enumerate(zip(markers, clips), start=1):
        length = float(c["fade_out"])
        bpm = float(tracks[i - 1].bpm)
        truths[i] = {
            "t_start": float(m.time), "t_end": float(c["out_end"]),
            "length_sec": length, "style": m.style, "bpm": bpm,
            "length_beats": length / (60.0 / bpm),
        }

    db = TrackDB(stem + ".db")
    try:
        mine_mix(wav, tl, db, source="synthetic", quiet=quiet)
        rows = db.load_transitions(min_confidence=None, status="ok")
    finally:
        db.close()
    return [(r, truths[r["idx"]]) for r in rows if r["idx"] in truths]


SET_SIZE = 7


def engine_and_ceiling(tracks: list, *, n_pairs: int = DEFAULT_PAIRS,
                       seed: int = 0, quiet: bool = True,
                       set_size: int = SET_SIZE) -> dict:
    """
    Measure the engine's own transitions, and the measurement ceiling, together.

    One pass yields both: rendering a set gives transitions whose lengths we know
    exactly (the ceiling reference) *and* measurements of them (the engine's
    contribution to the comparison). `n_pairs` is a target number of transitions;
    they come from as many `set_size`-track sets as needed.
    """
    rng = random.Random(seed)
    usable = [t for t in tracks if t.bpm and t.cue_points and t.downbeats]
    if len(usable) < 2:
        return {"n_pairs": 0, "matched": [], "measured": [], "rejected": 0}

    size = max(2, min(set_size, len(usable)))
    n_sets = max(1, int(np.ceil(n_pairs / max(1, size - 1))))

    matched, attempted, tag = [], 0, 0
    with tempfile.TemporaryDirectory() as work:
        for _ in range(n_sets):
            sel = rng.sample(usable, size)
            attempted += size - 1
            tag += 1
            try:
                matched.extend(_render_and_probe_set(sel, work, tag, quiet=quiet))
            except Exception:
                continue
    return {"n_pairs": attempted, "matched": matched,
            "measured": [r for r, _ in matched],
            "rejected": max(0, attempted - len(matched))}


def ceiling_hit_rates(engine: dict) -> dict:
    """
    How often the pipeline measures a *known* transition length correctly.

    The upper bound on any hit rate computed from mined data. Pairs each measured
    row with the truth for the same render.
    """
    errs, blend_errs = [], []
    for row, truth in engine.get("matched", []):
        meas_beats = (row["duration_sec"] or 0.0) / (60.0 / truth["bpm"])
        err = abs(meas_beats - truth["length_beats"])
        errs.append(err)
        # Cuts are trivially easy to measure (both truth and measurement are ~0
        # beats) and would flatter the ceiling. The number that matters for
        # calibration is how well *blends* are measured, so report it separately.
        if truth["length_beats"] > 4.0:
            blend_errs.append(err)
    if not errs:
        return {"n": 0}

    def rates(sample):
        a = np.asarray(sample)
        return {"n": len(a), "median_abs_beats": round(float(np.median(a)), 2),
                **{f"within_{tol:g}": round(float(np.mean(a <= tol)), 4)
                   for tol in TOLERANCE_BEATS}}

    out = rates(errs)
    out["blends"] = rates(blend_errs) if blend_errs else {"n": 0}
    return out


def _wasserstein(a: list, b: list) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


def compare_to_corpus(db, tracks: list, *, min_confidence: float = 0.5,
                      n_pairs: int = DEFAULT_PAIRS, seed: int = 0,
                      holdout: float = HOLDOUT_FRAC, quiet: bool = True) -> dict:
    """
    Compare the engine's transition shape against the held-out mined corpus.

    Returns a dict the report formatter renders; `None` sections mean there
    wasn't enough data, which is itself the finding when the corpus is thin.
    """
    rows = db.load_transitions(min_confidence=min_confidence, status="ok")
    train, test = holdout_split(rows, frac=holdout, seed=seed)

    engine = engine_and_ceiling(tracks, n_pairs=n_pairs, seed=seed, quiet=quiet)
    ceiling = ceiling_hit_rates(engine)

    def beats(rs):
        return [r["duration_beats"] for r in rs
                if r.get("duration_beats") is not None and not r.get("is_cut")]

    corpus_beats = beats(test) or beats(rows)
    engine_beats = beats(engine["measured"])

    result = {
        "corpus": {
            "n_total": len(rows), "n_train": len(train), "n_test": len(test),
            "min_confidence": min_confidence,
        },
        "engine": {
            "n_pairs": engine["n_pairs"], "n_measured": len(engine["measured"]),
            "n_rejected": engine["rejected"],
        },
        "ceiling": ceiling,
        "duration_beats": None,
        "cut_rate": None,
        "hit_rates": None,
    }
    if not corpus_beats:
        return result

    result["duration_beats"] = {
        "corpus_median": round(float(np.median(corpus_beats)), 2),
        "corpus_n": len(corpus_beats),
        "engine_median": (round(float(np.median(engine_beats)), 2)
                          if engine_beats else None),
        "engine_n": len(engine_beats),
        "wasserstein": (round(_wasserstein(corpus_beats, engine_beats), 3)
                        if engine_beats else None),
    }

    corpus_cuts = [bool(r["is_cut"]) for r in (test or rows)]
    engine_cuts = [bool(r["is_cut"]) for r in engine["measured"]]
    result["cut_rate"] = {
        "corpus": round(float(np.mean(corpus_cuts)), 4) if corpus_cuts else None,
        "engine": round(float(np.mean(engine_cuts)), 4) if engine_cuts else None,
    }

    # The engine's "choice" is its median transition length; the hit rate asks how
    # often a real DJ's transition landed within tolerance of it. This is the
    # closest computable analogue of the brief's per-transition hit rate, and it
    # must be read against `ceiling`.
    if engine_beats:
        centre = float(np.median(engine_beats))
        errs = np.abs(np.asarray(corpus_beats) - centre)
        result["hit_rates"] = {
            "engine_choice_beats": round(centre, 2),
            "median_abs_beats": round(float(np.median(errs)), 2),
            **{f"within_{tol:g}": round(float(np.mean(errs <= tol)), 4)
               for tol in TOLERANCE_BEATS},
        }
    return result


def format_validation_report(res: dict) -> str:
    c, e = res["corpus"], res["engine"]
    out = [
        "── Validation: engine vs mined corpus ──",
        f"corpus:  {c['n_total']} accepted transitions at confidence >= "
        f"{c['min_confidence']}  (train {c['n_train']} / held out {c['n_test']})",
        f"engine:  {e['n_measured']}/{e['n_pairs']} rendered pairs measured "
        f"({e['n_rejected']} rejected)",
        "",
    ]

    cl = res["ceiling"]
    out.append("MEASUREMENT CEILING — how often the pipeline measures a known")
    out.append("transition length correctly. No engine can score above this.")
    if cl.get("n"):
        out.append(f"  all transitions   n={cl['n']:<3} "
                   f"median |error| = {cl['median_abs_beats']} beats")
        for tol in TOLERANCE_BEATS:
            out.append(f"    within +/-{tol:g} beat(s): {cl[f'within_{tol:g}']:.0%}")
        bl = cl.get("blends") or {}
        if bl.get("n"):
            out.append(f"  blends only       n={bl['n']:<3} "
                       f"median |error| = {bl['median_abs_beats']} beats")
            out.append("    (cuts are trivial to measure and flatter the figure")
            out.append("     above; this is the number calibration depends on)")
            for tol in TOLERANCE_BEATS:
                out.append(f"    within +/-{tol:g} beat(s): "
                           f"{bl[f'within_{tol:g}']:.0%}")
    else:
        out.append("  (not enough rendered pairs to establish it)")
    out.append("")

    d = res["duration_beats"]
    if d:
        out.append("TRANSITION LENGTH (beats, blends only)")
        out.append(f"  corpus median {d['corpus_median']} (n={d['corpus_n']})")
        if d["engine_median"] is not None:
            out.append(f"  engine median {d['engine_median']} (n={d['engine_n']})")
            out.append(f"  Wasserstein-1 distance: {d['wasserstein']} beats")
        else:
            out.append("  engine: no measurable blends among the rendered pairs")
    else:
        out.append("TRANSITION LENGTH: no corpus data")

    cr = res["cut_rate"]
    if cr and cr["corpus"] is not None:
        out.append("")
        out.append("CUT RATE")
        out.append(f"  corpus {cr['corpus']:.0%}" +
                   (f"   engine {cr['engine']:.0%}" if cr["engine"] is not None
                    else ""))

    h = res["hit_rates"]
    if h:
        out.append("")
        out.append("HIT RATE — real DJ transitions falling within tolerance of the")
        out.append(f"engine's chosen length ({h['engine_choice_beats']} beats).")
        out.append("Read against the ceiling above, not on its own.")
        out.append(f"  median |error| = {h['median_abs_beats']} beats")
        for tol in TOLERANCE_BEATS:
            hv = h[f"within_{tol:g}"]
            cv = cl.get(f"within_{tol:g}")
            suffix = f"   (ceiling {cv:.0%})" if cv is not None else ""
            out.append(f"    within +/-{tol:g} beat(s): {hv:.0%}{suffix}")

    out.append("")
    out.append("Note: this comparison is distributional, not per-transition. The")
    out.append("corpus has no source tracks, so the engine cannot be run on the")
    out.append("same pair a DJ mixed, and cue position inside a track is not")
    out.append("recoverable from mix audio at all.")
    return "\n".join(out)
