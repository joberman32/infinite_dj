# Changelog

This file records meaningful behavior and architecture changes, including why
they were made. Read it before changing the mixing or playback pipeline: it
captures constraints that may not be obvious from a local code path.

## 2026-07-22 — Layered collage + uncapped splice crossfades

- Removed the splice-mode crossfade cap: crossfades run their full style length
  (up to n_mix_bars) even on short segments — a segment that's mostly crossfade
  is desirable, not a bug.
- New `render_layered` (`splice --layers N`): overlap-add collage where up to N
  tracks sound at once. All layers are stretched to one tempo (pool median,
  octave-folded) and entered on a shared bar grid spaced `layer_bars/N` bars
  apart, so beats stay aligned during the N-way overlap. Equal-power fades;
  peak-normalized (3-way sums verified non-clipping).
- `sequence_for_mixing(stochastic=True)` + larger cooldown for collage mode:
  samples from the top-5 candidates weighted by score instead of the argmax, so
  the collage wanders the whole library (24/25 tracks used, was 12/25) instead
  of looping the most-compatible cluster.

## 2026-07-22 — Splice mode (short-segment collage sets)

- New `splice` command / `render_set` mode: build a target-length collage from
  short segments of many tracks instead of full tracks. Params `--length` (min),
  `--min-seg`/`--max-seg` (seconds). Each track plays only a bounded segment and
  exits at a CLAP-serendipitous cut point (`_pick_splice_exit` picks the OUT cue
  whose embedding best matches the next track's entry); crossfades are capped to
  ~1/3 of the segment so they fit; rendering stops at the target length.
- `sequence_for_mixing(allow_repeats=True, cooldown=N)` lets a long collage
  revisit a small pool (n_tracks may exceed library size) with a recency window.
- `render_set(min_seg_sec, max_seg_sec, target_length_sec)` drive the mode;
  default (full-set) behavior unchanged. Verified: 10-min and 5-min collages,
  zero interior silence gaps.

## 2026-07-22 — CLAP validated and wired into set ordering

- First real end-to-end run of CLAP (torch/transformers installed from the
  optional extras) fixed three transformers-5.x breakages in `embeddings.py`
  (`audios=`→`audio=`, output-object unwrap, tensor-truthiness). The feature
  had never actually executed before this.
- Found CLAP had ~no effect on rendered sets: `sequence_for_mixing` scored on
  `e.harmonic`, not the CLAP-weighted `e.score`, so ordering ignored it; and
  the fixed 0.82 style threshold sat at ~65th percentile of a real library.
- Fix: CLAP now feeds set ordering via a **per-library percentile ranker**
  (`_percentile_ranker`, weight 0.75 in `sequence_for_mixing.score`) so the
  compressed 0.36–0.93 similarity band becomes a discriminating signal; the
  blend style threshold is now the library's 85th percentile
  (`library_sim_threshold`), passed through `render_set` →
  `choose_transition_style(high_sim_threshold=...)`.
- A/B (25-track Aphex+CC): mean consecutive-pair CLAP sim 0.749 → 0.785 (more
  timbral continuity), −0.02 mean harmonic, +1 tempo cut. Still fully optional
  and inert without embeddings.

## 2026-07-22 — 3-band breakpoint EQ crossfades

- Replaced the mixer's fixed cos/sin bass-swap + high-crossfade with a 3-band
  DJ-mixer EQ (low/mid/high) driven by per-track breakpoint automation lanes
  (`TransitionProfile`, `_make_profile`, `_split3`). Bands are built by
  difference-of-lowpass for exact reconstruction; mid and high crossfade with
  independent timing so, e.g., a drop→drop `swap` brings the incoming hats in
  early while holding its mids back. Idea from Vande Veire & De Bie's auto-DJ
  (AGPL) — reimplemented, not copied.
- `CrossfadeFilterState` now carries two lowpass states (200 Hz, 2600 Hz) per
  source. **Invariant preserved**: stateful chunked rendering must equal a
  single continuous render sample-for-sample (test
  `test_stateful_chunked_blend_matches_continuous_rendering`). If you touch the
  band split, keep low+mid+high == input and keep the offline `_split3` and the
  stateful `.split()` identical.
- Legacy `TransitionStyle` scalar knobs are retained; `_default_profile` builds
  a profile from them when `style.profile` is None (back-compat for the tests
  and any direct `TransitionStyle(...)` construction).

## 2026-07-21 — Multi-Core Parallel Library Analysis

- Upgraded `dj.py analyze` command to use Python's `concurrent.futures.ProcessPoolExecutor` for multi-core parallel processing (`dj.py`, `analyzer.py`).
- Added `--workers N` CLI flag (defaulting to CPU core count up to 8 workers), speeding up batch library analysis by 4–8x.
- Added `verbose: bool = True` to `analyze_track()` to suppress worker sub-task output while maintaining real-time batch progress logging in the main process (`analyzer.py`).
- Preserved single-threaded SQLite write operations in the main process (`db.save(meta)`) for thread and process safety (`dj.py`).
- Added test coverage in `tests/test_parallel_analysis.py`.

## 2026-07-21 — CLAP Neural Audio Embedding Integration


- Integrated HuggingFace **CLAP** (`laion/clap-htsat-fused`) embeddings for timbral and structural cue-point pairing.
- Extracted 512-dimensional L2-normalized feature vectors for 8-second audio windows surrounding every top-K `IN` and `OUT` cue point (`embeddings.py`, `cue_detector.py`, `analyzer.py`).
- Added optional `embedding` field to `CuePoint` dataclass and serialized it into SQLite JSON columns for backward-compatible database persistence (`models.py`, `db.py`).
- Implemented `cue_cosine_similarity` and `find_best_cue_pair` in `sequencer.py` to pair `OUT` and `IN` cue points based on acoustic vector similarity, phrase alignment, and cue confidence.
- Updated `build_compatibility_graph` in `sequencer.py` to factor CLAP cue similarity into set sequencing decisions (40% harmonic, 30% rhythm, 30% CLAP cue similarity).
- Updated `choose_transition_style` in `mixer.py` to select smooth 16-bar `blend` transitions for high CLAP vector similarity ($\ge 0.82$).
- Added CLI reporting for CLAP embedding status in `inspect`, `cues`, and `mix` subcommands (`dj.py`).
- Added comprehensive unit test suite in `tests/test_embeddings.py` covering serialization, vector math, pairing logic, and DB roundtrips.
- **2026-07-22 follow-up**: torch/transformers moved out of `requirements.txt`
  into optional `requirements-clap.txt` (~2 GB; the base pipeline must stay
  lightweight). Status: not yet validated with the real model — no library DB
  contains embeddings, and the 0.82 blend threshold is untuned. Everything
  falls back to energy/harmony matching when embeddings are absent.

## 2026-07-21 — Real-time transition reliability


Commit: `3df2794 Harden real-time transition playback`

- Scheduler-selected OUT cues now determine the exact start of a normal
  transition. The previous producer behavior replaced the selected cue with
  whichever downbeat came next, which could start the mix up to eight bars
  early. Forced skips and cue-less fallback transitions retain their separate,
  safe behavior.

- Live crossfades use a per-sample phase ramp rather than a single gain value
  for each 4096-frame producer chunk. This prevents audible gain/EQ stepping.
  EQ filter state is also preserved for the duration of a transition so every
  chunk does not restart its filters and introduce a transient.

- Incoming tracks are decoded, loudness-matched, time-stretched, and
  downbeat-aligned in a background preparation thread as soon as the scheduler
  selects them. The producer must keep the output buffer full, so it must not
  perform full-track I/O or Rubber Band processing during a handoff. If
  preparation is late, playback continues and the handoff waits for a safe
  future downbeat.

- The deque-and-lock output buffer was replaced with a preallocated,
  single-producer/single-consumer stereo ring buffer. The audio callback no
  longer takes a mutex or allocates its output buffer. It fills underflows with
  silence and records underruns; producer-side waiting occurs only when the
  ring is full.

- Render time and audible time are distinct. The producer remains ahead so it
  can write a future transition into the buffer, while playback/session time
  advances only when frames are consumed. Scheduler dwell and cue policy use a
  latency-compensated audible track position.

- Added regression coverage for cue timing, asynchronous preparation, seamless
  chunked DSP, ring-buffer ordering/underflow behavior, and latency
  compensation. The suite passed with the repository `.venv`.
