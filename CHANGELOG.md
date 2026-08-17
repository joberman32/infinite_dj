# Changelog

This file records meaningful behavior and architecture changes, including why
they were made. Read it before changing the mixing or playback pipeline: it
captures constraints that may not be obvious from a local code path.

## 2026-08-17 — Key-sync: pitch-shift the incoming track to fit the outgoing key

Harmonic compatibility was a fixed property of two tracks: `camelot_compatibility`
scored the pair and the sequencer worked around whatever it got. A real DJ has
a pitch fader — they *move* a track to fit. This adds that.

**Why small shifts work, which is not obvious.** Camelot hours step by 7
semitones (the circle of fifths). 7 is coprime to 12, so every semitone offset
lands on a *different* wheel offset — there's no "small shift, small wheel
move" relationship, and a ±1 semitone shift measured against a track's OWN key
lands 5-7 hours away, i.e. worse than doing nothing. But measured against an
arbitrary *partner* key, a small shift routinely closes a large gap, because
the partner may already sit exactly where that shift lands. Exhaustively, over
all 552 ordered Camelot pairs:

| | pairs |
|---|---|
| improve with some shift within ±3 semitones | **432 (78%)** |
| of those, need only ±1 semitone | **240 (56%)** |
| unshifted 0.0 (excluded by `MIN_SCORE`) | 336 |
| …of those, rescued to ≥0.3 | **336 (all)** |

Parallel major/minor (0.9) and reaching an exact key match from a non-matching
start are structurally unreachable: pitch-shifting transposes, it never changes
major↔minor. `pitch_shift_for_compatibility` returns `None` there rather than
pretending. `tests/test_key_shift.py` pins this whole table so a change to
`CAMELOT_MAP` or the scoring tiers can't silently invalidate the premise.

Rendered through `_time_stretch(audio, sr, ratio, pitch_shift_semitones)`,
which does tempo and pitch in **one** Rubber Band pass (`--tempo` + `--pitch` +
`--formant`) — two passes would double the processing and compound each
stage's quality loss. `--formant` keeps a shifted vocal's timbre so it doesn't
chipmunk.

### The bug this shipped with, and the test that caught it

First cut applied the shift only to the crossfade region, mirroring how tempo
already worked. That was wrong, and `test_mine_render_set.py` caught it —
two probe tests that had nothing to do with this feature started failing
because the render they measure had grown a spectral discontinuity.

`render_set` deliberately stretches only the crossfade region and lets the
incoming track resume its native tempo afterward (`cur_audio = nxt_audio`).
For tempo that's an accepted tradeoff: a ~5% nudge at a phrase boundary. Reusing
that shape for pitch meant the incoming track snapped back to its original key
the instant the blend ended. Measured on a synthetic tone pair, isolating the
incoming fundamental either side of the crossfade boundary:

| | incoming fundamental |
|---|---|
| during crossfade (shifted +1) | 423.3 Hz |
| immediately after | 400.0 Hz |
| **jump at the boundary** | **-0.98 semitones** |

A half-step lurch mid-track doesn't read as a modulation, it reads as out of
tune — the feature made the mix *worse* at exactly the moment it was supposed
to make it better. The unit tests all passed, because every one of them checked
a single `_time_stretch` call in isolation; nothing exercised what happened
after the crossfade. The probe test found it by measuring the rendered audio.

**Fixed by keeping the shift for the track's whole time on air**, which is what
a DJ actually does — set the key, leave it. Both renderers now hold a second
whole-track pitch-shifted buffer to resume from (`render_set`'s
`nxt_audio_after`, `PreparedIncoming.resume_audio()`). Same measurement after
the fix: **+0.05 semitones** at the boundary (FFT bin resolution), and the track
stays at 424.7 Hz instead of reverting to 400.

Two consequences worth knowing:

- **A key-synced transition costs two Rubber Band jobs, not one** — a combined
  tempo+pitch pass for the crossfade region, and a pitch-only whole-track pass
  for what plays after. Each buffer is still processed exactly once; the
  crossfade region is never double-processed.

  In the live engine those two passes run **concurrently**, which is not
  cosmetic. Preparation has to finish inside the minimum dwell (32 bars), and
  that budget shrinks as tempo rises: ~62 s at 124 BPM but only ~44 s at 174.
  Measured on a 5.5-min track, the passes are 20.0 s and 11.9 s. Sequentially
  that's 32.7 s — fine here, but it scales with track length, and a 10-min track
  at 174 BPM comes to ~60 s against a 44 s budget. Overrunning doesn't degrade
  gracefully: it falls through to `_handle_track_end`'s synchronous load, which
  blocks the producer thread while the ring buffer drains — the exact mechanism
  behind an audible gap (see 2026-08-14). Both passes are external subprocesses,
  so running them on separate threads genuinely overlaps rather than fighting the
  GIL, turning the cost into the slower of the two instead of the sum: 20.0 s
  measured, and ~37 s for that 10-min worst case, back inside budget.
- **The next transition has to plan against the key actually sounding.** A
  track that was key-synced on the way in is no longer in its analyzed key, so
  `transpose_key()` resolves what it became and that is threaded forward
  (`plan_transition(out_key=…)`, `StreamEngine._current_key_offset`). Offsets
  do **not** accumulate down a set: each shift is measured from that track's own
  native key, not chained off the previous offset, so every track stays within
  `MAX_KEY_SHIFT_SEMITONES` of how it was recorded and no clamp is needed.

The sequencer scores shift-aware too (`CompatibilityEdge.key_shift_semitones`),
docked by `KEY_SHIFT_PENALTY_PER_SEMITONE = 0.03` so a naturally compatible pair
still outranks one that needs a shift to reach the same tier. It derives the
shift from the same pure function on the same two keys as the renderer, so the
two can't disagree without being threaded together explicitly.

Known limitation: the engine's *track selection* still runs off the statically
built compatibility graph, which uses analyzed keys. Selection is a preference
ordering; the shift is the correction applied once a pair is chosen.

## 2026-08-15 — MFCC fallback: cue timbre similarity without installing CLAP

`cue_cosine_similarity` — the function `find_best_cue_pair`,
`choose_transition_style`'s textural-blend detection, and
`library_sim_threshold` all key off — only ever had one signal to read:
`CuePoint.embedding`, the 512D CLAP vector. CLAP is optional and dormant by
default (needs `torch`/`transformers`, ~2GB), so on a fresh install every one
of those consumers silently got `sim = None` and fell back to their
no-similarity-data behavior. Nothing was broken, but a third of the design —
"pick the transition style partly from how texturally similar the two cue
points sound" — never actually ran unless CLAP had been separately installed.

**The fix reuses machinery that already existed and was being thrown away.**
`analyzer.py`'s section-novelty detector already computes a 13-coefficient
MFCC array per track and discards it once the novelty curve is built. Added
`get_mfcc_timbre()` (`infinite_dj/embeddings.py`) computing the same kind of
vector on demand for a cue point: pooled mean+std of MFCC coefficients 1–12
(dropping coefficient 0, which tracks loudness, not spectral shape — cue
points already carry that separately as `.energy`) over the same OUT-looks-
back/IN-looks-forward 8-second window `get_cue_embedding` uses for CLAP. No
model to load, no optional dependency — librosa is already required.

The vector is stored in a **new** `CuePoint.timbre` / `Section.timbre` field,
not folded into `.embedding`. CLAP's 512D vectors and this 24D pooled-MFCC
vector are different spaces; a library re-analyzed after installing CLAP
would otherwise end up with some cue points in one space and some in the
other, and cosine similarity between them is numerically valid but
meaningless. `cue_cosine_similarity` now prefers `.embedding` when both cues
have one, falls back to `.timbre` when neither does, and returns `None`
(same as "no signal") if the two cues only have *different* kinds — it never
compares across spaces.

Because `find_best_cue_pair`, `choose_transition_style`, and
`library_sim_threshold` all go through `cue_cosine_similarity` and nothing
else, none of them needed to change. `library_sim_threshold` in particular
already measures the *real* pairwise-similarity distribution of whatever
vectors are actually populated and takes a percentile of it — so it
re-calibrates itself automatically to the different similarity distribution
MFCC vectors produce, with no separate tuning pass required.

Existing analyzed tracks have neither field populated for anything beyond
what they were analyzed with — `.timbre` only appears on cue points from a
re-`analyze` run after this change.

## 2026-08-15 — Shelving EQ: the bass swap now actually swaps the bass

**The bass swap did not work.** `_split3` splits into three bands and re-sums
them with independent gains. The bands are causal IIR outputs with *different
phase*, so they reconstruct only when their gains are equal — and a DJ EQ
exists to make them unequal. Set the low lane to 0 and the bass does not
leave; it reappears phase-shifted inside the mid band.

Measured on `_blend`, isolating the outgoing path (real track, 16-bar blend,
sub-200 Hz RMS relative to the automation curve the lane asks for):

| phase | low lane says | band-split | shelf |
|---|---|---|---|
| 0.50 | 0.50 | 0.65 | 0.51 |
| 0.60 | 0.08 | **0.57** | 0.10 |
| 0.70 | 0.00 | **0.45** | 0.02 |
| 0.80 | 0.00 | **0.30** | 0.01 |

Worst deviation from the intended curve: **0.48 band-split, 0.02 shelf.** At
phase 0.6 the outgoing kick is seven times louder than asked for. Sweeping a
tone through the "low band killed" state *boosts* 80–200 Hz by up to +5 dB
rather than removing it. So "single source at a time — only one kick drum ever
plays", the claim the mixing design rests on, was not happening: on a 16-bar
blend both kicks ran together for roughly the second half of every crossfade.

This was already half-known. `_split3_zerophase` (added for the collage)
documents the same effect and works around it with `sosfiltfilt` — but
zero-phase filtering needs the whole segment in hand, so the crossfade path,
which must stream, kept the causal version and kept the bug.

**The fix is a different topology, not a better band-split.** A real DJ
mixer's EQ is not a band-split; it is a cascade of minimum-phase shelving and
peaking filters applied to the signal, where the magnitude response *is* the
intended curve. Nothing is re-summed, so there is nothing to mis-cancel.
(Topology from Vande Veire & De Bie; their repo is AGPL, so this is a fresh
implementation from the published RBJ Audio EQ Cookbook formulas.)

Design points worth keeping:

- **`_split_overall` factors level out of shape.** The lanes carry both "how
  loud" and "how shaped". A shelf floors at `EQ_FLOOR_DB` and can never reach
  silence, yet the incoming track's lanes are all 0.0 at phase 0 and must
  contribute *nothing*. Pulling `overall = max(lanes)` out as a scalar gain
  restores both exactness properties the band-split had for free: equal lanes
  give a flat cascade times the level (no colouring), and all-zero lanes give
  true digital silence.
- **`SHELF_STAGES = 2`.** Cascading N shelves at dB/N is steeper than one
  shelf at the full dB. At matched bass kill (60 Hz → 0.065): one −24 dB shelf
  leaves 400 Hz at 0.710, two −12 dB shelves leave it at 0.813.
- **`EQ_FLOOR_DB = −30`**, not −∞ or −40. A killed band lands ~30 dB under the
  incoming track — comfortably masked — for ~2.5 dB of collateral at 400 Hz.
  −40 dB buys inaudible extra kill for real low-mid loss (400 Hz → 0.619).
- **`SHELF_MID_Q = 0.50`**, chosen for evenness rather than isolation. The mid
  lane is the *primary midrange crossfade*, not a surgical cut, so an uneven
  bell turns a level change into a notch at 720 Hz. Lower Q is flatter across
  the band's 3.7 octaves but reaches further into its neighbours; that
  collateral only bites when the mid is killed with both neighbours wide open,
  which current profiles never reach (mid lead ≤ 0.50 while the bass swap
  completes at ≥ 0.62). 0.50 favours the case that always happens without
  going all the way to the 0.30 the bandwidth would theoretically imply.
- **Coefficients ride a global control grid**, recomputed every
  `EQ_CTRL_HOP` (512) samples and *carried in state across chunks*. The first
  implementation sampled the lane at each chunk's start, which made the render
  depend on where the producer happened to cut its chunks; a chunk beginning
  mid-segment must keep the coefficients derived at that segment's true global
  start, exactly like `zi`. Pinned by
  `test_uneven_chunk_boundaries_do_not_shift_the_shelf_control_grid`, which
  caught it.

`EQ_TOPOLOGY` selects `"shelf"` (default) or the legacy `"split"`. An
explicitly supplied filter state always wins, so a transition already under
way keeps its topology mid-flight rather than switching EQ under itself.
Streaming callers build state via `make_crossfade_state()` instead of naming a
class, so the flag moves live playback and offline rendering together.

Cost: 179x realtime vs 290x for the band-split on a 30 s crossfade in 4096-
frame chunks. The producer thread needs >1x.

Note for anyone extending this: `slope`/`q` are read from the module globals at
call time, not bound as default arguments. An earlier version used
`slope: float = SHELF_SLOPE`, which binds at import and silently ignored
`SHELF_SLOPE` changes — a parameter sweep showed slope having exactly zero
effect, which was a measurement artifact, not a result.

### Unexpected consequence: it moved a documented negative result

Two `test_transition_probe.py` tests failed on this change — the ones pinning
CALIBRATION.md §4, "per-band automation is **not recoverable**". They failed
because they got *better*, and that is worth understanding before trusting it.

`build_mix` renders its fixtures through the mixer's real `_blend`, so the
renderer under the measurement is a variable of the experiment, not a
constant. Same estimator, same sweep, only the renderer swapped:

| renderer | true range | measured | compression |
|---|---|---|---|
| band-split | 11.46 s | 4.21 s | 2.72x |
| shelving | 11.46 s | 11.18 s | **1.02x** |

The 4.21 s reproduces CALIBRATION.md's documented 4.2 s, so this is the same
measurement that produced the finding. The estimator was never the limit here:
the renderer had smeared the bass swap it was being asked to locate. The mid-
span bias moved for the same reason (2.25x → 1.27x) — `_split3`'s mid band is
`lp(2600) - lp(200)` on causal filters and therefore carries phase-shifted
residue of *both* neighbours.

Handled by naming the renderer in each test rather than relaxing either
assertion: the split-renderer results are preserved verbatim (they remain true
of that renderer) and the shelf results are pinned separately. **`cp` is
re-opened as a question, not recovered** — the fixture uses an idealised shelf,
while real mixers have unknown corners and real DJs ride faders in ways
`_make_profile` doesn't model. Deciding it needs a sweep against mined mixes,
which has not been run.

Tests: `tests/test_shelf_eq.py`; chunked-vs-continuous is now asserted for
both topologies in `tests/test_engine_scheduling.py`; the probe tests gained a
`topology` fixture and split into per-renderer cases.

## 2026-08-14 — Collage overlap: equal-power layer gain, a real limiter, and an overlap timeline in the player

Listening at high Serendipity (`render_collage`, 3-5 simultaneous layers)
surfaced clipping-sounding audio when many layers overlapped, and the player
had no way to see overlap even though it's the interesting part. Both traced
to the same root cause: `place()`'s `active` list (how many layers are
currently sounding) was private bookkeeping for beat-lock/EQ decisions —
never used to control gain, never exported.

**Clipping.** Overlapping layers were summed with a plain `master[pos:end] +=
seg`, no per-overlap gain compensation (contrast `render_set`'s crossfade
path, which pads with `_apply_gain(mix, 0.9)`). The only safety net was a
single **global peak scalar** applied once after an entire buffer/chunk was
already summed — dense passages rode right up to that ceiling with no
true-peak margin, and in the streaming/radio path the ceiling only ever
dropped, so one dense passage permanently quietened the rest of a session.
Historical "peak ≤ 0.95, no clipping" CHANGELOG notes were always a single
global check, never re-verified after `chaos`/5-layer `insane` shipped, and
there was zero test coverage of gain-summing across layers.

Fix, three parts:

- **`_layer_gain(n)` = 1/√n`, applied in `place()`** at the moment a segment
  is summed in, using `len(active) + 1` (already computed there for the
  beat-lock decision). Equal-power headroom: summed power grows like ln(N)
  instead of N as layers stack. Fixed at the concurrency measured when a
  segment *enters* — it doesn't adapt if layers around it end or join later;
  accepted as an approximation, with the limiter below as backstop. Each
  clip's `layer_gain` is recorded (feeds the new player UI, see below).
- **`_limiter`**: a short-lookahead (15ms), smoothed-release (150ms) peak
  limiter, replacing the single whole-buffer scalar as the actual peak
  control. Envelope computed at a downsampled control rate via
  `scipy.ndimage.maximum_filter1d`, cheap even on a full-length render.
  Confirmed safe to add real DSP cost here: `RadioSession._run` renders on a
  background thread, not the real-time `sounddevice` callback in `engine.py`.
- **Streaming residual ceiling can now recover.** With the limiter doing the
  real peak control, the old whole-block `gain` scalar's job shrinks to
  trimming whatever's left — so it now ducks instantly (unchanged) but
  recovers slowly (`gain = min(1.0, gain + 0.02)` per 20s block, full
  recovery in minutes) instead of staying pinned for the rest of the radio
  session. This overrides the scalar's previous documented rationale
  ("ceiling that only ever drops, so committed blocks join without the
  pumping a per-block normalize would cause") — deliberately: that risk
  applied to a per-block *normalize*, not to a slow exponential recovery
  riding on top of a limiter that's already doing the transient-level work.

New tests in `tests/test_collage_gain.py`: `_layer_gain`'s formula, `_limiter`
respecting its ceiling and recovering (including across streaming calls),
`render_collage` at `layers=5, chaos=1.0` staying under the ceiling with every
clip's `layer_gain` matching the concurrency reconstructed from the clip list
itself, and a regression test that a dense block's ducked streaming ceiling
recovers over subsequent calm blocks rather than staying pinned.

**Overlap visualization.** `build_timeline` already exported per-clip
`start`/`end`/`fade_in`/`fade_out`, and the player's `playerState(t)` already
computed which clips were sounding at a given time — but the UI only ever
rendered a two-line PREV/NOW/NEXT title fade. `timeline.py` now also passes
through `fade_shape`/`eq`/`layer_gain` (already computed by `render_collage`,
previously dropped). The player gained an `#overlap` panel: a scrolling
~30s multi-lane timeline (DOM+CSS, matching the existing meter-bar idiom —
no canvas, no framework, no build step) where each visible clip is a
colour-coded bar (reusing `timeline.py`'s existing per-track HSL colour,
`_color_for`, previously computed but unused by the frontend), fade-tapered
via a CSS mask from `fade_in`/`fade_out`, brightened/dimmed live by the same
`clipGain` math the title-fade already used, plus an "N layers" count. Colour
is scoped to this one panel — the rest of the player stays the deliberately
monochrome design from the earlier `1f5bdd1` redesign. Lanes are assigned
once per clip (greedy interval colouring, cached by clip identity) so a bar
never jumps rows across a radio poll even though `TL.clips` is replaced
wholesale each time; wired into the existing `loop()` timer, not a new one
(the player deliberately avoids `requestAnimationFrame`, which pauses on
hidden tabs).

## 2026-08-14 — Gap detection for the real-time engine

Added checks in `engine.py` so a live `play` session surfaces "one track
fades out and a gap remains before the next starts" instead of it only being
audible in the moment. Three independent detectors, since there are three
different mechanisms that can produce it:

- **True buffer starvation** (`_record_playback`): the ring buffer ran dry
  and the audio callback emitted zeros. Consecutive shortfall across callback
  calls is accumulated into a run; a run crossing `GAP_WARN_SECONDS` (0.25s)
  is warned about live and recorded as a closed `gap_events` entry (with
  duration, track, and next-queued track) once the buffer catches back up.
  A single starved callback doesn't warn — only a sustained run does, so
  ordinary scheduling jitter doesn't spam the log.
- **Near-silent content at a transition boundary** (`_chunk_level_dbfs`
  probe where a pending transition activates): if both the outgoing tail and
  incoming head are below `NEAR_SILENCE_DBFS` (-45dBFS) right where the
  crossfade starts, the whole crossfade will read as a gap even though audio
  is technically flowing. This is legitimate for a deliberate `blend` between
  two sparse sections, so it's logged as a heads-up rather than corrected.
- **The synchronous fallback in `_handle_track_end`**: if no next track was
  prepared in time, this path loads and time-stretches on the producer
  thread itself — the one thread whose job is to keep the ring buffer fed.
  That's the highest-risk path for a real, multi-second gap, so it now times
  the blocking load and records a `gap_events` entry tagged
  `"source": "blocking_load"` when it's slow.

The actual root-cause fix is `_maybe_reprepare_next`, called every scheduler
tick once a next track is selected: a failed background preparation (an
exception in the worker — bad file, Rubber Band error) clears
`_preparing_key` without ever setting `_prepared_incoming`, and nothing
retried it before this change. Left alone, that silently pushed the handoff
onto the blocking fallback above. Retrying here, well before the track ends,
is what actually keeps the engine out of that branch — the three detectors
above are instrumentation for when something still gets through, not a
substitute for this.

`cmd_play` now prints a session-end summary of `gap_events` (count + total
seconds, or a clean "no audible gaps" line) so a run's health doesn't require
scrolling back through the live log.

Tests: `tests/test_engine_scheduling.py` — `GapDetectionTests`,
`ReprepareOnFailureTests`, `BlockingFallbackGapTests`.

## 2026-08-14 — `fade`/`build` tie-break: deliberate draw instead of noise sign

Follow-up to 2026-08-08's finding: `choose_transition_style` picked `fade` vs
`build` by the raw sign of `eo - ei` (exit energy vs. entry energy), but
`_match_entry` chooses the entry cue by *minimizing* that same difference —
so the branch was equalizing two energies and then reading the sign of the
noise left over. Measured on the mined corpus: median margin 0.055, 46% of
beatmatched pairs inside 0.05, 12% inside 0.01. That's not a musical decision,
it's measurement noise wearing a decision's clothes — and with the library at
263 tracks, `build`+`fade` are ~40% of all transitions, so it decides the
majority of what a full set actually sounds like at this boundary.

Fix: a `TIE_MARGIN = 0.08` band around the tie. Outside it, the sign still
decides, unchanged (`test_decisive_margins_are_unaffected`). Inside it,
`choose_transition_style` draws from `_seeded_unit` — a SHA-256-based
deterministic float in [0, 1), not `hash()` (salted per-process, so not
reproducible run-to-run) — weighted linearly by the margin so an exact tie is
50/50 and the draw converges to the sign comparison as it nears `TIE_MARGIN`.

This was framed in the 2026-08-08 entry as "an argument for the seeded-variance
design rather than against it: replacing a coin flip on noise with a
deliberate, reproducible choice is a strict improvement" — the design here is
that deliberate choice. Two things it guarantees that the old code didn't:

- **Reproducibility.** Same cues in, same style out, every run — `plan_transition`
  stays a pure, replayable function (required by `gaps`/`triage`/the corpus
  validator), which a literal RNG draw would have broken.
- **No systematic bias.** A true tie is close to 50/50 over many draws
  (`test_exact_tie_is_roughly_balanced_over_many_draws`), rather than
  whatever bias floating-point ordering of `eo`/`ei` happened to introduce.

`choose_transition_style` gained an optional `seed_extra` tuple so a caller can
salt the draw; `plan_transition` gained `occurrence` (times this ordered
track pair has already transitioned in the current render) and passes
`(track_out.file_path, track_in.file_path, occurrence)` as that salt.
`render_set` tracks per-pair occurrence counts and threads them through — so a
repeated pair (radio mode, over a long run) isn't guaranteed to draw the same
coin twice. Callers that don't track repeats (`library_health.plan_transition`
calls, one-off `gaps` measurement) default `occurrence=0` and still get a
reproducible, pair-specific draw from the cues alone.

Not touched: the `blend`/`swap` thresholds (0.45/0.70) and the `sim >=
high_sim_threshold` blend gate. Both are hard cutoffs on values calibration.py
already documents as not mineable from mix audio (different energy
normalization between mined-mix and per-track measurement); a margin-based
tie-break there would need its own justification, not an extension of this one.

## 2026-08-09 — the library is no longer the bottleneck (first bulk `fetch`)

Ran the `gaps → fetch → analyze → triage → gaps` loop for real. `fetch
--from-gaps` targeted 135 BPM ±8 (the densest cluster), screened 5,527 tracks
across 899 netlabel releases via Essentia sidecars, and kept 243 — of which 239
downloaded (4 lost to transient Archive 5xx) and 238 survived a dest-path
collision. 3.3 GB, ~66 MB of sidecar traffic to screen it.

What that did to the engine's own report:

| | 25 tracks (2.28h) | 263 tracks (23.78h) |
|---|---|---|
| beatmatchable pairs | 36% | **64%** |
| Camelot coverage | 13/24 | **24/24** |
| `cut` | 64.5% | **36.0%** |
| `swap` | 6.3% | **20.1%** |
| `build` | 13.3% | **25.7%** |
| `fade` | 10.2% | **14.6%** |
| `blend` | 5.7% | **3.5%** |

Triage: 253 good, 4 usable, 6 reject.

**This retires "the library is too small and non-diverse" as an explanation for
monotonous transitions.** Hard cuts nearly halved and three of four crossfade
styles roughly doubled purely from composition — no engine change. What remains
is an engine property: `blend` *fell* to 3.5%, and `build`+`fade` together are
40% of transitions while being separated only by `eo >= ei`, a comparison
`_match_entry` actively drives toward zero (median margin 0.055 on the old
library). Variety work now has to come from the mixer, not the music.

Screening rejections, for anyone tuning the filters: 2,361 tracks had no
Essentia sidecar at all (~43% of the collection — the single biggest loss),
799 duration, 720 no declared license, 697 wrong subgenre, 678 off-tempo.

Two small known defects, neither fixed here: `Candidate.dest_path` does not
disambiguate collisions, so two sources mapping to one filename silently lose
one (1 of 243); and licensing is 46% `-nd`, fine for listening but not for
publishing a mix built from it — use `--license by,by-sa,by-nc-sa` if that
matters. `.gitignore` now covers `music/` and `*.db.bak-*`, since the audio
library and its `PROVENANCE.jsonl` were untracked-but-not-ignored.

## 2026-08-08 — `gaps` now plans instead of re-deriving (corrects a wrong finding)

`library_gaps` predicted transition styles by calling `choose_transition_style`
on `best_cue_out` / `best_cue_in`. The set renderer never uses those. It goes
through `plan_transition`, which picks the exit with `_pick_exit_cue` (a
`groove_floor=0.4` that rejects dead-valley exits) and the entry with
`_match_entry` (which searches the incoming track's *downbeats* for energy
matching the exit, rather than using the low-energy scored IN cues).

Measured over all 600 ordered pairs of the 25-track library:

| | via `best_cue_out`/`best_cue_in` | via `plan_transition` |
|---|---|---|
| median exit energy | 0.15 | **0.51** |
| exits above the 0.70 swap gate | 0% | **16%** |
| `swap` | 0.0% — "unreachable" | **6.3%** |
| `fade` | 2.2% | **10.2%** |
| `blend` | 18.0% | 5.7% |
| `cut` | 64.5% | 64.5% |

**The "`swap` is unreachable" finding was an artifact of the report, not a
property of the engine.** All four crossfade styles fire. The reasoning behind
it was sound in isolation — `cue_detector` does reward energy valleys in OUT
scoring — but `_pick_exit_cue` already compensates, which is invisible if you
read the cue list instead of running the planner. Anything predicting engine
behaviour by re-deriving it will drift from the engine; `plan_transition` was
extracted precisely so callers don't have to. A test now pins that
`library_gaps` calls it and mentions neither `choose_transition_style` nor
`_strongest`.

Two consequences:

- `low_energy_tracks` / `high_energy_tracks` now count tracks by the exit energy
  the planner actually selects, not by whether any cue anywhere clears a gate.
  On the reference library that moved 25 low / 21 high to 7 low / 4 high.
- The headline finding is now the **cut share** rather than `beatmatchable_frac`.
  They are the same number — a cut fires exactly when tempos clash — but the old
  `bm_frac < 0.30` gate stayed silent at 36% beatmatchable, i.e. while 64.5% of
  transitions were hard cuts. The report was failing to mention the single most
  audible fact about the library.

Separately measured and worth recording, since it bears on the transition
variance work: `fade` vs `build` is decided by `eo >= ei`, and covers 141 of the
213 beatmatched pairs. But `_match_entry` picks the entry cue by *minimizing*
`|eo - ei|`. Median margin at that branch is **0.055**, with 46% of pairs inside
0.05 and 12% inside 0.01. The engine equalizes the two energies and then
branches on the sign of the difference it just equalized, so for two thirds of
all crossfades the style choice carries almost no musical signal. That is an
argument *for* the seeded-variance design rather than against it: replacing a
coin flip on noise with a deliberate, reproducible choice is a strict
improvement.

## 2026-08-08 — `fetch`: screened library expansion from the Internet Archive

New `fetch_archive.py` and `dj.py fetch`. Downloads Creative Commons electronic
music from the Archive's `netlabels` collection, screened *before* download
against what the library is missing.

**Why screening before download matters.** `gaps` established that engine
behaviour is gated on library *composition*, not size: at 25 tracks spread over
108–161 BPM, only 36% of pairs beatmatch and 64.5% of transitions come out as
0.3s hard cuts. Bulk-fetching 24 hours of arbitrary electronic music would not
fix that — 24 hours of equally-scattered tempos still cuts. What raises the
beatmatchable fraction is tracks clustered near a tempo the library already has.

**How it's possible.** Around 2014-15 the Archive ran Essentia over most of its
audio and left the results beside each file: `<stem>_esshigh.json.gz` (genre and
danceability classifiers, ~2 KB) and `<stem>_esslow.json.gz` (BPM, key,
duration, ~24 KB). Screening a track therefore costs ~26 KB instead of the ~6 MB
of downloading it and finding out. Checks run cheapest-first — license and
listing duration are free, then the 2 KB genre file, then the 24 KB rhythm file
only when a `--bpm` target makes it relevant — so most rejects cost 2 KB or
nothing. `tests/test_fetch_archive.py` pins that ordering, because losing it
turns a minutes-long screen into an hours-long one.

`--from-gaps` closes the loop: it runs `library_gaps`, aims at the densest BPM
cluster the library already has, and fetches only the shortfall to the target.
That is the whole reason the two tools were built in this order.

Notes and limits:

- The Archive's BPM is a *screening hint only*. Nothing from Essentia enters the
  track database; `analyze` re-derives everything from audio, as before.
- Roughly half the netlabels corpus has no Essentia sidecars. Those tracks are
  skipped by default (`--allow-unscreened` keeps them, unfiltered).
- Every download appends to `PROVENANCE.jsonl` (source URL, creator, license).
  CC-BY and friends require attribution to survive the copy. Items with no
  declared license are skipped.
- Much of the collection is `by-nc-nd`. Fine for private listening; a published
  mix built from it is a derivative work. The plan output flags this, and
  `--license` filters on the short code.
- Tempo comparison folds to the engine's [90, 180) window, so a 63 BPM track
  matches a 126 BPM target — the same octave the engine will see.

## 2026-08-08 — Library health: `triage` and `gaps`

Two read-only reports over cached `TrackMeta` (no audio decoded), added while
planning a library expansion from 25 tracks to ~24 hours. New
`library_health.py`, `dj.py triage`, `dj.py gaps`.

- `triage` grades each track `good`/`usable`/`reject` on whether the engine can
  mix it: tempo confidence, downbeat count, beat-grid coverage of the audio,
  cue availability, duration, octave-fold edge cases. Hard failures reject;
  weak-but-workable ones stay. **Documented blind spot:** the stored grid is
  equidistant *by construction* (`_refine_tempo_phase` lays it down with
  `arange`), so a confidently-wrong tempo grades `good`. Only ears catch that,
  and `test_a_confidently_wrong_grid_is_invisible` pins the limitation so it
  isn't mistaken for a bug.
- `gaps` reports the library against the engine's *real* gates rather than
  against size. It calls `choose_transition_style` on sampled pairs instead of
  reimplementing its logic, so style reachability can't drift from the mixer.
  The two energy thresholds it does duplicate (0.45 / 0.70) are pinned by
  `test_gap_thresholds_mirror_the_mixer_gates`, which greps the mixer source.
- **First run surfaced two findings that reframe the transition-monotony work:**
  on the 25-track library, **64.5% of sampled pairs produce `cut`** (tempo
  scatter across 108–161 BPM, only 36% beatmatchable), and **`swap` is
  unreachable at 0%** while `fade` sits at 2.2%. So the engine is mostly making
  hard cuts, and of the real crossfades only `blend` and `build` fire in
  practice. `swap` requires both cue energies > 0.70, but `cue_detector`'s OUT
  scoring explicitly rewards energy *valleys* (`energy_valley×2.5`,
  `low_absolute_energy×0.5`), so the strongest OUT cue is systematically
  low-energy — the two components work against each other by construction.
- Reports are advisory only. Nothing filters the library or changes rendering.

## 2026-07-29 — Corpus mining and calibration (Phase 1)

Full results and the Phase 2 recommendation are in [CALIBRATION.md](CALIBRATION.md).
Read that before extending any of this; the headline is that the bottleneck is
measurement accuracy, not the engine.

- `plan_transition` extracted from `render_set`'s loop as a pure function, so
  cue and style selection can be replayed without rendering audio. Verified
  byte-identical on a 6-track set covering both the beatmatch and cut paths.
  Two constraints keep it faithful: `dur_out` must be the *loaded* audio length
  (not `TrackMeta.duration`), and the sample-domain clamps stay in `render_set`
  because one of them depends on the style that planning returns.
- New `mix_grid.py`: piecewise tempo tracking for whole mixes.
  `analyzer._compute_beats` assumes one tempo per file, which is right for a
  track and wrong for a 60-minute set. Reuses `_refine_tempo_phase` unmodified.
- New `transition_probe.py`: measures a transition from mix audio alone via a
  two-source non-negative decomposition per band. The obvious approach — a
  broadband similarity ramp — recovers 3.7 bars for a known 16-bar crossfade,
  because the low band dominates spectral energy and is a *switch, not a ramp*.
  Band separation is mandatory, not an optimisation.
- New `mix_corpus.py` + `mixes`/`transitions` tables (`DB_VERSION` 3). Rejected
  boundaries are persisted with all sub-scores so thresholds can be re-tuned by
  re-querying rather than re-running audio analysis — which paid for itself
  immediately, as the first real run rejected 100% of boundaries on an absolute
  residual threshold that only suited synthetic tones.
- New `calibration.py`, wired into `choose_transition_style` (crossfade bars),
  `render_set`/`plan_transition` (`min_solo_bars`), the engine's dwell bounds and
  the sequencer's minimum edge score. Defaults are exactly the previous
  hardcoded values, so an absent `calibration.json` is a no-op; a permanent test
  pins byte-identical rendering. Values below 20 observations are refused.
- New `validation.py` + `dj.py validate`. Both sides of any comparison are
  rendered and probed through the same pipeline — reading the engine's automation
  lanes analytically would attribute the probe's biases to the DJ. The report
  always shows the measurement ceiling beside the hit rate.
- New CLI: `mine`, `probe`, `corpus`, `calibrate`, `validate`. Nothing fetches
  anything; the corpus is whatever is placed in a folder.
- Two negative results worth not rediscovering: per-band automation phase
  (`_make_profile`'s `cp` and leads) is ~2.7x compressed and **not** calibratable
  despite being the highest-value target on paper; and blend *duration* carries
  ~25 beats of median error on 32-48 beat transitions, so only its central
  tendency over many transitions is usable. Cut-vs-blend classification and
  transition placement are the parts that work well.
- Doc fixes: `MIN_DWELL_BARS`/`MAX_DWELL_BARS` were documented as 16/64 but are
  32/96; `render_set` returns a 4-tuple, not 3; the Camelot cross-mode 0.5 rule
  was undocumented; `render_set`'s `n_mix_bars` parameter is dead, since
  `style.n_bars` overwrites it.

## 2026-07-26 — Repair endless radio and live crossfade progress

- Radio lookahead is now measured from the browser's audible playback cursor,
  not from time zero. The player sends a monotonic playback heartbeat with each
  state poll; `RadioSession` resumes rendering whenever
  `generated_sec - played_sec` falls below the lookahead. Previously it reached
  four minutes of total output and then idled forever.
- Heartbeats are clamped to generated audio so malformed future positions cannot
  trigger unbounded catch-up rendering. Radio state exposes played and buffered
  seconds for diagnosis.
- The live engine now stores the final scalar from each per-sample crossfade
  phase ramp in `PlaybackState.mix_progress`. It previously stored the full
  NumPy vector, which made the terminal UI fail when converting progress to a
  percentage.

## 2026-07-24 — `chaos`: make INSANE genuinely wild

- `render_collage(chaos=0..1)` is a wildness master. As it rises: weave crowds
  out the calmer movements, segments shorten, hops shrink (more layers, faster
  switching), sections get diced, and picks alternate contrast/complement.
  INSANE sets `chaos=1.0`, `layers=5`, `max_seg_bars=12`; HIGH sets `0.35`.
  Exposed as `splice --chaos` too. `chaos=0` is byte-identical to before.
- **Sub-segments** (`_segment_pool`): entries previously landed only at whole
  structural section *starts* (~5–7/track, ≥8 s), which capped how short a
  splice could be. Sections are now diced every N bars (≤6 per section so the
  pool stays tractable); sub-segments inherit the parent's CLAP embedding and
  label, since near-constant timbre is what defines a section.
- **Complementary blending**: `pick_contrast` was farthest-only and could never
  choose textures that sit *together*. Now `pick_next(complement=…)` can rank by
  CLAP-nearest, and weave alternates between the two as chaos rises. Large diced
  pools are scored on a random 300-candidate slice to keep picks cheap.
- **Beat-grid safety (two bugs caught in verification).** Shortening segments
  initially made layering *worse* (5 → 2): `place()` floored hops at one whole
  bar, so a 4 s splice could only stack twice. Fractional hops fixed that, but
  26/33 possible hops then landed at fractional beats (1.33, 1.60 …) — layers
  off the beat grid, the rhythmic mush this project already fixed once. Hops are
  now snapped to the shared beat grid, and *only* when sub-bar hops are enabled:
  a bar is not always an exact 4 sample-beats (e.g. 147.8 BPM), so quantising
  whole-bar hops would drift. Verified across 128/111.2/147.8 BPM: zero off-beat
  hops at chaos=1, zero change at chaos=0.
- Measured (6-track pool, 90 s): chaos 0 → 1 takes avg segment 18.8 s → 4.6 s
  (min 3.6 s, below the old 8 s section floor — sub-segmentation working), max
  simultaneous layers 4 → 8, entries/min 7.9 → 39.6, peak 0.95 (no clipping).

## 2026-07-24 — Radio mode: endless, continuously-rendered mixes

- `RADIO` in the studio starts an endless mix instead of a fixed-length render.
  It primes a few seconds so playback starts almost immediately, then a
  background thread keeps extending the collage until you hit EXIT.
- **What made it possible** was the stretch-on-demand work (see above): renders
  now run 20–45x faster than playback, so the buffer races ahead of the
  listener. Measured over HTTP: prime 0.37s → 30s of audio ready (gate ≤3s),
  140s buffered within 6s of wall time.
- `render_collage(state=...)` is resumable; `RadioSession` (`radio.py`) drives
  it, emitting finalised audio as 10s WAV chunks and idling once ~4 minutes
  ahead. Endpoints: `POST /api/radio`, `GET /api/radio/state`,
  `GET /api/radio/chunk`, `POST /api/radio/stop`.
- A growing stream can't go through `<audio>`/MSE, so the player schedules
  decoded chunks on the AudioContext clock, each starting exactly where the last
  ends. Both playback paths now share one `bus` → analysers, so the stereo
  meters work unchanged. The playhead is `actx.currentTime`, which means
  `suspend()` pauses the clock for free.
- Radio has no duration or seek: the scrubber gives way to a LIVE badge and
  EXIT (stops the session and deletes its chunks). Verified in-browser: 26s
  spanning 3 chunk seams with **zero** silent samples, meters live, PREV/NOW/NEXT
  updating, no console errors, clean EXIT with no files left behind.
- **Only stretch on a *material* overlap.** Locking on any overlap at all meant
  a 1-2 bar join at the end of a 70-bar segment forced the whole segment to be
  stretched — 67% of LOW segments, the exact waste the change was meant to
  remove. Locking only above a 25% overlap took LOW to **0% stretched and 65.6x
  realtime** (from 67% / 13.6x); HIGH 71% -> 33% and 11.6x -> 20.1x. LOW radio
  now plays entirely at native tempo: faster *and* higher fidelity.
- `mixspec.radio_profile(level)` shapes character per level rather than
  switching renderers (`render_set` has no resumable form), so LOW radio
  *approximates* — does not replicate — offline LOW. INSANE is bracketed:
  its bottleneck is CLAP candidate scoring, not rendering.

## 2026-07-24 — Studio: setup pane + render-on-demand

- New `dj.py studio` launches a browser app where you pick a track pool and
  dial in the mix, then generate it — no CLI flags required.
- **Two axes, not three.** The brainstormed "segmentation high/low" and "raw
  min/max play time" are the same axis at different abstraction levels, so they
  ship as one **Pace** control (preset + optional raw override).
  **Serendipity** (low/medium/high/insane) is the master: it selects the
  renderer and, at `insane`, deliberately supersedes Pace.
- `infinite_dj/mixspec.py`: `library_groups` (artist→album→track, from the
  folder `"Artist - Album"` or falling back to the filename for slug folders)
  and `resolve_params`/`render_from_spec` — the single mapping from UI spec to
  `render_set`/`render_collage` arguments, unit-tested without touching audio.
- `webserver.py` is now an app: `/api/library`, `POST /api/render` (background
  thread + job registry), `/api/render?job=`, and job-scoped `/audio?job=` /
  `/timeline.json?job=`. `serve --audio` still works — it pre-seeds a "default"
  job, so the player's job-less requests resolve unchanged.
- `webplayer/setup.{html,css,js}` in the same B&W monospace language;
  `index.html` → `player.html`; `player.js` reads `?job=`.
- **Known behavior:** at LOW the length is a target, not a cap — full-set mode
  plays to genuine structural exit cues, so a 3-minute request rounds up to the
  nearest whole musical spans (~8 min on this library, where cues sit ~2 min
  apart). Fixed along the way: LOW previously ignored length entirely and
  rendered the whole library (4287 s for a 180 s request).
- Deferred: variable/nonlinear crossfade DSP (phase 2) and the live
  engine + "Next"/look-ahead over a websocket (phase 3).

## 2026-07-24 — dBFS stereo meter

- Changed the player meter from amplified linear RMS to a −60–0 dBFS scale.
  Fixed mixer-style colour zones now mark green below −12 dBFS, yellow from
  −12 to −6 dBFS, and orange above −6 dBFS. The fill scales over the fixed
  gradient so the colours represent stable thresholds rather than proportions
  of the current reading.
- Prev/now/next identify each splice with a compact `Track · 2P` label: the
  ordinal is that track's segment number and the letter is the section class
  (`P`eak, `R`ising, etc.). Archive prefixes, file extensions, underscores,
  and leading track numbers are removed from displayed titles. Adjacent
  segments from the same source track remain visible in the previous slot.
- Replaced the numeric `MIXING %` bar with a minimalist title crossfade. During
  an overlap, outgoing and incoming splice labels fade and drift through one
  shared title position. Animation progress is normalized to the clips' actual
  simultaneous overlap—not merely the incoming clip's longer fade-in envelope—
  so the visual handoff completes when the outgoing audio ends.

## 2026-07-22 — Interactive web player (MVP, phased)

- Productization MVP: a dependency-free local web player that plays a rendered
  set/collage and visualizes it live, synced to `audio.currentTime`.
- `render_set`/`render_collage` now also return a **clips** list (per-track
  segment on the output timeline: out_start/out_end, fade in/out, mode, section);
  return arity changed to `(audio, sr, markers, clips)` — the two `dj.py` callers
  updated. New `infinite_dj/timeline.py` (`build_timeline`/`write_timeline`)
  joins clips with track metadata (title/bpm/camelot key/energy/colour) into a
  compact JSON; embeddings never leak in.
- `infinite_dj/webplayer/` (index.html, player.css, player.js): a vanilla-JS
  dashboard — now-playing card (title/bpm/key/section/mode, per-track colour),
  Camelot key wheel (SVG, highlights current + incoming), energy meter,
  crossfade-progress ring, up-next countdown, a mini arrangement timeline with
  playhead, and transport/seek. Theme-aware. It computes a `PlayerState(t)` from
  the clips — the same shape the real-time engine can emit later (the phasing
  hinge).
- `infinite_dj/webserver.py`: tiny stdlib `http.server` with HTTP Range (audio
  seeking). `dj.py serve` + `--serve`/`--timeline`/`--port` on the renderers.
- Verified in-browser: loads, plays, syncs now-playing/Camelot/energy/up-next,
  track flips at overlaps, no console errors. Fix: primary-clip selection
  includes fade-in edges (t=0 no longer stuck on "Loading"). Tests: 23 pass
  (+2 timeline).

## 2026-07-22 — Structured variable-pace collage (render_collage)

- Replaced the fixed-cadence `render_layered` (a new track every ~9 s) with
  `render_collage`, a structured scheduler whose editing pace ebbs and flows.
  It composes *movements*: **feature** (one track holds, playing several of its
  own sections contiguously in natural time order), **weave** (rapid, heavily
  overlapping segments chosen to be timbrally CONTRASTING — CLAP-farthest — from
  what's already sounding, within or across tracks), and **breathe** (a long
  segment mostly alone). Movement weights loosely arc over the set.
- Segment lengths vary (`--min-seg-bars`/`--max-seg-bars`); `--seed` makes pacing
  reproducible; `--layers` is the weave overlap ceiling. Tracks are stretched to
  the set tempo once and cached (many faster repeated splices).
- Reuses the beat-locked overlap-add core, `Section.embedding`, the farthest-
  first primitive, and `camelot_compatibility` (harmonic tie-break among the
  most-contrasting weave candidates). Markers tag the movement (feature/weave/
  breathe). Verified: inter-entry gap std 0→15 s, feature holds are same-track
  time-ordered, no silence gaps, peak ≤ 0.95.

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
