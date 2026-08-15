# Calibrating the transition engine against real DJ mixes — Phase 1 results

Phase 1 asked: can we replace the engine's hand-tuned transition constants with
distributions mined from real DJ mixes, and is the result good enough that a
learned model (Phase 2) isn't worth attempting?

**Short answer: the pipeline is built and reproducible, but the measurement is
not accurate enough to calibrate transition length — the primary target. Neither
the empirical-priors approach nor a learned model is blocked by the *engine*;
both are blocked by the same thing, which is that mix-audio-only measurement
can't recover blend duration to better than ~25 beats. Phase 2 as originally
scoped would inherit that ceiling and should not be started.** The full
reasoning is at the end.

---

## 1. What was built

| Module | Purpose |
|---|---|
| [`mix_grid.py`](infinite_dj/mix_grid.py) | Piecewise tempo tracking over a whole mix. Reuses `analyzer._refine_tempo_phase` unmodified. |
| [`transition_probe.py`](infinite_dj/transition_probe.py) | The measurement DSP. Two-source non-negative decomposition per band, logistic ramp fit, seven-stage reject stack. I/O-free. |
| [`mix_corpus.py`](infinite_dj/mix_corpus.py) | Tracklist parsing (`.txt`/`.cue`/`.json`), corpus scan, per-boundary measurement, stats. |
| [`calibration.py`](infinite_dj/calibration.py) | Mined values with provenance and a min-observations refusal; falls back to today's constants. |
| [`validation.py`](infinite_dj/validation.py) | Engine-vs-corpus comparison, and the measurement ceiling. |

CLI: `dj.py mine`, `probe`, `corpus`, `calibrate`, `validate`.

```bash
python dj.py mine ~/Music/dj-mixes      # audio + .txt/.cue/.json sidecar per mix
python dj.py corpus                     # distributions + rejection report
python dj.py calibrate --out calibration.json
python dj.py validate
python dj.py probe somemix --idx 3      # full detail for one boundary
```

Nothing fetches anything. No scraper was written; the corpus is whatever is put
in the folder. `DB_VERSION` 2→3 adds `mixes` and `transitions` tables.

## 2. Scope change forced by the data situation

The task's alignment pipeline (CQT + subsequence DTW, per `mir-aidj/djmix-analysis`)
needs the **source track audio** for every track in every tracklist, not just the
mix. Verified against the reference repo: it downloads mix audio *and* requires a
manually-collected tracklist plus the tracks themselves. We have 25 local tracks
and no mixes, and a 50–200 mix corpus implies sourcing 500–2000 individual
tracks — a music-acquisition problem, not a scraping one.

Agreed alternative: **mix-audio-only mining**. Consequences:

| Target from the brief | Status |
|---|---|
| Transition duration in beats | Measurable, but poorly — see §3 |
| Key delta | **Recoverable and trustworthy** (never touches the probe) |
| Cue-out point in track A / cue-in in track B | **Not recoverable at all** — needs source audio |
| Tempo delta between A and B | **Not recoverable** — beatmatching erases it; the applied stretch is unobservable in a finished mix |
| Cue position within track | **Not recoverable** |

Replacements: native tempo delta → **mix-tempo trajectory** (`tempo_step_pct`,
how far a DJ lets set tempo drift, which the engine currently has no model of at
all); cue position → **Camelot-score distribution** over real transitions.

## 3. Measurement accuracy — the load-bearing result

Measured by rendering sets whose crossfade lengths are known exactly, then mining
them back through the identical pipeline. Two independent runs, 31 and 25
transitions from `combined.db`.

| Quantity | Accuracy |
|---|---|
| Transition **centre** location | ~3 s median error on 14–26 s crossfades — **good** |
| **Cut vs blend** classification | 10/12 true cuts measured under one bar; blends never collapse to cuts — **good** |
| **Blend duration** | **median \|error\| ≈ 25 beats on 32–48 beat transitions** |
| Blend duration within ±1 beat | **0%** |
| Blend duration within ±4 beats | **0%** |
| Blend duration within ±8 beats | **10%** |
| Accept rate (boundaries surviving the guards) | 71–83% |
| Confidence vs accuracy correlation | **none established** |

Representative blend measurements (truth → measured, beats):
`48→56`, `32→70`, `48→4`, `48→10`, `32→9`, `32→24`, `32→9`, `48→0`, `32→18`, `48→21`.

Errors are large, roughly symmetric in sign but dominated by underestimates, and
**not flagged by the confidence score** — confidence ran 0.45–0.67 across both
the 8-beat-error and the 48-beat-error cases.

### Why it's this bad

Diagnosed rather than assumed:

- **The band structure, not the method, is the first constraint.** A broadband
  similarity ramp — the obvious approach — recovers 3.7 bars for a known 16-bar
  crossfade, because the low band dominates spectral energy in dance music and
  the low band is a *switch, not a ramp*. Per-band measurement is mandatory.
- **Reference non-stationarity dominates the residual error.** The clean-A and
  clean-B reference spectra sit tens of seconds from the transition, and real
  tracks aren't spectrally stationary, so a mean spectrum models the audio next
  to the transition poorly. Widening the search window from 60/25 s to 90/45 s
  degraded median duration error from 1.6 s to 6.0 s.
- **Spectral leakage between bands is irreducible for percussive content.** When
  a DJ brings the incoming highs in early, STFT spread of impulsive hats makes
  the *mid* band look like it opened early too, over-reporting the mid span ~2x.
  Confirmed by rebuilding the identical mix with the highs held back.
- **A logistic can't represent a bass switch**, so the low band's span reads
  ~2.6x high and contributes centre only.

### What was tried and didn't help enough

Multi-resolution analysis (8192-point transform for the low band: 52 bins instead
of 13, reference correlation 0.229→0.053), energy-weighted ramp fitting,
coefficient pooling over a beat with adaptive fine re-fit, best-band selection by
fit quality, a self-calibrating residual floor, local re-fit around the located
centre. Each was a real improvement — together they took blend measurement from
unusable to poor, and cut detection from 1/5 to 10/12 — but none moved the ±4-beat
hit rate off zero.

## 4. Per-band automation: a negative result

The plan rated `_make_profile`'s `cp`, `in_mid_lead` and `in_high_lead` the
**highest-value** calibration target — pure taste parameters, currently
hand-picked. They are **not recoverable**.

Sweeping the true bass-swap centre across 11.5 s moves the measured centre 4.2 s.
Monotone, so it looks encouraging in isolation, but 2.7x compressed, and the
inter-band separations that would calibrate it are the same size as the
measurement error. Absolute band centres land within ±2 s on a 15–30 s crossfade,
which sounds fine until you notice that's the same order as the quantity being
measured. An earlier apparent success at recovering *band ordering* turned out to
be luck: improving the low band's resolution flipped it.

Pinned by `test_band_phase_does_not_track_cp_under_the_split_renderer`.

### ⚠ Partially superseded (2026-08-15): it was the renderer, not the estimator

The numbers above were produced by rendering the synthetic sweep through
`_split3`'s band-split — which, it turned out, did not put the bass swap where
its own automation said. The low lane deviated from the rendered result by up
to 0.48 (CHANGELOG 2026-08-15). The estimator was hunting for a swap the
renderer had smeared.

Same estimator, same sweep, only the renderer changed:

| renderer | true range | measured | compression |
|---|---|---|---|
| band-split (`EQ_TOPOLOGY="split"`) | 11.46 s | 4.21 s | **2.72x** |
| shelving (`EQ_TOPOLOGY="shelf"`) | 11.46 s | 11.18 s | **1.02x** |

The 4.21 s reproduces the 4.2 s above, so this is the same measurement that
produced the finding. §4's threshold for widening calibration scope was "below
~1.5x"; on synthetic mixes that is now met.

**What this does not establish.** The fixture renders with an *idealised*
shelf EQ. Real mixers have unknown corners and slopes, and a real DJ rides
faders in ways `_make_profile` does not model, so this is an upper bound under
ideal conditions — not a field result. Nothing here touches §3's blend-duration
error (~25 beats), which is independent and still governs. Treat `cp` as
**re-opened as a question, not recovered**: deciding it needs a sweep against
mined mixes, which has not been run.

Pinned by `test_shelf_renderer_makes_cp_recoverable_on_synthetic_mixes`.

## 5. What is honestly calibratable

Split by whether a value depends on the weak duration measurement:

**Not probe-dependent — trustworthy:**

| Value | Source |
|---|---|
| `dwell_bars` → `MIN_DWELL_BARS` / `MAX_DWELL_BARS` | Tracklist + tempo track |
| `camelot_score` → `sequencer` minimum edge score | `harmony.detect_key` on clean windows |
| `tempo_step_pct` | Tempo track (accurate to ±0.01 BPM on synthetic) |

**Probe-dependent — usable only as a central tendency over many transitions,
since per-transition error is ~25 beats:**
`duration_bars` → per-style crossfade length, `solo_bars` → `min_solo_bars`,
`cut_rate`.

**Deliberately excluded, with reasons recorded in
[`calibration.py`](infinite_dj/calibration.py) so they aren't re-litigated:**
`_make_profile` parameters (§4), `MAX_STRETCH` (a Rubber Band artifact ceiling,
and unobservable anyway), `LOW_CUT`/`MID_CUT` (the measurement's own coordinate
system — circular), `high_sim_threshold` (`library_sim_threshold` already
calibrates per-library), `choose_transition_style`'s 0.45/0.70 energy thresholds
(mined energy is normalized by the *mix's* peak, `cue.energy` by the *track's*, so
quantiles don't transfer), the cue-detector weights (they predict cue position
inside a track, which this mode cannot observe).

One design decision stated explicitly because it is a decision, not data: the
four crossfade styles are mapped onto **quantiles of a single mined
distribution** (blend←p75, fade←p50, build/swap←p25). Nothing in a mix says "this
was a swap", so the *ordering* of styles remains a design choice and only the
magnitudes are empirical.

## 6. Corpus statistics

**No real corpus exists yet** — there are no DJ mixes on this machine, and by
agreement nothing was fetched. Deliverable 2 cannot report real distributions
until mixes are placed in a folder; `dj.py corpus` produces the full report the
moment they are.

The report includes the checks that make thin or broken data visible: accept rate
and reject-reason histogram, an explicit note that rejection favours
timbrally-similar pairs (so surviving durations are **biased short**), per-genre
breakout rather than pooling, and a **phrase-multiple histogram** that must peak
near 8/16/32 bars — DJs mix on phrase boundaries, so a flat spread means the miner
is broken. That last one is the free correctness check on real data, needing no
ground truth.

Calibration refuses any value with fewer than 20 observations. On a 4-transition
demo corpus every value correctly stayed at its default and said so.

## 7. Validation

`dj.py validate` compares the engine's transition shape against the held-out
corpus (held out **per mix**, since transitions in one mix share a DJ, tracklist
and tempo track).

Two departures from the brief, both forced:

- **The comparison is distributional, not per-transition.** The corpus has no
  source tracks, so the engine cannot be run on the same pair a DJ mixed.
- **Cue points are not compared at all**, because cue position inside a source
  track is not recoverable. The brief's "±0.5 semitone" also doesn't apply —
  `harmony.py` emits discrete Camelot codes, so key agreement is reported as
  exact-match vs same-compatibility-class.

Both sides are **rendered and probed** through the identical pipeline rather than
reading the engine's automation lanes analytically. The probe's biases (a ~0.787
scale factor on symmetric profiles, ~2x on asymmetric ones, inter-band leakage)
cancel only that way; an analytic comparison would attribute every one of them to
the DJ.

The report always shows the **measurement ceiling** beside the hit rate — how
often the pipeline measures a *known* length correctly. Without it, a
measurement limitation reads as an engine defect. The ceiling for blends is 0% at
±1 and ±4 beats, so any hit rate computed at those tolerances is uninterpretable.

## 8. Recommendation

Phase 1 delivers working, reproducible, well-tested machinery and one clear
answer: **the bottleneck is measurement, not the engine, and not the choice
between rule-based and learned control.** Blend duration — the quantity the whole
exercise was meant to calibrate — comes back with ~25 beats of median error on
32–48 beat transitions and 0% inside ±4 beats, and the confidence score does not
identify the bad measurements. Distributions built on that would be dominated by
measurement error rather than real DJ variation, and a hit rate at the brief's ±1
beat tolerance is uninterpretable because the ceiling is already 0% there. Phase 2
as scoped would train on the same signal and inherit the same ceiling, so it is
not the right next step; the DJtransGAN-style approach also needs aligned source
tracks, which is the constraint we started from. Three things would actually
change the picture, in order of value per effort: **(a)** obtain even 5–10 mixes
*with* their source tracks — from your own sets or a DJ friend's — which unlocks
real DTW alignment, makes cue points recoverable, and turns the ceiling from a
guess into a measurement; **(b)** use what *is* trustworthy now, namely the
Camelot-score distribution, dwell bounds and set-tempo drift, none of which touch
the probe, and all of which are already wired and likely to loosen the
sequencer's overly strict harmonic gate; **(c)** if mix-audio-only is the only
option, treat cut-vs-blend classification and transition *placement* as the
usable outputs and drop duration calibration, since those are the two things the
probe does well. My recommendation is (a) then (b), and to leave Phase 2 closed
until (a) produces a real ceiling number.
