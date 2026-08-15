# Infinite DJ

Auto-mixing DJ engine for local music libraries. Analyzes tracks once, then sequences and mixes them in real-time — beat-matched, harmonically compatible, with 3-phase EQ crossfades that can fire at any point in a track.

## Change history for agents

Before modifying analysis, sequencing, mixing, buffering, or live playback,
read [CHANGELOG.md](CHANGELOG.md). It records the behavior changes made in this
repository and the reasoning behind them, including real-time constraints that
are not always apparent from an individual module. Add an entry there for any
meaningful behavior or architecture change, explaining both what changed and
why.

## Setup

```bash
pip install -r requirements.txt

# macOS
brew install rubberband portaudio

# Ubuntu/Debian
sudo apt install rubberband-cli libportaudio2
```

## Entry Point

`dj.py` is the CLI. All commands go through it:

```bash
python dj.py <command> [options]
python dj.py --db custom.db <command>   # override default DB path (infinite_dj.db)
```

## Commands

| Command | Purpose |
|---|---|
| `analyze <dir> [--force]` | Scan and analyze audio files (one-time) |
| `library` | List all analyzed tracks |
| `inspect <title>` | Full analysis detail for one track |
| `cues <title>` | Show scored IN/OUT cue points |
| `compatible <title> [--top N]` | Find harmonically compatible tracks |
| `mix <track_a> <track_b> --out file.wav [--bars N]` | Render a single transition |
| `sequence [--start title] [--n N] [--arc peak\|steady\|build\|wave]` | Print a sequence plan |
| `render-set --out file.wav [--n N] [--arc ...]` | Render a full set offline |
| `splice --out file.wav [--length MIN] [--min-seg S] [--max-seg S] [--layers N] [--arc ...]` | Collage set: sequential splices, or `--layers N` for a structured overlap-add collage (feature/weave/breathe) |
| `render-set`/`splice --serve [--port N]` | Render then launch the interactive web player; `--timeline PATH` also writes the timeline JSON |
| `studio [--port N] [--out-dir DIR]` | Launch the studio: pick tracks/albums, set Serendipity + Pace, then either generate a fixed-length mix or start **RADIO** (endless, plays until you hit EXIT) |
| `serve --audio file.wav [--timeline JSON] [--port N]` | Launch the web player for an already-rendered set |
| `play [--start title] [--arc ...] [--out file.wav] [--duration N]` | Real-time playback |
| `fetch [--from-gaps] [--bpm N] [--hours H] [--dry-run]` | Download screened Creative Commons tracks from the Internet Archive |
| `triage [--grade G] [--json PATH]` | Grade each track on whether the engine can mix it well |
| `gaps [--target-hours H] [--json PATH]` | What the library is missing, per the engine's real gates |
| `mine <mix_dir> [--force]` | Mine DJ mixes + tracklist sidecars for calibration data |
| `probe <mix> --idx N` | Full measurement detail for one mined boundary (debugging view) |
| `corpus [--min-confidence F] [--json PATH]` | Mined distributions + rejection-bias report |
| `calibrate [--out PATH] [--show]` | Derive engine constants from the corpus |
| `validate [--pairs N]` | Engine vs corpus, reported against the measurement ceiling |

Track arguments accept partial title matches or absolute file paths.

## Architecture

```
dj.py                    CLI — argument parsing, command dispatch
infinite_dj/
├── models.py            TrackMeta, CuePoint, Section dataclasses
├── analyzer.py          Full audio analysis pipeline (run once per track)
├── harmony.py           Krumhansl-Schmuckler key detection + Camelot wheel
├── cue_detector.py      Multi-factor IN/OUT cue point scoring
├── db.py                SQLite cache — skips re-analysis unless file changed
├── mixer.py             Beat alignment + 3-phase EQ crossfade renderer
├── sequencer.py         Compatibility graph + greedy/energy-arc sequencing
├── engine.py            Real-time streaming engine + lookahead scheduler
├── mix_grid.py          Piecewise tempo tracking for whole DJ mixes
├── transition_probe.py  Measures one transition from mix audio alone (I/O-free DSP)
├── mix_corpus.py        Tracklist parsing + corpus mining + distributions
├── calibration.py       Mined constants w/ provenance; falls back to defaults
├── validation.py        Engine-vs-corpus comparison + measurement ceiling
├── library_health.py    Per-track triage + per-library gap report (metadata only)
└── fetch_archive.py     Screened CC downloads from the Internet Archive netlabels
```

## Growing the library

The loop is `gaps` → `fetch` → `analyze` → `triage` → `gaps`:

```bash
python dj.py gaps                                  # what's missing, per the engine's gates
python dj.py fetch --from-gaps --dry-run           # what that means as a shopping list
python dj.py fetch --from-gaps                     # download it
python dj.py analyze music/archive                 # the engine's own analysis, from audio
python dj.py triage --grade reject                 # what didn't survive
```

`fetch` screens against the Archive's own Essentia sidecars (BPM, key, genre,
danceability — published beside each file) so a `--bpm` target costs ~26 KB per
track to evaluate instead of ~6 MB. Checks run cheapest-first and
`tests/test_fetch_archive.py` pins that ordering.

Screening is a *filter*, not analysis: nothing from Essentia reaches the track
database. `analyze` re-derives BPM, key, cues and grid from the audio, as always.

⚠ **To predict what the engine will do, call `plan_transition` — never re-derive
it.** `gaps` originally called `choose_transition_style` on `best_cue_out` /
`best_cue_in`, which the set renderer never uses (it exits via `_pick_exit_cue`,
with a groove floor, and enters via `_match_entry`, which energy-matches the
exit). That put median exit energy at 0.15 instead of 0.51 and reported `swap`
as unreachable when it fires on 6.3% of pairs. See CHANGELOG.

Downloads append attribution to `PROVENANCE.jsonl` in the destination. Much of
the netlabels collection is `by-nc-nd` — fine for listening, not for publishing
a mix built from it; use `--license` to filter.

## Calibration from real DJ mixes

`CALIBRATION.md` has the Phase 1 results and the Phase 2 recommendation. **Read
it before touching `transition_probe.py` or extending the mining** — it records
what is and isn't recoverable from mix audio, and two negative results that are
expensive to rediscover (per-band automation phase isn't recoverable; blend
duration carries ~25 beats of error, so only its central tendency is usable).

The engine reads `calibration.json` if present. Absent or thin, every value falls
back to the constant it replaced, and rendering is byte-identical — pinned by
`tests/test_calibration.py`.

## Key Design Details

### Analysis pipeline (`analyzer.py`)
- ~10-30s per track; results cached in SQLite by file hash
- Extracts: BPM, Camelot key (Krumhansl-Schmuckler on chroma), beat/downbeat/phrase grid, 1s-resolution energy curve, structural sections, scored IN/OUT cue points at every downbeat, integrated loudness (RMS dBFS)
- **Rigid equidistant beat grid** (`_refine_tempo_phase`): librosa's `beat_track` supplies the metrical level (octave), then the tempo is refined to a precise constant value and the global beat phase is found by autocorrelation of the onset envelope (finer `BEAT_HOP=256`). Beats are laid down as a perfectly equidistant grid (`arange(phase, dur, 60/bpm)`) rather than following the audio — so two beatmatched tracks stay phase-locked over a long crossfade instead of drifting (the "beatmatch sounds off" cause). Reimplemented from Vande Veire & De Bie's DnB auto-DJ (ideas, not code — that repo is AGPL).
- **Octave fold**: the refined tempo is folded into `[BPM_MIN, BPM_MAX)` (90–180).
- **Downbeat anchoring**: bar-1 is the beat phase (of every 4) carrying the most onset energy — not a naive `beats[::4]`

### Cue point scoring (`cue_detector.py`)
- Scans the full track (not just start/end windows) — any phrase boundary can be an entry or exit
- OUT score: `phrase_boundary×3.5 + energy_valley×2.5 + falling_trend×1.5 + sparse_texture×1.0 + low_absolute_energy×0.5`
- IN score: `phrase_boundary×3.5 + rising_trend×1.5 + energy_valley×1.5 + sparse_texture×1.0 + low_absolute_energy×0.5`
- `top_k = 5` cue points per type per track

### Mixing (`mixer.py`)
- **Breakpoint EQ automation** (`TransitionProfile` / `_make_profile`): a crossfade is described by per-track **automation lanes** (piecewise-linear `(phase, value)` breakpoints) for volume + each of low/mid/high. Bass swaps single-source; mid and high crossfade with independent timing (e.g. a `swap` brings the incoming hats in early but holds its mids back).
- **Shelving EQ renders those lanes** (`ShelfEQState` / `ShelfCrossfadeState`, `EQ_TOPOLOGY = "shelf"`): a minimum-phase cascade of shelving + peaking biquads, so the magnitude response *is* the automation curve. Stateful, with coefficients updated on a **global** control grid every `EQ_CTRL_HOP` samples and carried across chunks, so real-time chunked rendering matches offline sample-for-sample regardless of where chunks fall. Topology from Vande Veire & De Bie (AGPL) — reimplemented from the RBJ cookbook.
- ⚠ **`_split3` band-splitting is legacy** (`EQ_TOPOLOGY = "split"`, `CrossfadeFilterState`). It re-sums three causal-IIR bands, which only reconstruct when their gains are *equal* — so cutting a band phase-shifts it into its neighbour instead of removing it. The bass swap deviated from its own automation curve by up to 0.48 and boosted 80–200 Hz by +5 dB when asked to kill it. Kept only for A/B. **Don't build on it.** See CHANGELOG 2026-08-15.
- Build streaming filter state with `make_crossfade_state()`, never by naming a state class — that's what keeps live playback and offline rendering on the same topology.
- **Adaptive crossfade styles** (`TransitionStyle` / `choose_transition_style`): the crossfade's length and per-band automation are chosen from the energy at the exit and entry cues:
  - `blend` (breakdown→intro, both sparse): long 16-bar smooth blend
  - `swap` (drop→drop, both busy): short 8-bar, incoming highs held back, quick bass swap
  - `fade` (busy→calm): medium 12-bar gentle
  - `build` (calm→rising): short 8-bar, incoming brought up sooner
  - `cut` (tempos incompatible): a short ~0.3s fade — never a long overlap of two unsynced grooves
- Shared `_blend(out, in, phase, style)` primitive (offline + engine): style-shaped high crossfade + single-source bass swap (only one kick at a time)
- Loudness-matched to a fixed `MASTER_LOUDNESS` target
- **Stretch budget** (`MAX_STRETCH = 0.08`, half/double aware); beyond budget → `cut`
- Time-stretch via Rubber Band; a downbeat at native time `d` maps to `d / ratio` after stretching (ratio > 1 speeds up)

### Full-set rendering (`render_set` in `mixer.py`)
- Lays all tracks on ONE continuous timeline: each plays solo at its native tempo, consecutive tracks overlap only during an adaptive crossfade, only the final track fades out. No silence gaps, no double-rendered tracks.
- **Breathing room**: a track plays a substantial solo (`min_solo_bars`, default 32 or the calibrated value) and only exits at a strong, phrase-aligned OUT cue past that dwell.
- ⚠ `render_set`'s `n_mix_bars` parameter is **dead**: `style.n_bars` overwrites it. Crossfade length is only reachable through `choose_transition_style`.
- Per-transition tempo reference (outgoing track's native tempo) — no global tempo lock/drift.
- Output is 16-bit PCM at the source sample rate (44.1 kHz).
- Returns `(audio, sr, [SetMarker], [clip])`; `render-set` prints transition timestamps, style + stretch.

### Set sequencing (`sequence_for_mixing` in `sequencer.py`)
- The sequencer `render-set` uses: strongly prefers beat-matchable (tempo-compatible) neighbours so the render uses gentle blends rather than hard cuts, then harmony and energy-arc fit break ties. Produces a no-repeat permutation for a full set.

### Real-time engine (`engine.py`)
Three threads:
- **Producer**: decodes audio, executes crossfades, fills ring buffer (deque of chunks)
- **Scheduler**: every 500ms, looks `LOOKAHEAD_BARS` ahead for high-confidence OUT cues
- **Audio callback**: `sounddevice` pulls from ring buffer

Scheduler fires a transition when:
1. A good OUT cue is within 8 bars AND at least `MIN_DWELL_BARS` have played, OR
2. `MAX_DWELL_BARS` have elapsed (hard cap)

Key constants in `engine.py`:
- `MIN_DWELL_BARS = 32` — minimum bars before early exit
- `MAX_DWELL_BARS = 96` — hard cap, forces transition
- `LOOKAHEAD_BARS = 16` — scheduler lookahead window
- `BUFFER_SECONDS = 8.0` — ring buffer size (increase if glitching)

The dwell bounds are read through `_dwell_bounds()`, which prefers mined values
from `calibration.py` and falls back to the constants above.

### Harmonic compatibility (`harmony.py`)
Camelot wheel scoring used by both `compatible` command and sequencer:
- Same key: 1.0 | Parallel major/minor: 0.9 | ±1 step: 0.8 | ±2 steps: 0.6 | ±3 steps: 0.3
- Cross-mode (A↔B) within ±1 step: 0.5 | else: 0.0

### Sequencing (`sequencer.py`)
- Compatibility graph edge: `0.6 × harmonic_score + 0.4 × bpm_compatibility`
- `MIN_SCORE = 0.3` — minimum to add an edge
- Arc shapes: `peak` (build to peak then down), `steady`, `build`, `wave`

## Data Model

`TrackMeta` (the central object):
- Identity: `file_path`, `title`, `duration`
- Rhythm: `bpm`, `bpm_confidence`, `beats[]`, `downbeats[]`, `phrases[]`
- Harmony: `key` (Camelot, e.g. "8B"), `key_name` (e.g. "C major"), `key_confidence`
- Energy: `energy_curve[]` — normalized RMS per second
- Loudness: `loudness` — integrated RMS in dBFS (negative); used for gain-matching transitions
- Structure: `sections[]` (Section: start/end/label/energy), `cue_points[]` (CuePoint: timestamp/type/phrase_aligned/energy/confidence)

## Supported Audio Formats

`.mp3`, `.flac`, `.wav`, `.aiff`, `.aif`, `.ogg`, `.m4a`

## Dependencies

- `librosa` — audio analysis
- `soundfile` — audio I/O
- `numpy`, `scipy` — signal processing
- `pedalboard` — EQ filters
- `pyrubberband` — time-stretching (requires `rubberband` binary)
- `sounddevice` — real-time audio output (optional; falls back to headless)

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
