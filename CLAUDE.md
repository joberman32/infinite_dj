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
| `serve --audio file.wav [--timeline JSON] [--port N]` | Launch the web player for an already-rendered set |
| `play [--start title] [--arc ...] [--out file.wav] [--duration N]` | Real-time playback |

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
└── engine.py            Real-time streaming engine + lookahead scheduler
```

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
- **3-band breakpoint EQ** (`_split3` / `CrossfadeFilterState` / `TransitionProfile`): each track is split into low/mid/high by difference-of-lowpass (perfect reconstruction) — a DJ-mixer EQ. A crossfade is described by per-track **automation lanes** (piecewise-linear `(phase, value)` breakpoints) for volume + each band, built by `_make_profile`. Bass swaps single-source; mid and high crossfade with independent timing (e.g. a `swap` brings the incoming hats in early but holds its mids back). Stateful (`CrossfadeFilterState`) so real-time chunked rendering matches offline sample-for-sample. Idea from Vande Veire & De Bie (AGPL) — reimplemented.
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
- **Breathing room**: a track plays a substantial solo (`min_solo_bars`, default 32) and only exits at a strong, phrase-aligned OUT cue past that dwell.
- Per-transition tempo reference (outgoing track's native tempo) — no global tempo lock/drift.
- Output is 16-bit PCM at the source sample rate (44.1 kHz).
- Returns `(audio, sr, [SetMarker])`; `render-set` prints transition timestamps, style + stretch.

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
- `MIN_DWELL_BARS = 16` — minimum bars before early exit
- `MAX_DWELL_BARS = 64` — hard cap, forces transition
- `LOOKAHEAD_BARS = 16` — scheduler lookahead window
- `BUFFER_SECONDS = 8.0` — ring buffer size (increase if glitching)

### Harmonic compatibility (`harmony.py`)
Camelot wheel scoring used by both `compatible` command and sequencer:
- Same key: 1.0 | Parallel major/minor: 0.9 | ±1 step: 0.8 | ±2 steps: 0.6 | ±3 steps: 0.3 | else: 0.0

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
