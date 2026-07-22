# Infinite DJ

Auto-mixing DJ engine for local music libraries. Analyzes tracks once, then sequences and mixes them in real-time — beat-matched, harmonically compatible, with 3-phase EQ crossfades that can fire at any point in a track.

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
- **BPM octave normalization**: `beat_track` often locks to half/double time; detected tempo is folded into the `[BPM_MIN, BPM_MAX)` octave (90–180) and the beat grid is re-gridded to match (midpoints inserted when doubling, decimated when halving) — not just the BPM number relabeled
- **Downbeat anchoring**: bar-1 is the beat phase (of every 4) carrying the most onset energy, via `librosa.onset.onset_strength` — not a naive `beats[::4]`

### Cue point scoring (`cue_detector.py`)
- Scans the full track (not just start/end windows) — any phrase boundary can be an entry or exit
- OUT score: `phrase_boundary×3.5 + energy_valley×2.5 + falling_trend×1.5 + sparse_texture×1.0 + low_absolute_energy×0.5`
- IN score: `phrase_boundary×3.5 + rising_trend×1.5 + energy_valley×1.5 + sparse_texture×1.0 + low_absolute_energy×0.5`
- `top_k = 5` cue points per type per track

### Mixing (`mixer.py`)
- Shared `_blend(out, in, phase)` primitive (used by both offline renderer and real-time engine):
  - Highs/mids: equal-power crossfade across the whole region (never both tracks at full → no mud)
  - Bass: single-source swap around the midpoint (only one kick plays at a time)
- Loudness-matched: incoming track gain-adjusted toward the outgoing track's (offline set uses a fixed `MASTER_LOUDNESS` target)
- **Stretch budget** (`MAX_STRETCH = 0.08`): `TransitionPlan.__post_init__` picks the least-stretch match (direct / half / double time). Within budget → beatmatch; beyond → a short 4-bar `cut` (no tempo mangling). `plan.beatmatched` / `plan.method` expose the decision.
- Time-stretch via Rubber Band; a downbeat at native time `d` maps to `d / ratio` after stretching (ratio > 1 speeds up)
- Default mix length: 16 bars

### Full-set rendering (`render_set` in `mixer.py`)
- Lays all tracks on ONE continuous timeline: each plays solo at its native tempo, consecutive tracks overlap only during a beat-locked crossfade, only the final track fades out. No silence gaps, no double-rendered tracks.
- Per-transition tempo reference (the outgoing track's native tempo) — no global tempo lock/drift; only genuinely far-apart pairs become cuts.
- Returns `(audio, sr, [SetMarker])`; `render-set` prints transition timestamps + method.

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
