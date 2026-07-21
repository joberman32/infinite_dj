# Infinite DJ

Auto-mixing DJ engine for local music libraries. Analyzes your library once,
then sequences and mixes tracks in real-time — beat-matched, harmonically
compatible, with EQ-shaped transitions that can fire at any point in a track.

## Setup

```bash
pip install -r requirements.txt

# macOS
brew install rubberband portaudio

# Ubuntu/Debian
sudo apt install rubberband-cli libportaudio2
```

## Full Workflow

### 1. Analyze your library (one-time)
```bash
python dj.py analyze ~/Music/
python dj.py analyze ~/Music/ --force   # re-analyze everything
```

### 2. Browse & explore
```bash
python dj.py library                        # all tracks: BPM, key, cues, sections
python dj.py inspect "song name"            # full detail: energy curve, cue points
python dj.py cues "song name"               # all IN/OUT cue points + scores
python dj.py compatible "song name" --top 10  # best harmonic matches
```

### 3. Mix two specific tracks (offline render)
```bash
python dj.py mix "track a" "track b" --out transition.wav
python dj.py mix "track a" "track b" --out transition.wav --bars 16
```

### 4. Preview a sequence
```bash
python dj.py sequence --n 10 --arc peak
python dj.py sequence --start "opening track" --arc build
```
Arc options: `peak` | `steady` | `build` | `wave`

### 5. Render a full set offline
```bash
python dj.py render-set --out set.wav --n 20 --arc peak
```

### 6. Live infinite DJ (real-time)
```bash
python dj.py play                          # plays through speakers
python dj.py play --arc wave               # wave energy arc
python dj.py play --start "first track"    # specific opener
python dj.py play --out preview.wav --duration 120  # render 2min preview
```
Press Ctrl+C to stop live playback.

---

## Architecture

```
dj.py                    CLI — 9 commands
infinite_dj/
├── models.py            TrackMeta, CuePoint, Section dataclasses
├── analyzer.py          Full audio analysis pipeline (run once per track)
├── harmony.py           Krumhansl-Schmuckler key detection + Camelot wheel
├── cue_detector.py      Multi-factor IN/OUT cue point scoring (full-track scan)
├── db.py                SQLite cache — skips re-analysis unless file changed
├── mixer.py             Beat alignment + 3-phase EQ crossfade renderer
├── sequencer.py         Compatibility graph + greedy/energy-arc sequencing
└── engine.py            Real-time streaming engine + lookahead scheduler
```

## How it works

### Analysis (one-time, ~10-30s per track)
Every track gets fingerprinted: BPM, Camelot key (via Krumhansl-Schmuckler
on chroma features), beat/downbeat/phrase grid, 1s-resolution energy curve,
structural sections, and scored IN/OUT cue points at every downbeat.

### Cue point detection
Every downbeat is scored as a potential transition point:

```
OUT point score = phrase_boundary×3.5 + energy_valley×2.5
               + falling_trend×1.5 + sparse_texture×1.0
               + low_absolute_energy×0.5

IN point score  = phrase_boundary×3.5 + rising_trend×1.5
               + energy_valley×1.5 + sparse_texture×1.0
               + low_absolute_energy×0.5
```

Phase 3: scans the entire track, not just start/end windows. Any phrase
boundary anywhere can be an entry or exit point.

### Mixing (3-phase EQ crossfade)
```
Phase 1 (bars 1-N/3):    Outgoing full + incoming highs fade in
Phase 2 (bars N/3-2N/3): Bass swap — cut outgoing bass, bring incoming bass
Phase 3 (bars 2N/3-N):   Incoming full + outgoing highs fade out
```
BPM gaps resolved via Rubber Band time-stretching (up to ~12% cleanly).

### Real-time engine (play command)
Three threads:
- **Producer**: decodes audio, executes crossfades, fills ring buffer
- **Scheduler**: monitors playback position every 500ms, looks LOOKAHEAD_BARS
  ahead for high-confidence OUT cues, fires transitions opportunistically
- **Audio callback**: sounddevice pulls from ring buffer in real-time

Scheduler logic:
1. Always pre-select the next track (so it's ready instantly)
2. If a good OUT cue is coming within 8 bars AND we've played ≥16 bars → fire
3. Hard cap at 64 bars forces a transition regardless

### Harmonic compatibility (Camelot wheel)
```
Same key:              1.0
Parallel major/minor:  0.9   (e.g. C major → C minor)
±1 step on wheel:      0.8   (e.g. C major → G major, perfect 5th)
±2 steps:              0.6
±3 steps:              0.3
Everything else:       0.0
```

## Key parameters to tune

In `engine.py`:
- `MIN_DWELL_BARS = 16` — minimum bars before an early exit is allowed
- `MAX_DWELL_BARS = 64` — hard cap; forces transition
- `LOOKAHEAD_BARS = 16` — how far ahead the scheduler looks for OUT cues
- `BUFFER_SECONDS = 8.0` — audio buffer size (increase if you get glitches)

In `cue_detector.py`:
- `top_k = 5` — max cue points per type per track

In `sequencer.py`:
- `harm_weight = 0.6, rhythm_weight = 0.4` — compatibility scoring weights
- `MIN_SCORE = 0.3` — minimum edge score to accept a transition
