"""
Engine constants, optionally grounded in a mined corpus of real DJ mixes.

The engine's transition behaviour is set by hand-picked numbers: crossfade
lengths of 16/12/8/8 bars in `choose_transition_style`, `min_solo_bars=32`,
`MIN_DWELL_BARS`/`MAX_DWELL_BARS`, the sequencer's minimum edge score. This
module lets a corpus replace them, while keeping every default in Python so the
engine runs byte-identically when no corpus exists.

## How a value gets used

Each field carries the number of observations behind it, and a value backed by
fewer than `MIN_N` is **refused** in favour of the default. That's the
degradation mechanism: the corpus can grow field by field, and a thin corpus
falls back rather than injecting noise. `dj.py calibrate --show` prints which is
which, because a silently-applied bad number is worse than no number.

## Probe-dependent vs not — read this before trusting a value

The mined statistics do not all have the same standing, and the difference is
large enough to matter:

  - **Not probe-dependent** — `dwell_bars` (announced start to announced start,
    via the tempo track), `camelot_score` (key detection on clean windows),
    `tempo_step_pct` (the tempo track). These involve no transition measurement
    and are as good as the tracklist and `analyzer`/`harmony` already are.
  - **Probe-dependent** — `transition_bars`, `solo_bars`, `cut_rate`. These come
    from `transition_probe`, whose absolute duration accuracy on real material is
    ~8 s median error on 14-26 s crossfades. Errors are roughly zero-mean, so a
    *central tendency* over many transitions is still estimable, but a single
    measurement is not trustworthy and the distribution's spread is inflated by
    measurement error rather than reflecting real DJ variation.

There is also a systematic scale factor: the probe measures the 10-90% travel of
a band's fader, which is ~0.787 of nominal crossfade length for a symmetric
profile. Rather than inverting that factor here — which would extrapolate a
synthetic measurement onto real material — probe-dependent values are meant to be
consumed through the render-and-probe comparison in `validate`, where the same
estimator runs on both sides and the factor cancels. `PROBE_DEPENDENT` marks them.

## What is deliberately not here

Per-band automation phase (`_make_profile`'s `cp`, `in_mid_lead`,
`in_high_lead`). The plan for this work rated those the highest-value target;
measurement says otherwise. Sweeping the true bass-swap centre 11.5 s moves the
measured one 4.2 s — monotone but 2.7x compressed, with the inter-band
separations that would calibrate it the same size as the measurement error. See
`transition_probe`'s module notes. Also absent, and unchanged from the plan's
reasoning: `MAX_STRETCH` (a Rubber Band artifact ceiling, and the applied stretch
is unobservable in a finished mix), `LOW_CUT`/`MID_CUT` (they are the measurement's
own coordinate system, so calibrating them from their output is circular),
`high_sim_threshold` (`sequencer.library_sim_threshold` already calibrates it
per-library), `choose_transition_style`'s 0.45/0.70 energy thresholds (a mined
energy is normalised by the *mix's* peak and `cue.energy` by the *track's*, so
quantiles don't transfer), and the cue-detector weights (they predict cue position
inside a source track, which mix-only mining cannot observe at all).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional

CALIBRATION_VERSION = 1

# Minimum observations before a mined value is preferred over its default.
MIN_N = 20

DEFAULT_PATH = "calibration.json"

# Values derived from `transition_probe` measurements rather than from the
# tracklist, tempo track or key detector. See the module docstring.
PROBE_DEPENDENT = frozenset({
    "blend_bars", "fade_bars", "build_bars", "swap_bars",
    "min_solo_bars", "cut_rate",
})


@dataclass(frozen=True)
class Value:
    """One calibrated number, with enough provenance to audit it."""
    value: float
    n: int = 0
    source: str = "default"          # "default" | "p25" | "p50" | ...
    spread: Optional[tuple] = None   # (low, high) percentiles, when mined

    @property
    def is_mined(self) -> bool:
        return self.source != "default" and self.n >= MIN_N


def _default(v: float) -> Value:
    return Value(value=v, n=0, source="default")


@dataclass(frozen=True)
class Calibration:
    """
    Engine constants. Defaults are exactly today's hardcoded values, so an absent
    or empty calibration file is a no-op.
    """
    # Crossfade length per transition style (bars). See `apply_from_stats` for
    # why the four styles are mapped onto quantiles of one distribution.
    blend_bars: Value = field(default_factory=lambda: _default(16))
    fade_bars: Value = field(default_factory=lambda: _default(12))
    build_bars: Value = field(default_factory=lambda: _default(8))
    swap_bars: Value = field(default_factory=lambda: _default(8))

    # How long a track plays alone before it may mix out (mixer.render_set).
    min_solo_bars: Value = field(default_factory=lambda: _default(32))

    # Real-time scheduler dwell bounds (engine.py).
    min_dwell_bars: Value = field(default_factory=lambda: _default(32))
    max_dwell_bars: Value = field(default_factory=lambda: _default(96))

    # Minimum compatibility score for a sequencer graph edge.
    harmonic_min_score: Value = field(default_factory=lambda: _default(0.3))

    # Reporting-only for now: no consumer reads these yet.
    cut_rate: Value = field(default_factory=lambda: _default(0.0))
    tempo_step_pct: Value = field(default_factory=lambda: _default(0.0))

    corpus: dict = field(default_factory=dict)

    def get(self, name: str) -> float:
        """The value to use for `name` — mined if well-supported, else default."""
        v: Value = getattr(self, name)
        if v.is_mined:
            return v.value
        return _DEFAULTS[name]

    def provenance(self, name: str) -> str:
        v: Value = getattr(self, name)
        if v.is_mined:
            tag = "mined" + (", probe-dependent" if name in PROBE_DEPENDENT else "")
            return f"{v.value} ({tag}, n={v.n}, {v.source})"
        if v.n:
            return (f"{_DEFAULTS[name]} (default; corpus had n={v.n} < {MIN_N})")
        return f"{_DEFAULTS[name]} (default)"

    def describe(self) -> str:
        lines = ["── Calibration ──"]
        if self.corpus:
            lines.append("corpus: " + ", ".join(f"{k}={v}"
                                                for k, v in self.corpus.items()))
        else:
            lines.append("corpus: none — every value is a built-in default")
        for f in fields(self):
            if f.name in ("corpus",):
                continue
            lines.append(f"  {f.name:<20} {self.provenance(f.name)}")
        return "\n".join(lines)


_DEFAULTS = {f.name: f.default_factory().value
             for f in fields(Calibration) if f.name != "corpus"}


def _value_from_dict(d) -> Optional[Value]:
    if not isinstance(d, dict) or "value" not in d:
        return None
    spread = d.get("spread")
    return Value(value=float(d["value"]), n=int(d.get("n", 0)),
                 source=str(d.get("source", "default")),
                 spread=tuple(spread) if spread else None)


def load(path: str = DEFAULT_PATH) -> Calibration:
    """
    Read a calibration file. A missing, empty or malformed file yields defaults —
    calibration is an optional refinement, never a hard dependency.
    """
    if not path or not os.path.exists(path):
        return Calibration()
    try:
        with open(path, "r") as fh:
            raw = json.load(fh)
    except (ValueError, OSError):
        return Calibration()
    values = raw.get("values") or {}
    kwargs = {}
    for f in fields(Calibration):
        if f.name == "corpus":
            continue
        v = _value_from_dict(values.get(f.name))
        if v is not None:
            kwargs[f.name] = v
    return Calibration(corpus=raw.get("corpus") or {}, **kwargs)


def to_dict(cal: Calibration) -> dict:
    out = {"version": CALIBRATION_VERSION, "corpus": cal.corpus, "values": {}}
    for f in fields(Calibration):
        if f.name == "corpus":
            continue
        v: Value = getattr(cal, f.name)
        if v.source == "default":
            continue
        d = {"value": v.value, "n": v.n, "source": v.source}
        if v.spread:
            d["spread"] = list(v.spread)
        if f.name in PROBE_DEPENDENT:
            d["probe_dependent"] = True
        out["values"][f.name] = d
    return out


def save(cal: Calibration, path: str = DEFAULT_PATH):
    with open(path, "w") as fh:
        json.dump(to_dict(cal), fh, indent=2)


def build_from_stats(stats: dict) -> Calibration:
    """
    Turn a `mix_corpus.corpus_stats` result into a Calibration.

    Mapping decisions, stated because they are decisions rather than data:

      - **The four crossfade styles are mapped onto quantiles of one
        distribution.** We have no style labels for real transitions — nothing in
        a mix says "this was a swap" — so the *ordering* of the styles (blend
        longest, swap/build shortest) stays a design choice and only the
        magnitudes come from the corpus. blend<-p75, fade<-p50, build/swap<-p25.
      - `min_solo_bars` takes the corpus p25 rather than the median, matching its
        role as a floor rather than a typical value.
      - Dwell bounds take p10/p90, so the scheduler's hard cap reflects how long
        real DJs actually let a track run.
      - `harmonic_min_score` takes the p10 of the observed Camelot scores. Real
        DJs mix out of key far more than the Camelot ladder implies, so this is
        likely to *loosen* the sequencer, and it is one of the trustworthy
        statistics — it never touches the probe.
    """
    kwargs = {}

    def q(dist, key, source):
        if not dist:
            return None
        return Value(value=round(float(dist[key]), 4), n=int(dist["n"]),
                     source=source,
                     spread=(dist.get("p10"), dist.get("p90")))

    dur = stats.get("duration_bars")
    if dur:
        for name, key in (("blend_bars", "p75"), ("fade_bars", "p50"),
                          ("build_bars", "p25"), ("swap_bars", "p25")):
            v = q(dur, key, key)
            if v:
                kwargs[name] = v

    solo = q(stats.get("solo_bars"), "p25", "p25")
    if solo:
        kwargs["min_solo_bars"] = solo

    dwell = stats.get("dwell_bars")
    if dwell:
        lo, hi = q(dwell, "p10", "p10"), q(dwell, "p90", "p90")
        if lo:
            kwargs["min_dwell_bars"] = lo
        if hi:
            kwargs["max_dwell_bars"] = hi

    cam = q(stats.get("camelot_score"), "p10", "p10")
    if cam:
        kwargs["harmonic_min_score"] = cam

    if stats.get("cut_rate") is not None and stats.get("n_at_confidence"):
        kwargs["cut_rate"] = Value(value=float(stats["cut_rate"]),
                                   n=int(stats["n_at_confidence"]),
                                   source="rate")
    step = q(stats.get("tempo_step_pct"), "p50", "p50")
    if step:
        kwargs["tempo_step_pct"] = step

    return Calibration(corpus=stats.get("counts") or {}, **kwargs)


# ── Process-wide access ──────────────────────────────────────────────────────
#
# The engine reads calibration from one lazily-loaded instance. A module-level
# singleton rather than threading a parameter through render_set, the engine and
# the sequencer: this is ambient configuration, and the alternative is a
# parameter on every call site that nothing ever passes explicitly.

_active: Optional[Calibration] = None


def active() -> Calibration:
    global _active
    if _active is None:
        _active = load(os.environ.get("INFINITE_DJ_CALIBRATION", DEFAULT_PATH))
    return _active


def set_active(cal: Optional[Calibration]):
    """Override the active calibration (tests, and `--calibration` on the CLI)."""
    global _active
    _active = cal


def reset():
    """Forget the loaded calibration so the next `active()` re-reads it."""
    global _active
    _active = None


def value(name: str) -> float:
    """Shorthand for `active().get(name)` at a call site."""
    return active().get(name)
