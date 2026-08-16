"""
CLAP embeddings (`CuePoint.embedding`) are high-quality but optional — dormant
on a fresh install with no `torch`/`transformers`. `cue_cosine_similarity` had
no other signal to fall back on, so `find_best_cue_pair` and
`choose_transition_style`'s similarity term silently did nothing for anyone
who hadn't separately installed CLAP.

`get_mfcc_timbre` reuses the analysis pipeline's existing MFCC machinery
(already computed once per track for section-novelty detection, just
discarded afterward) as a lightweight, always-available fallback: pooled
mean+std of 12 MFCC coefficients (dropping coefficient 0, which tracks
loudness rather than spectral shape) over the same OUT/IN window convention
CLAP uses. It lives in a separate `CuePoint.timbre` field rather than reusing
`.embedding`, so a library with a mix of CLAP-analyzed and MFCC-analyzed
tracks can never have the two vector kinds cosine-compared against each other.

These tests pin: the extraction window convention, the coefficient-0 drop,
and the embedding/fallback/neither precedence in `cue_cosine_similarity` —
the single choke point `find_best_cue_pair`, `choose_transition_style`, and
`library_sim_threshold` all consume without any changes of their own.
"""
import numpy as np
import pytest

from infinite_dj.embeddings import get_mfcc_timbre
from infinite_dj.models import CuePoint
from infinite_dj.sequencer import cue_cosine_similarity

SR = 22050


def _tone(freq: float, seconds: float = 20.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _cosine(v1, v2) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ── get_mfcc_timbre ────────────────────────────────────────────────────────

def test_returns_a_pooled_vector_of_the_expected_length():
    """13 MFCCs, minus coefficient 0, pooled as mean+std -> 24 floats."""
    v = get_mfcc_timbre(_tone(440), SR, 10.0, "out")
    assert v is not None
    assert len(v) == 24


def test_same_stationary_texture_reads_as_identical_from_different_windows():
    """A pure tone's timbre doesn't depend on which second of it you sample."""
    signal = _tone(440)
    v_a = get_mfcc_timbre(signal, SR, 5.0, "out")
    v_b = get_mfcc_timbre(signal, SR, 15.0, "in")
    assert _cosine(v_a, v_b) > 0.99


def test_different_textures_read_as_dissimilar():
    """The property the whole feature rests on: real discrimination, not noise."""
    tone = _tone(440)
    noise = np.random.RandomState(0).randn(len(tone)).astype(np.float32) * 0.3
    v_tone = get_mfcc_timbre(tone, SR, 10.0, "out")
    v_noise = get_mfcc_timbre(noise, SR, 10.0, "out")
    assert _cosine(v_tone, v_noise) < 0.5


def test_out_looks_backward_and_in_looks_forward_from_the_timestamp():
    """Same window convention as CLAP's extract_embedding: at a boundary where
    the texture changes, OUT and IN must see opposite sides, not the same audio.
    """
    low = _tone(220, seconds=10.0)
    high = _tone(3000, seconds=10.0)
    signal = np.concatenate([low, high])

    # At the boundary (t=10s): OUT looks back into the low tone, IN looks
    # forward into the high one.
    v_out = get_mfcc_timbre(signal, SR, 10.0, "out")
    v_in = get_mfcc_timbre(signal, SR, 10.0, "in")
    assert v_out is not None and v_in is not None
    assert _cosine(v_out, v_in) < 0.9


def test_empty_window_returns_none():
    v = get_mfcc_timbre(_tone(440, seconds=1.0), SR, 0.0, "out", window_sec=8.0)
    assert v is None


# ── cue_cosine_similarity precedence ────────────────────────────────────────

def _cue(**kwargs) -> CuePoint:
    base = dict(timestamp=0.0, type="out", phrase_aligned=True,
                energy=0.5, confidence=0.8)
    base.update(kwargs)
    return CuePoint(**base)


def test_prefers_clap_embedding_when_both_cues_have_one():
    c1 = _cue(embedding=[1.0, 0.0], timbre=[1.0, 0.0])
    c2 = _cue(embedding=[1.0, 0.0], timbre=[0.0, 1.0])   # CLAP agrees, MFCC would disagree
    assert cue_cosine_similarity(c1, c2) == pytest.approx(1.0)


def test_falls_back_to_mfcc_timbre_when_neither_cue_has_clap():
    c1 = _cue(timbre=[1.0, 0.0])
    c2 = _cue(timbre=[1.0, 0.0])
    assert cue_cosine_similarity(c1, c2) == pytest.approx(1.0)


def test_mismatched_vector_kinds_are_not_compared():
    """One cue only has CLAP, the other only has MFCC — never mix spaces."""
    c1 = _cue(embedding=[1.0, 0.0])
    c2 = _cue(timbre=[1.0, 0.0])
    assert cue_cosine_similarity(c1, c2) is None


def test_neither_vector_present_returns_none():
    c1 = _cue()
    c2 = _cue()
    assert cue_cosine_similarity(c1, c2) is None
