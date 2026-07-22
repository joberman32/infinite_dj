"""
Tests for CLAP embedding features in Infinite DJ.
"""

import os
import tempfile
import numpy as np
from infinite_dj.models import CuePoint, TrackMeta, Section
from infinite_dj.sequencer import cue_cosine_similarity, find_best_cue_pair
from infinite_dj.db import TrackDB


def test_cue_point_embedding_serialization():
    embedding = [0.1 * i for i in range(10)]
    cue = CuePoint(
        timestamp=12.5,
        type="out",
        phrase_aligned=True,
        energy=0.8,
        confidence=0.9,
        embedding=embedding,
    )
    d = cue.to_dict()
    assert d["embedding"] == embedding

    restored = CuePoint.from_dict(d)
    assert restored.timestamp == 12.5
    assert restored.type == "out"
    assert restored.embedding == embedding


def test_cue_cosine_similarity():
    v1 = [1.0, 0.0, 0.0] + [0.0] * 509
    v2 = [1.0, 0.0, 0.0] + [0.0] * 509
    v3 = [0.0, 1.0, 0.0] + [0.0] * 509

    c1 = CuePoint(10.0, "out", True, 0.5, 0.9, embedding=v1)
    c2 = CuePoint(20.0, "in", True, 0.5, 0.9, embedding=v2)
    c3 = CuePoint(30.0, "in", True, 0.5, 0.9, embedding=v3)

    sim_identical = cue_cosine_similarity(c1, c2)
    assert sim_identical is not None
    assert abs(sim_identical - 1.0) < 1e-3

    sim_orthogonal = cue_cosine_similarity(c1, c3)
    assert sim_orthogonal is not None
    assert abs(sim_orthogonal - 0.0) < 1e-3


def test_find_best_cue_pair():
    v_out1 = [1.0, 0.0] + [0.0] * 510
    v_out2 = [0.0, 1.0] + [0.0] * 510

    v_in1  = [0.0, 1.0] + [0.0] * 510
    v_in2  = [1.0, 0.0] + [0.0] * 510

    cue_out1 = CuePoint(30.0, "out", True, 0.5, 0.8, embedding=v_out1)
    cue_out2 = CuePoint(60.0, "out", True, 0.5, 0.8, embedding=v_out2)

    cue_in1  = CuePoint(10.0, "in", True, 0.5, 0.8, embedding=v_in1)
    cue_in2  = CuePoint(20.0, "in", True, 0.5, 0.8, embedding=v_in2)

    track_out = TrackMeta(
        file_path="/tmp/track_out.wav",
        title="Track Out",
        duration=120.0,
        bpm=124.0,
        bpm_confidence=0.9,
        beats=[0.0, 0.5],
        downbeats=[0.0, 2.0],
        phrases=[0.0, 16.0],
        key="8A",
        key_name="A minor",
        key_confidence=0.9,
        energy_curve=[0.5] * 120,
        sections=[],
        cue_points=[cue_out1, cue_out2],
        analyzed_at=1000.0,
    )

    track_in = TrackMeta(
        file_path="/tmp/track_in.wav",
        title="Track In",
        duration=120.0,
        bpm=124.0,
        bpm_confidence=0.9,
        beats=[0.0, 0.5],
        downbeats=[0.0, 2.0],
        phrases=[0.0, 16.0],
        key="8A",
        key_name="A minor",
        key_confidence=0.9,
        energy_curve=[0.5] * 120,
        sections=[],
        cue_points=[cue_in1, cue_in2],
        analyzed_at=1000.0,
    )

    best_out, best_in, score = find_best_cue_pair(track_out, track_in)
    # cue_out1 (v_out1) matches cue_in2 (v_in2)
    assert best_out.timestamp == 30.0
    assert best_in.timestamp == 20.0


def test_db_roundtrip_with_embeddings():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db, tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        db = TrackDB(tmp_db.name)
        v = [0.12345] * 512
        cue = CuePoint(15.0, "in", True, 0.7, 0.85, embedding=v)

        meta = TrackMeta(
            file_path=tmp_wav.name,
            title="DB Track Test",
            duration=60.0,
            bpm=128.0,
            bpm_confidence=0.95,
            beats=[0.0, 0.468],
            downbeats=[0.0, 1.875],
            phrases=[0.0, 15.0],
            key="9A",
            key_name="E minor",
            key_confidence=0.9,
            energy_curve=[0.6] * 60,
            sections=[],
            cue_points=[cue],
            analyzed_at=10000.0,
        )

        db.save(meta)
        loaded = db.load(tmp_wav.name)
        assert loaded is not None
        assert len(loaded.cue_points) == 1
        assert loaded.cue_points[0].embedding == v
        db.close()



if __name__ == "__main__":
    test_cue_point_embedding_serialization()
    test_cue_cosine_similarity()
    test_find_best_cue_pair()
    test_db_roundtrip_with_embeddings()
    print("All embedding unit tests passed successfully!")
