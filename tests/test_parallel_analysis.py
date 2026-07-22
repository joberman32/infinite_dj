"""
Unit tests for parallel multi-core library analysis.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from infinite_dj.analyzer import analyze_track
from dj import _analyze_file_worker
from infinite_dj.db import TrackDB


def generate_dummy_wav(path: str, duration_sec: float = 5.0, sr: int = 22050):
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    # Simple sine wave with periodic beats
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, y, sr)


def test_analyze_track_quiet():
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        generate_dummy_wav(tmp.name, duration_sec=4.0)
        meta = analyze_track(tmp.name, verbose=False)
        assert meta is not None
        assert meta.duration >= 3.9
        assert meta.bpm > 0


def test_analyze_file_worker():
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        generate_dummy_wav(tmp.name, duration_sec=4.0)
        fpath, meta, err, elapsed = _analyze_file_worker(tmp.name)
        assert err is None
        assert meta is not None
        assert elapsed > 0.0


def test_parallel_executor_batch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        files = []
        for i in range(3):
            p = os.path.join(tmp_dir, f"track_{i}.wav")
            generate_dummy_wav(p, duration_sec=3.0 + i)
            files.append(p)

        db_path = os.path.join(tmp_dir, "test.db")
        db = TrackDB(db_path)

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_analyze_file_worker, f) for f in files]
            for fut in as_completed(futures):
                fpath, meta, err, elapsed = fut.result()
                assert err is None
                assert meta is not None
                db.save(meta)

        stats = db.stats()
        assert stats["n"] == 3
        db.close()


if __name__ == "__main__":
    test_analyze_track_quiet()
    test_analyze_file_worker()
    test_parallel_executor_batch()
    print("All parallel analysis tests passed successfully!")
