"""
CLAP Neural Audio Embedding Extractor.

Uses HuggingFace's ClapModel (laion/clap-htsat-fused) to extract 512D latent audio
embeddings for the audio windows surrounding IN and OUT cue points.
"""

import sys
import numpy as np
from typing import List, Optional
import librosa

# Target sample rate expected by CLAP models
CLAP_TARGET_SR = 48000


class CLAPExtractor:
    _instance = None

    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_error = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False

        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            print("  [CLAP] Loading neural embedding model (laion/clap-htsat-fused)...", flush=True)
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model = ClapModel.from_pretrained(self.model_name)
            self.model.eval()
            self._loaded = True
            print("  [CLAP] Model loaded successfully.", flush=True)
            return True
        except Exception as e:
            self._load_error = str(e)
            print(f"  [CLAP Warning] Could not load CLAP model: {e}\n"
                  f"  [CLAP Warning] Embeddings are optional — install with: "
                  f"pip install -r requirements-clap.txt", file=sys.stderr)
            return False

    def extract_embedding(
        self,
        y: np.ndarray,
        sr: int,
        timestamp: float,
        cue_type: str,
        window_sec: float = 8.0,
    ) -> Optional[List[float]]:
        """
        Extract a 512-dimensional CLAP embedding vector for a cue point.

        - OUT cue: Extracts audio window [timestamp - window_sec, timestamp]
        - IN cue: Extracts audio window [timestamp, timestamp + window_sec]
        """
        if not self.load_model():
            return None

        duration = len(y) / sr
        if cue_type == "out":
            start_t = max(0.0, timestamp - window_sec)
            end_t   = min(duration, timestamp)
        else:  # "in" or default
            start_t = max(0.0, timestamp)
            end_t   = min(duration, timestamp + window_sec)

        start_sample = int(round(start_t * sr))
        end_sample   = int(round(end_t * sr))

        segment = y[start_sample:end_sample]
        if len(segment) == 0:
            return None

        # Convert stereo to mono if needed
        if segment.ndim > 1:
            segment = segment.mean(axis=1 if segment.shape[1] == 2 else 0)

        # Resample to 48kHz for CLAP
        if sr != CLAP_TARGET_SR:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=CLAP_TARGET_SR)

        # Pad to at least 1 second if very short
        min_samples = CLAP_TARGET_SR
        if len(segment) < min_samples:
            segment = np.pad(segment, (0, min_samples - len(segment)))

        try:
            import torch
            inputs = self.processor(
                audios=segment,
                sampling_rate=CLAP_TARGET_SR,
                return_tensors="pt"
            )
            with torch.no_grad():
                audio_embeds = self.model.get_audio_features(**inputs)
                # L2 normalize
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
                vector = audio_embeds[0].cpu().numpy().tolist()
                return [round(float(v), 5) for v in vector]
        except Exception as e:
            print(f"  [CLAP Error] Embedding extraction failed at t={timestamp:.1f}s: {e}", file=sys.stderr)
            return None


def get_cue_embedding(
    y: np.ndarray,
    sr: int,
    timestamp: float,
    cue_type: str,
    window_sec: float = 8.0
) -> Optional[List[float]]:
    """Helper function to extract CLAP embedding for a cue point."""
    extractor = CLAPExtractor.get_instance()
    return extractor.extract_embedding(y, sr, timestamp, cue_type, window_sec)
