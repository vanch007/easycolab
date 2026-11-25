"""EasyColab Cinema-Grade Short-Drama Generation System (drama_studio).

Synthesizes the cinematography & lint rules of `manju-laoli-skill` with the
mechanical reference compilation, FFmpeg signal detection, contact-sheet VLM review,
and non-destructive timeline architecture of `ArcReel`, powered end-to-end by
EasyColab's GPU model stack (Qwen3.8-27B-Uncensored-VL, MOSS-Image, Qwen-Image-Edit-2511,
Qwen3-TTS 1.7B, Fish Audio Drama 3 35B, Qwen3-ASR 1.7B, and MiniMax-H3 10Eros-Max).
"""

from .models import (
    CharacterSpec,
    EmotionBeat,
    EpisodeBible,
    LocationSpec,
    ProductionConfig,
    PropSpec,
    ShotSpec,
)
from .pipeline import CinemaDramaPipeline, run_drama_production

__all__ = [
    "CharacterSpec",
    "CinemaDramaPipeline",
    "EmotionBeat",
    "EpisodeBible",
    "LocationSpec",
    "ProductionConfig",
    "PropSpec",
    "ShotSpec",
    "run_drama_production",
]
