"""Shared experiment configuration.

This module contains names and hyperparameters only. Environment dynamics live
in :mod:`online_sdft.environment`; learning algorithms live in
:mod:`online_sdft.methods`.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "bandit"
FIG = ROOT / "figures"

ACTIONS = ("INTERRUPT", "LATER", "ARCHIVE")
ACTION_CODES = ("A", "B", "C")
METHODS = (
    "Base",
    "ICL",
    "RAG",
    "REINFORCE",
    "Online-SFT",
    "Online-SDFT",
)
CATEGORIES = (
    "manager",
    "calendar",
    "monitoring",
    "teammate",
    "social",
    "receipt",
    "promo",
)
REGIMES = ("weekday", "on-call", "off-hours")

PHASE_LENGTH = 80
STREAM_LENGTH = PHASE_LENGTH * len(REGIMES)
FEATURE_DIM = len(CATEGORIES) + 8

EXPLORATION_EPSILON = 0.06
REPLAY_SIZE = 24
ONLINE_BATCH_SIZE = 4
ICL_K = 12
# Give retrieval the same prompt budget as the recency-only ICL baseline.
RAG_K = ICL_K

MODEL_ID = "LiquidAI/LFM2.5-230M"
LORA_R = 4
LORA_ALPHA = 8
SFT_LR = 2e-4
SDFT_LR = 3e-4
REINFORCE_LR = 1e-4
REINFORCE_BASELINE_STEP = 0.05
REINFORCE_ENTROPY_COEF = 0.01
TEACHER_TEMPERATURE = 0.95
STUDENT_TEMPERATURE = 1.0

SYSTEM_PROMPT = """You are an on-device notification router.
Choose exactly one route and reply with only its code:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE without a notification
Use the current notification and any past teacher examples. Do not add explanation."""
