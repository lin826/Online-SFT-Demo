"""Compatibility facade for the structured online-SDFT package.

New code should import the focused modules directly:

- online_sdft.environment: simulator, rewards, and teacher
- online_sdft.methods: Liquid LFM policy and compared algorithms
- online_sdft.experiment: chronological interaction loop
- online_sdft.reporting: aggregation and figures
"""

from online_sdft.config import (
    ACTION_CODES,
    ACTIONS,
    CATEGORIES,
    EXPLORATION_EPSILON,
    FEATURE_DIM,
    FIG,
    ICL_K,
    LORA_ALPHA,
    LORA_R,
    METHODS,
    MODEL_ID,
    ONLINE_BATCH_SIZE,
    OUT,
    PHASE_LENGTH,
    RAG_K,
    REGIMES,
    REPLAY_SIZE,
    SDFT_LR,
    SFT_LR,
    STREAM_LENGTH,
    STUDENT_TEMPERATURE,
    TEACHER_TEMPERATURE,
)
from online_sdft.environment import (
    DEFAULT_ENVIRONMENT,
    Event,
    NotificationRoutingEnvironment,
    StudentObservation,
    category_profile,
    context_text,
    factual_feedback,
    make_event,
    make_stream,
    one_hot,
    oracle_utilities,
    softmax,
    student_observation,
    teacher_policy,
)
from online_sdft.experiment import epsilon_greedy, main, run_method
from online_sdft.methods import (
    AGENT_CLASSES,
    BaseAgent,
    ICLAgent,
    LiquidLLMPolicy,
    OnlineAgent,
    OnlineSDFTAgent,
    OnlineSFTAgent,
    RAGAgent,
    StudentPolicy,
    TeacherRecord,
    create_agent,
)
from online_sdft.reporting import (
    find_qualitative_examples,
    mean_ci,
    summarize_metrics,
    write_figures,
)

__all__ = [name for name in globals() if not name.startswith("_")]
