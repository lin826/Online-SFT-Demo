"""LLM policy and the five compared online-learning methods.

Every method consumes :class:`StudentObservation`, which contains no
post-decision telemetry. Environment execution and rewards are intentionally
absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import (
    ACTION_CODES,
    ACTIONS,
    CATEGORIES,
    ICL_K,
    LORA_ALPHA,
    LORA_R,
    MODEL_ID,
    ONLINE_BATCH_SIZE,
    RAG_K,
    REPLAY_SIZE,
    SDFT_LR,
    SFT_LR,
    STUDENT_TEMPERATURE,
    SYSTEM_PROMPT,
)
from .environment import StudentObservation, one_hot


class StudentPolicy(Protocol):
    """Minimal policy interface shared by the real LFM and test doubles."""

    def start_run(self, learning_rate: float | None) -> None: ...

    def probs(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> np.ndarray: ...

    def update(self, batch: list[tuple[str, np.ndarray]]) -> float: ...


class LiquidLLMPolicy:
    """LFM2.5 student whose A/B/C next-token logits define route scores."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        # T4-class Colab GPUs are optimized for FP16 and do not accelerate
        # BF16. Keep CPU/MPS in FP32 so the published CPU protocol is
        # unchanged, while CUDA runs use tensor cores for practical reruns.
        self.model_dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            dtype=self.model_dtype,
        )
        adapter_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            target_modules=r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$",
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, adapter_config).to(self.device)
        self.model.config.use_cache = False

        self.action_token_ids = []
        for code in ACTION_CODES:
            token_ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"action code {code!r} is not one token: {token_ids}"
                )
            self.action_token_ids.append(token_ids[0])

        self._initial_adapter = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.optimizer: Any | None = None

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def start_run(self, learning_rate: float | None) -> None:
        """Reset LoRA so every method and seed starts identically."""
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(self._initial_adapter[name].to(self.device))
        self.optimizer = None
        if learning_rate is not None:
            self.optimizer = self.torch.optim.AdamW(
                (
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
                lr=learning_rate,
            )

    def render_prompt(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> str:
        lines = []
        for index, row in enumerate(examples or [], start=1):
            code = ACTION_CODES[row["teacher_action"]]
            # Keep the demonstration and query schemas identical. This makes
            # the frozen policy's job genuine few-shot classification rather
            # than asking it to translate `teacher=A` into `route: A`.
            lines.append(
                f"example {index} notification: {row['context']}"
            )
            lines.append(f"example {index} route: {code}")
        if lines:
            lines.append(
                "Treat these as user-specific evidence, not universal rules."
            )
        lines.append(f"current notification: {context}")
        lines.append("current route:")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _action_logits(self, prompts: list[str]):
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        # Compute normalization and the distillation loss in FP32 even when
        # the frozen CUDA backbone runs in FP16.
        logits = self.model(**encoded).logits[:, -1, :].float()
        action_ids = self.torch.tensor(
            self.action_token_ids,
            device=self.device,
        )
        return logits.index_select(-1, action_ids)

    def probs(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> np.ndarray:
        self.model.eval()
        with self.torch.no_grad():
            logits = self._action_logits([self.render_prompt(context, examples)])
            probabilities = self.torch.softmax(
                logits / STUDENT_TEMPERATURE,
                dim=-1,
            )[0]
        return probabilities.float().cpu().numpy()

    def update(self, batch: list[tuple[str, np.ndarray]]) -> float:
        """Apply one soft-target cross-entropy LoRA update."""
        if self.optimizer is None:
            raise RuntimeError(
                "start_run must receive a learning rate before update"
            )
        self.model.train()
        prompts = [self.render_prompt(context) for context, _ in batch]
        targets = self.torch.tensor(
            np.stack([target for _, target in batch]),
            device=self.device,
            dtype=self.torch.float32,
        )
        logits = self._action_logits(prompts)
        loss = -(
            targets * self.torch.log_softmax(logits, dim=-1)
        ).sum(-1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            max_norm=1.0,
        )
        self.optimizer.step()
        return float(loss.detach().cpu())


@dataclass
class TeacherRecord:
    """One causal record retained after a method has already acted."""

    observation: StudentObservation
    teacher_action: int
    feedback: dict

    def prompt_example(self) -> dict:
        return {
            "context": self.observation.text,
            "teacher_action": self.teacher_action,
        }


def mixed_context_similarity(
    query: StudentObservation,
    candidate: StudentObservation,
) -> float:
    """Gower-style similarity over the six student-visible fields.

    Category and regime use exact match; importance, deadline, and affinity
    use bounded absolute distance; hour uses circular distance. Equal weights
    avoid an opaque, benchmark-tuned metric and the constant/derived entries
    in ``StudentObservation.features`` are deliberately ignored.
    """
    category_count = len(CATEGORIES)
    query_features = query.features
    candidate_features = candidate.features

    category_similarity = float(
        np.argmax(query_features[:category_count])
        == np.argmax(candidate_features[:category_count])
    )
    continuous_similarities = [
        1.0 - abs(float(query_features[index] - candidate_features[index]))
        for index in range(category_count, category_count + 3)
    ]

    query_clock = query_features[category_count + 3 : category_count + 5]
    candidate_clock = candidate_features[
        category_count + 3 : category_count + 5
    ]
    clock_cosine = float(
        np.clip(np.dot(query_clock, candidate_clock), -1.0, 1.0)
    )
    hour_similarity = 1.0 - float(np.arccos(clock_cosine) / np.pi)

    query_regime = int(round(2 * query_features[category_count + 5]))
    candidate_regime = int(
        round(2 * candidate_features[category_count + 5])
    )
    regime_similarity = float(query_regime == candidate_regime)
    components = [
        category_similarity,
        *continuous_similarities,
        hour_similarity,
        regime_similarity,
    ]
    return float(np.mean(components))


class OnlineAgent:
    """Base class for one method on one chronological stream."""

    name = "Base"
    learning_rate: float | None = None

    def __init__(self, policy: StudentPolicy):
        self.policy = policy
        self.memory: list[TeacherRecord] = []
        self.replay: list[tuple[str, np.ndarray]] = []
        self.policy.start_run(self.learning_rate)

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        del observation
        return []

    def action_probs(self, observation: StudentObservation) -> np.ndarray:
        return self.policy.probs(
            observation.text,
            self.prompt_examples(observation),
        )

    def training_target(
        self,
        teacher_distribution: np.ndarray,
        teacher_action: int,
    ) -> np.ndarray | None:
        del teacher_distribution, teacher_action
        return None

    def observe(
        self,
        observation: StudentObservation,
        teacher_distribution: np.ndarray,
        teacher_action: int,
        feedback: dict,
        rng: np.random.Generator,
    ) -> None:
        """Retain legal history and, for trainable agents, update online."""
        self.memory.append(
            TeacherRecord(
                observation=observation,
                teacher_action=teacher_action,
                feedback=feedback,
            )
        )
        target = self.training_target(
            teacher_distribution,
            teacher_action,
        )
        if target is None:
            return

        self.replay.append((observation.text, target))
        self.replay = self.replay[-REPLAY_SIZE:]
        indices = [len(self.replay) - 1]
        if len(self.replay) > 1:
            indices += rng.choice(
                len(self.replay) - 1,
                size=min(ONLINE_BATCH_SIZE - 1, len(self.replay) - 1),
                replace=False,
            ).tolist()
        self.policy.update([self.replay[index] for index in indices])


class BaseAgent(OnlineAgent):
    """Frozen LFM without external memory."""


class ICLAgent(OnlineAgent):
    """Frozen LFM prompted with the latest causal teacher demonstrations."""

    name = "ICL"

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        del observation
        return [
            record.prompt_example()
            for record in self.memory[-ICL_K:]
        ]


class RAGAgent(OnlineAgent):
    """Frozen LFM prompted with nearest causal teacher demonstrations."""

    name = "RAG"

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        if not self.memory:
            return []
        similarities = [
            (
                mixed_context_similarity(
                    observation,
                    record.observation,
                ),
                index,
                record,
            )
            for index, record in enumerate(self.memory)
        ]
        closest = sorted(
            similarities,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[:RAG_K]
        # Put the best match closest to the current query. For equal scores,
        # the newest record is also closest to the query in the prompt.
        closest.sort(key=lambda item: (item[0], item[1]))
        return [record.prompt_example() for _, _, record in closest]


class OnlineSFTAgent(OnlineAgent):
    """Online LoRA update from one sampled hard teacher rollout."""

    name = "Online-SFT"
    learning_rate = SFT_LR

    def training_target(
        self,
        teacher_distribution: np.ndarray,
        teacher_action: int,
    ) -> np.ndarray:
        del teacher_distribution
        return one_hot(teacher_action, len(ACTIONS))


class OnlineSDFTAgent(OnlineAgent):
    """Online LoRA update from the full soft teacher distribution."""

    name = "Online-SDFT"
    learning_rate = SDFT_LR

    def training_target(
        self,
        teacher_distribution: np.ndarray,
        teacher_action: int,
    ) -> np.ndarray:
        del teacher_action
        return teacher_distribution.copy()


AGENT_CLASSES = {
    agent_class.name: agent_class
    for agent_class in (
        BaseAgent,
        ICLAgent,
        RAGAgent,
        OnlineSFTAgent,
        OnlineSDFTAgent,
    )
}


def create_agent(method: str, policy: StudentPolicy) -> OnlineAgent:
    """Construct one named method or fail loudly on an invalid benchmark arm."""
    try:
        agent_class = AGENT_CLASSES[method]
    except KeyError as error:
        raise ValueError(f"unknown method {method!r}") from error
    return agent_class(policy)
