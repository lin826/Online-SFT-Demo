"""Build the fully self-contained Online-SDFT demonstration notebook and GIF."""

from __future__ import annotations

import base64
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "online_sdft_bandit_demo.ipynb"
GIF_PATH = ROOT / "figures" / "online_sdft_process.gif"


def embedded_core() -> str:
    """Embed focused modules without any repository imports at runtime."""

    def module_source(
        relative_path: str,
        end_marker: str | None = None,
    ) -> str:
        source = (ROOT / relative_path).read_text()
        if end_marker is not None:
            source = source[:source.index(end_marker)]
        lines = []
        skipping_relative_import = False
        for line in source.splitlines():
            if skipping_relative_import:
                if ")" in line:
                    skipping_relative_import = False
                continue
            if line == "from __future__ import annotations":
                continue
            if line == "from pathlib import Path":
                continue
            if line.startswith("from ."):
                if "(" in line and ")" not in line:
                    skipping_relative_import = True
                continue
            if line.startswith(("ROOT = ", "OUT = ", "FIG = ")):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    modules = [
        module_source("online_sdft/config.py"),
        module_source("online_sdft/environment.py"),
        module_source("online_sdft/methods.py"),
        module_source(
            "online_sdft/experiment.py",
            end_marker="\ndef experiment_config",
        ),
        module_source(
            "online_sdft/reporting.py",
            end_marker="\ndef summarize_metrics",
        ),
    ]
    return "\n\n".join(modules) + "\n"


GIF_FUNCTIONS = r'''
from io import BytesIO
from PIL import Image as PILImage, ImageDraw, ImageFont
from matplotlib import font_manager
import matplotlib.pyplot as plt

GIF_WIDTH, GIF_HEIGHT = 1200, 675
PALETTE = {
    "bg": "#F8FAFC",
    "navy": "#14233B",
    "blue": "#4F6BED",
    "teal": "#0D9488",
    "coral": "#E76F51",
    "slate": "#526277",
    "line": "#D7E0EA",
    "pale_blue": "#EEF2FF",
    "pale_teal": "#E7F6F3",
    "pale_coral": "#FCEEEA",
    "white": "#FFFFFF",
}


def _font(size, bold=False):
    weight = "bold" if bold else "normal"
    path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight=weight)
    )
    return ImageFont.truetype(path, size=size)


FONTS = {
    "title": _font(34, True),
    "heading": _font(17, True),
    "body": _font(16),
    "body_bold": _font(16, True),
    "small": _font(13),
    "status": _font(20, True),
}


def _rgb(hex_color):
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _blend(foreground, background, alpha):
    fg, bg = _rgb(foreground), _rgb(background)
    return tuple(round(alpha * f + (1 - alpha) * b) for f, b in zip(fg, bg))


def _center(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _arrow(draw, start, end, fill, width=4):
    draw.line((start, end), fill=fill, width=width)
    x, y = end
    direction = 1 if end[0] >= start[0] else -1
    draw.polygon(
        [(x, y), (x - 12 * direction, y - 7), (x - 12 * direction, y + 7)],
        fill=fill,
    )


def _pill(draw, box, text, fill, outline, text_fill, font=None):
    font = font or FONTS["body_bold"]
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline, width=2)
    _center(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font, text_fill)


def _card(draw, box, title, intensity, active=False):
    outline = PALETTE["teal"] if active else _blend(PALETTE["navy"], PALETTE["bg"], intensity)
    fill = _blend(PALETTE["white"], PALETTE["bg"], max(0.55, intensity))
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=4 if active else 2)
    _center(draw, ((box[0] + box[2]) / 2, box[1] + 34), title,
            FONTS["heading"], _blend(PALETTE["navy"], PALETTE["bg"], intensity))


def _draw_context(draw, box, intensity):
    color = _blend(PALETTE["blue"], PALETTE["bg"], intensity)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1 + 55, y1 + 70, x2 - 55, y1 + 155), radius=14,
                           fill=_blend(PALETTE["pale_blue"], PALETTE["bg"], intensity),
                           outline=color, width=2)
    draw.ellipse((x1 + 75, y1 + 91, x1 + 105, y1 + 121), fill=color)
    draw.line((x1 + 115, y1 + 95, x2 - 75, y1 + 95), fill=color, width=5)
    draw.line((x1 + 115, y1 + 116, x2 - 100, y1 + 116), fill=color, width=4)
    labels = ["importance  0.88", "deadline     0.94", "time        15:00"]
    for index, label in enumerate(labels):
        y = y1 + 185 + index * 42
        _pill(draw, (x1 + 24, y, x2 - 24, y + 30), label,
              _blend(PALETTE["white"], PALETTE["bg"], intensity),
              _blend(PALETTE["line"], PALETTE["bg"], intensity),
              _blend(PALETTE["slate"], PALETTE["bg"], intensity), FONTS["small"])
    _center(draw, ((x1 + x2) / 2, y1 + 330), "student-visible x_t",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))


def _draw_student(draw, box, intensity):
    x1, y1, x2, _ = box
    labels = [("INTERRUPT", 0.18), ("LATER", 0.67), ("ARCHIVE", 0.15)]
    for index, (label, probability) in enumerate(labels):
        y = y1 + 82 + index * 70
        selected = label == "LATER"
        fill = PALETTE["pale_teal"] if selected else PALETTE["white"]
        outline = PALETTE["teal"] if selected else PALETTE["line"]
        _pill(draw, (x1 + 20, y, x2 - 20, y + 48), label,
              _blend(fill, PALETTE["bg"], intensity),
              _blend(outline, PALETTE["bg"], intensity),
              _blend(PALETTE["navy"], PALETTE["bg"], intensity))
        draw.rectangle((x1 + 30, y + 55, x1 + 30 + 145 * probability, y + 61),
                       fill=_blend(PALETTE["blue"], PALETTE["bg"], intensity))
        draw.rectangle((x1 + 30 + 145 * probability, y + 55, x2 - 30, y + 61),
                       fill=_blend(PALETTE["line"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 330), "no feedback · no oracle",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))


def _draw_commit(draw, box, intensity):
    x1, y1, x2, _ = box
    _pill(draw, (x1 + 35, y1 + 80, x2 - 35, y1 + 132), "LATER",
          _blend(PALETTE["pale_teal"], PALETTE["bg"], intensity),
          _blend(PALETTE["teal"], PALETTE["bg"], intensity),
          _blend(PALETTE["teal"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 160), "action is frozen",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))
    for index, label in enumerate(("INTERRUPT  LOCKED", "ARCHIVE    LOCKED")):
        y = y1 + 190 + index * 58
        _pill(draw, (x1 + 22, y, x2 - 22, y + 40), label,
              _blend(PALETTE["pale_coral"], PALETTE["bg"], intensity),
              _blend(PALETTE["coral"], PALETTE["bg"], intensity),
              _blend(PALETTE["coral"], PALETTE["bg"], intensity), FONTS["small"])
    _center(draw, ((x1 + x2) / 2, y1 + 322), "score r_t before feedback",
            FONTS["small"], _blend(PALETTE["navy"], PALETTE["bg"], intensity))


def _draw_feedback(draw, box, intensity):
    x1, y1, x2, _ = box
    teal = _blend(PALETTE["teal"], PALETTE["bg"], intensity)
    draw.ellipse((x1 + 70, y1 + 80, x2 - 70, y1 + 160),
                 fill=_blend(PALETTE["pale_teal"], PALETTE["bg"], intensity),
                 outline=teal, width=3)
    draw.line((x1 + 91, y1 + 121, x1 + 109, y1 + 139), fill=teal, width=5)
    draw.line((x1 + 109, y1 + 139, x1 + 141, y1 + 102), fill=teal, width=5)
    _center(draw, ((x1 + x2) / 2, y1 + 195), "OPENED_DIGEST",
            FONTS["body_bold"], teal)
    _pill(draw, (x1 + 22, y1 + 228, x2 - 22, y1 + 270), "one factual outcome",
          _blend(PALETTE["white"], PALETTE["bg"], intensity),
          _blend(PALETTE["line"], PALETTE["bg"], intensity),
          _blend(PALETTE["navy"], PALETTE["bg"], intensity), FONTS["small"])
    _center(draw, ((x1 + x2) / 2, y1 + 315), "unchosen = unknown",
            FONTS["small"], _blend(PALETTE["coral"], PALETTE["bg"], intensity))


def _draw_teacher(draw, box, intensity):
    x1, y1, x2, _ = box
    values = [("I", 0.12), ("L", 0.76), ("A", 0.12)]
    for index, (label, value) in enumerate(values):
        y = y1 + 82 + index * 44
        draw.text((x1 + 25, y), label, font=FONTS["small"],
                  fill=_blend(PALETTE["navy"], PALETTE["bg"], intensity))
        draw.rounded_rectangle((x1 + 50, y, x2 - 25, y + 18), radius=8,
                               fill=_blend(PALETTE["line"], PALETTE["bg"], intensity))
        draw.rounded_rectangle((x1 + 50, y, x1 + 50 + 135 * value, y + 18), radius=8,
                               fill=_blend(PALETTE["teal"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 225), "soft teacher q_t",
            FONTS["small"], _blend(PALETTE["navy"], PALETTE["bg"], intensity))
    for index in range(4):
        left = x1 + 33 + index * 40
        fill = PALETTE["teal"] if index == 0 else PALETTE["blue"]
        draw.rounded_rectangle((left, y1 + 255, left + 28, y1 + 285), radius=6,
                               fill=_blend(fill, PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 310), "fresh + up to 3 recent",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 335), "updates t+1 only",
            FONTS["small"], _blend(PALETTE["teal"], PALETTE["bg"], intensity))


def _draw_frame(progress):
    image = PILImage.new("RGB", (GIF_WIDTH, GIF_HEIGHT), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    for x in range(0, GIF_WIDTH, 40):
        draw.line((x, 0, x, GIF_HEIGHT), fill="#F0F3F7", width=1)
    for y in range(0, GIF_HEIGHT, 40):
        draw.line((0, y, GIF_WIDTH, y), fill="#F0F3F7", width=1)

    _center(draw, (GIF_WIDTH / 2, 48), "ONLINE SDFT · ONE CAUSAL ROUND",
            FONTS["title"], PALETTE["navy"])
    _center(draw, (GIF_WIDTH / 2, 87), "predict → commit → observe one world → learn for the next round",
            FONTS["body"], PALETTE["slate"])

    left, gap, card_width = 24, 18, 216
    cards = [(left + index * (card_width + gap), 160,
              left + index * (card_width + gap) + card_width, 535)
             for index in range(5)]
    centers = [((box[0] + box[2]) / 2, 128) for box in cards]
    for index in range(4):
        _arrow(draw, (centers[index][0] + 18, 128),
               (centers[index + 1][0] - 18, 128), PALETTE["line"], 3)

    current = min(4, int(progress))
    for index, (cx, cy) in enumerate(centers):
        fill = PALETTE["teal"] if index <= current else PALETTE["line"]
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), fill=fill,
                     outline=PALETTE["white"], width=3)

    if progress < 4:
        start = centers[int(progress)][0]
        finish = centers[int(progress) + 1][0]
        fraction = progress - int(progress)
        dot_x = start + (finish - start) * fraction
    else:
        dot_x = centers[-1][0]
    draw.ellipse((dot_x - 7, 121, dot_x + 7, 135), fill=PALETTE["blue"])

    titles = ("1 CONTEXT", "2 LIQUID LFM", "3 COMMIT + SCORE",
              "4 FEEDBACK", "5 TEACH + UPDATE")
    renderers = (_draw_context, _draw_student, _draw_commit, _draw_feedback, _draw_teacher)
    for index, box in enumerate(cards):
        if index < current:
            intensity = 0.78
        elif index == current:
            intensity = 1.0
        else:
            intensity = 0.32
        _card(draw, box, titles[index], intensity, active=index == current)
        renderers[index](draw, box, intensity)

    if current == 4:
        _center(draw, (GIF_WIDTH / 2, 551), "Updated policy faces the next context",
                FONTS["small"], PALETTE["teal"])
        _arrow(draw, (cards[4][2] - 15, 574), (cards[0][0] + 15, 574), PALETTE["teal"], 4)

    status = (
        "Context x_t arrives; no current feedback exists.",
        "Liquid LFM selects its own action from π_t(.|x_t).",
        "The selected action and regret are frozen before feedback.",
        "Only the selected action produces factual feedback.",
        "The teacher's soft target updates a tiny batch for t+1.",
    )[current]
    draw.rounded_rectangle((90, 598, 1110, 650), radius=18,
                           fill=PALETTE["navy"], outline=PALETTE["navy"])
    _center(draw, (600, 624), status, FONTS["status"], PALETTE["white"])
    return image


def make_online_sdft_gif():
    frames = []
    for stage in range(5):
        for substep in range(7):
            progress = stage + substep / 7 if stage < 4 else 4
            frames.append(_draw_frame(progress))
        pause_progress = stage + 0.99 if stage < 4 else 4.0
        frames.extend([_draw_frame(pause_progress)] * 2)
    buffer = BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True,
                   append_images=frames[1:], duration=135, loop=0,
                   optimize=True, disposal=2)
    return buffer.getvalue()
'''.strip()


GAME_ENGINE = r'''
import random
from IPython.display import Markdown, display

GAME_ACTIONS = ("INTERRUPT", "LATER", "ARCHIVE")
GAME_SCENARIOS = (
    {
        "name": "Afternoon calendar alert",
        "category": "calendar", "time": "15:00",
        "importance": 0.88, "deadline": 0.94, "affinity": 0.45,
        "busy": 0.88, "incident": 0.0, "manager": 0.0, "social": 0.0,
        "seed": 7,
    },
    {
        "name": "On-call monitoring incident",
        "category": "monitoring", "time": "02:10",
        "importance": 0.82, "deadline": 0.90, "affinity": 0.25,
        "busy": 0.30, "incident": 1.0, "manager": 0.0, "social": 0.0,
        "seed": 11,
    },
    {
        "name": "Off-hours message from a close friend",
        "category": "social", "time": "20:30",
        "importance": 0.22, "deadline": 0.08, "affinity": 0.85,
        "busy": 0.15, "incident": 0.0, "manager": 0.0, "social": 1.0,
        "seed": 19,
    },
    {
        "name": "Late-morning promotion",
        "category": "promo", "time": "11:20",
        "importance": 0.08, "deadline": 0.03, "affinity": 0.15,
        "busy": 0.55, "incident": 0.0, "manager": 0.0, "social": 0.0,
        "seed": 23,
    },
)


def _game_utilities(scenario):
    importance = scenario["importance"]
    affinity = scenario["affinity"]
    urgency = importance * scenario["deadline"]
    interrupt = (1.45 * urgency + 0.42 * affinity - 1.20 * scenario["busy"]
                 + 1.00 * scenario["incident"] + 0.60 * scenario["manager"]
                 + 0.50 * scenario["social"])
    later = (0.72 * importance + 0.58 * affinity - 0.62 * urgency
             + 0.22 * scenario["busy"] - 0.62 * scenario["incident"])
    archive = (0.72 * (1 - importance) + 0.36 * (1 - affinity)
               - 0.80 * urgency - 0.50 * scenario["social"])
    return (interrupt, later, archive)


def _execute_game_action(scenario, action_index, selected_utility):
    rng = random.Random(scenario["seed"])
    draw = rng.random()
    engage_p = max(0.04, min(0.92, 0.36 + 0.24 * selected_utility))
    if action_index == 0:
        outcome = ("OPENED_PUSH" if draw < engage_p else
                   "DISMISSED_PUSH" if draw < engage_p + 0.45 else "IGNORED_PUSH")
        reward = {"OPENED_PUSH": 0.72, "DISMISSED_PUSH": -0.58,
                  "IGNORED_PUSH": -0.78}[outcome] - 0.30 * scenario["busy"]
    elif action_index == 1:
        outcome = "OPENED_DIGEST" if draw < engage_p else "IGNORED_DIGEST"
        reward = {"OPENED_DIGEST": 0.48, "IGNORED_DIGEST": -0.16}[outcome]
        reward -= 0.28 * scenario["importance"] * scenario["deadline"]
    else:
        organic_p = max(0.04, min(0.30, 0.08 + 0.15 * scenario["affinity"]))
        outcome = "ORGANIC_INBOX_OPEN" if draw < organic_p else "NO_OBSERVATION"
        reward = 0.16 if outcome == "ORGANIC_INBOX_OPEN" else 0.0
    return outcome, reward


def play_notification_round(scenario_id, action):
    if not 0 <= scenario_id < len(GAME_SCENARIOS):
        raise ValueError(f"scenario_id must be 0–{len(GAME_SCENARIOS) - 1}")
    action = action.upper()
    if action not in GAME_ACTIONS:
        raise ValueError(f"action must be one of {GAME_ACTIONS}")

    scenario = GAME_SCENARIOS[scenario_id]
    action_index = GAME_ACTIONS.index(action)
    utilities = _game_utilities(scenario)
    best_index = max(range(len(GAME_ACTIONS)), key=lambda index: utilities[index])
    outcome, reward = _execute_game_action(
        scenario, action_index, utilities[action_index]
    )
    regret = utilities[best_index] - utilities[action_index]

    display(Markdown(f"""
### Your round: {scenario['name']}

| Visible before acting | Value |
| --- | --- |
| Category / time | `{scenario['category']}` / {scenario['time']} |
| Importance | {scenario['importance']:.2f} |
| Deadline pressure | {scenario['deadline']:.2f} |
| Personal affinity | {scenario['affinity']:.2f} |

**You committed to:** `{action}`

**One factual outcome:** `{outcome}` · teacher reward `{reward:+.3f}`

#### Debrief — evaluator only

Current busyness was `{scenario['busy']:.2f}`. The scoring-best route was
`{GAME_ACTIONS[best_index]}`, so this action's regret was `{regret:.3f}` utility units.

> The learner receives the factual outcome and reward, not the hidden busyness,
> best route, or utilities shown in this debrief.
"""))
'''.strip()


GAME_PLAY = r'''
# Change these two values, then run this cell.
SCENARIO_ID = 0
MY_ACTION = "INTERRUPT"  # INTERRUPT, LATER, or ARCHIVE

play_notification_round(SCENARIO_ID, MY_ACTION)
'''.strip()


INSTALL_DEPS = r'''
import importlib.metadata as package_metadata
import importlib.util
import os
import subprocess
import sys


# Colab currently preinstalls torchao 0.10.0 alongside a much newer PEFT.
# TorchAO is optional for this unquantized LoRA experiment, and PEFT rejects
# that stale version at import time. Removing it is safer than replacing
# Colab's matched PyTorch/CUDA build with a different TorchAO/Torch pair.
try:
    torchao_version = package_metadata.version("torchao")
except package_metadata.PackageNotFoundError:
    torchao_version = None

if torchao_version is not None:
    print(f"Removing unused torchao {torchao_version} to avoid a PEFT conflict...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
    )

packages = [
    "transformers==5.13.1",
    "peft==0.19.1",
    "numpy>=1.26,<3",
    "matplotlib>=3.8,<4",
    "Pillow>=10,<13",
]
if importlib.util.find_spec("torch") is None:
    # Local Jupyter fallback. Colab already supplies a CUDA-matched PyTorch.
    packages.append("torch>=2.3")

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", *packages]
)

import peft
import torch
import transformers

in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
if in_colab and not torch.cuda.is_available():
    raise RuntimeError(
        "Colab is using CPU. Choose Runtime > Change runtime type > T4 GPU, "
        "then run this cell again."
    )

device_name = (
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/MPS"
)
print(
    "Runtime ready | "
    f"torch={torch.__version__} | transformers={transformers.__version__} | "
    f"peft={peft.__version__} | device={device_name}"
)
'''.strip()


RUNNER = r'''
from io import StringIO
import torch


def run_experiment_in_memory(seeds=3):
    curve_fields = ["seed", "method", "t", "phase", "regime", "step_correct",
                    "step_regret", "cum_accuracy", "cum_regret"]
    metrics, rollouts, curves = [], [], []

    for seed in range(seeds):
        policy = LiquidLLMPolicy(model_id=MODEL_ID)
        if seed == 0:
            hardware = (
                torch.cuda.get_device_name(0)
                if policy.device.type == "cuda"
                else str(policy.device)
            )
            print(f"Loaded {policy.model_id} on {hardware}; "
                  f"{policy.trainable_parameters:,} trainable LoRA parameters",
                  flush=True)
        stream = make_stream(seed)
        for method in METHODS:
            rollout_buffer = StringIO()
            curve_buffer = StringIO()
            curve_writer = csv.DictWriter(curve_buffer, fieldnames=curve_fields,
                                          lineterminator="\n")
            curve_writer.writeheader()
            metric_row = run_method(seed, method, stream, policy,
                                    rollout_buffer, curve_writer)
            metrics.append(metric_row)
            print(
                f"seed {seed + 1}/{seeds} | {method:<11} | "
                f"accuracy={metric_row['online_accuracy']:.3f} | "
                f"regret={metric_row['cum_regret']:.2f}",
                flush=True,
            )
            rollout_buffer.seek(0)
            rollouts.extend(json.loads(line) for line in rollout_buffer)
            curve_buffer.seek(0)
            curves.extend(csv.DictReader(curve_buffer))
        resolved_device = str(policy.device)
        policy.optimizer = None
        del policy
        gc.collect()
        if resolved_device == "mps":
            torch.mps.empty_cache()
        elif resolved_device == "cuda":
            torch.cuda.empty_cache()

    metric_names = [key for key in metrics[0] if key not in {"seed", "method"}]
    summary = {}
    for method in METHODS:
        rows = [row for row in metrics if row["method"] == method]
        summary[method] = {
            metric: mean_ci([float(row[metric]) for row in rows])
            for metric in metric_names
        }

    grouped = defaultdict(dict)
    for row in rollouts:
        grouped[(row["seed"], row["t"])][row["method"]] = row
    qualitative = []
    for (seed, t), rows in sorted(grouped.items()):
        if len(rows) != len(METHODS) or t <= PHASE_LENGTH:
            continue
        oracle = rows["Online-SDFT"]["oracle_action_scoring_only"]
        if (rows["Online-SDFT"]["action"] == oracle
                and all(rows[name]["action"] != oracle
                        for name in METHODS if name != "Online-SDFT")):
            qualitative.append({"seed": seed, "t": t, "oracle": oracle,
                                "regime": rows["Online-SDFT"]["regime"],
                                "category": rows["Online-SDFT"]["category"],
                                "methods": rows})
        if len(qualitative) == 8:
            break
    return metrics, rollouts, curves, summary, qualitative


N_SEEDS = 3
metrics, rollouts, curves, summary, qualitative = run_experiment_in_memory(N_SEEDS)
print(f"Finished {N_SEEDS} paired streams × {len(METHODS)} methods × {STREAM_LENGTH} decisions")
'''.strip()


RESULTS = r'''
from IPython.display import Markdown, display

header = "| Method | Online accuracy | 95% CI | Cumulative regret | 95% CI |\n| --- | ---: | ---: | ---: | ---: |"
rows = []
for method in METHODS:
    accuracy = summary[method]["online_accuracy"]
    regret = summary[method]["cum_regret"]
    rows.append(
        f"| {method} | {100 * accuracy['mean']:.2f}% | ±{100 * accuracy['ci95']:.2f} | "
        f"{regret['mean']:.2f} | ±{regret['ci95']:.2f} |"
    )
display(Markdown(header + "\n" + "\n".join(rows)))
'''.strip()


AGGREGATE_PLOT = r'''
import matplotlib.pyplot as plt

colors = {"Base": "#98A2B3", "ICL": "#D9903D", "RAG": "#D95C59",
          "Online-SFT": "#7D68B3", "Online-SDFT": "#0D9488"}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
x_positions = np.arange(len(METHODS))
accuracy = [100 * summary[name]["online_accuracy"]["mean"] for name in METHODS]
accuracy_ci = [100 * summary[name]["online_accuracy"]["ci95"] for name in METHODS]
regret = [summary[name]["cum_regret"]["mean"] for name in METHODS]
regret_ci = [summary[name]["cum_regret"]["ci95"] for name in METHODS]

for ax, values, errors, title, ylabel in (
    (axes[0], accuracy, accuracy_ci, "Prequential online accuracy", "Accuracy (%)"),
    (axes[1], regret, regret_ci, "Cumulative contextual-bandit regret", "Regret"),
):
    bars = ax.bar(x_positions, values, yerr=errors, capsize=4,
                  color=[colors[name] for name in METHODS], width=0.68)
    ax.set_xticks(x_positions, METHODS, rotation=14, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
fig.suptitle("One stream · score first, learn second · mean ± 95% CI", fontweight="bold")
fig.tight_layout()
plt.show()
'''.strip()


LEARNING_CURVES = r'''
import matplotlib.pyplot as plt

colors = {"Base": "#98A2B3", "ICL": "#D9903D", "RAG": "#D95C59",
          "Online-SFT": "#7D68B3", "Online-SDFT": "#0D9488"}

by_accuracy, by_regret = defaultdict(list), defaultdict(list)
for row in curves:
    key = (row["method"], int(row["t"]))
    by_accuracy[key].append(float(row["cum_accuracy"]))
    by_regret[key].append(float(row["cum_regret"]))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
for method in METHODS:
    ts = sorted(t for name, t in by_accuracy if name == method)
    acc = np.array([by_accuracy[(method, t)] for t in ts])
    mean_acc = np.array([values.mean() for values in acc]) * 100
    ci_acc = np.array([
        0.0 if len(values) == 1 else
        1.96 * values.std(ddof=1) / np.sqrt(len(values))
        for values in acc
    ]) * 100
    mean_regret = [np.mean(by_regret[(method, t)]) for t in ts]
    width = 2.8 if method == "Online-SDFT" else 1.5
    axes[0].plot(ts, mean_acc, color=colors[method], label=method, lw=width)
    axes[0].fill_between(ts, mean_acc - ci_acc, mean_acc + ci_acc,
                         color=colors[method], alpha=0.08)
    axes[1].plot(ts, mean_regret, color=colors[method], label=method, lw=width)

for ax in axes:
    for boundary in (PHASE_LENGTH, 2 * PHASE_LENGTH):
        ax.axvline(boundary, color="#667085", ls="--", lw=1)
    ax.grid(alpha=0.22)
    ax.set_xlabel("Online decisions")
axes[0].set(title="Cumulative online accuracy", ylabel="Accuracy so far (%)", ylim=(0, 100))
axes[1].set(title="Regret accumulated in arrival order", ylabel="Cumulative regret")
axes[0].legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()
'''.strip()


AUDIT_INVARIANT = r'''
from IPython.display import Markdown, display

archive_rows = [row for row in rollouts if row["action"] == "ARCHIVE"]
archive_outcomes = Counter(row["feedback"]["outcome"] for row in archive_rows)
assert set(archive_outcomes) <= {"ORGANIC_INBOX_OPEN", "NO_OBSERVATION"}
assert all("oracle_action_scoring_only" in row for row in rollouts)

display(Markdown(
    f"**{len(archive_rows):,} archived decisions** produced only factual outcomes: "
    f"`{dict(archive_outcomes)}`. No archived decision generated a push or digest click."
))
'''.strip()


ACTION_MIX = r'''
import matplotlib.pyplot as plt

counts = {method: Counter(row["action"] for row in rollouts if row["method"] == method)
          for method in METHODS}
fig, ax = plt.subplots(figsize=(10, 4.2))
bottom = np.zeros(len(METHODS))
action_colors = {"INTERRUPT": "#D95C59", "LATER": "#D9903D", "ARCHIVE": "#667085"}
for action in ACTIONS:
    values = np.array([counts[method][action] for method in METHODS])
    ax.bar(METHODS, values, bottom=bottom, label=action, color=action_colors[action])
    bottom += values
ax.set(title="Actions actually executed across all streams", ylabel="Decisions")
ax.legend(ncol=3)
ax.grid(axis="y", alpha=0.2)
plt.show()
'''.strip()


EXAMPLES = r'''
from IPython.display import Markdown, display

for example in qualitative[:4]:
    display(Markdown(
        f"### Step {example['t']} · {example['regime']} · {example['category']}  \n"
        f"Scoring-best route: **{example['oracle']}**"
    ))
    for method in METHODS:
        row = example["methods"][method]
        marker = "✅" if row["action"] == example["oracle"] else "·"
        print(f"{marker} {method:<13} {row['action']:<10} → {row['feedback']['outcome']}")
'''.strip()


def reader_code_cell(source: str, title: str):
    """Create a code cell whose input is collapsed in Colab/Jupyter viewers."""
    source = f'# @title {title} {{ display-mode: "form" }}\n{source}'
    return nbf.v4.new_code_cell(
        source,
        metadata={
            "cellView": "form",
            "jupyter": {"source_hidden": True},
            "tags": ["hide-input"],
        },
    )


def markdown_cells() -> list:
    animation_source = (
        GIF_FUNCTIONS
        + "\n\nfrom IPython.display import Image as NotebookImage, Markdown, display\n"
          "gif_bytes = make_online_sdft_gif()\n"
          "display(NotebookImage(data=gif_bytes))"
    )
    return [
        nbf.v4.new_markdown_cell(
            """# Learning when silence has no click: Online SDFT with Liquid LFM

This notebook runs a real
[`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M)
student. It is **repository-independent**: it contains the complete simulator,
policies, and analysis and reads no repository files. Its setup cell downloads
the `LiquidAI/LFM2.5-230M` checkpoint and Python packages on first use.

## Choose a reading path

| If you want to… | Jump to |
| --- | --- |
| See the answer first | [1. Results at a glance](#1-results-at-a-glance) |
| Understand the interaction visually | [2. Watch one causal round](#2-watch-one-causal-round) |
| Put yourself in the decision | [3. Play the router](#3-play-the-router) |
| Check why the setup is realistic | [4. Understand the setting](#4-understand-the-setting) |
| Compare Online-SFT and Online-SDFT | [5. Compare the methods](#5-compare-the-methods) |
| Reproduce everything | [6. Reproduce the experiment](#6-reproduce-the-experiment-optional) |
| Inspect curves and audits | [7. Metrics](#7-inspect-the-metrics-optional) · [8. Audits](#8-audit-the-realism-optional) |

> Long code cells are collapsed by default where the notebook viewer supports it. Expand only the implementation details you want."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Results at a glance

Every one of the 240 decisions counts, including cold start and exploration.
Mean over 3 paired streams:

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 37.08% ± 3.30 | 81.50 ± 2.24 |
| ICL | 37.08% ± 1.70 | 81.65 ± 0.87 |
| RAG | 38.61% ± 0.98 | 81.63 ± 6.75 |
| Online-SFT | 39.17% ± 5.10 | 102.82 ± 10.98 |
| **Online-SDFT** | **62.50% ± 5.66** | **43.33 ± 4.81** |

**Short conclusion:** the full soft teacher distribution is a substantially better online target than one sampled hard teacher action. Stop here if you only need the result."""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Watch one causal round

The animation shows the non-cheating order: predict, commit and score, observe one factual world, then learn for the next round."""
        ),
        reader_code_cell(animation_source, "Generate the in-memory Online-SDFT animation"),
        nbf.v4.new_markdown_cell(
            """### 2.1 What the animation establishes

- The student acts without current feedback or privileged telemetry.
- The score is frozen before the user outcome exists.
- Only the selected route produces feedback; unchosen outcomes stay unknown.
- The teacher's soft distribution updates at most four records for `t+1`.

If the student chooses `ARCHIVE`, no notification exists. The only factual outcomes are an organic inbox open or no observation—never a push or digest click."""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Play the router

Choose a scenario using only the visible cues, decide what **you** would do, then edit the two values in the play cell and run it.

| ID | Notification | Visible cues |
| ---: | --- | --- |
| `0` | Afternoon calendar alert | High importance and deadline pressure |
| `1` | On-call monitoring incident | High urgency during an incident |
| `2` | Off-hours message from a close friend | Low urgency, high affinity |
| `3` | Late-morning promotion | Low importance, deadline, and affinity |

Commit mentally before revealing the outcome. The engine cell is collapsed because its hidden state would spoil the game."""
        ),
        reader_code_cell(GAME_ENGINE, "Hidden game engine and scenario state"),
        nbf.v4.new_code_cell(
            GAME_PLAY,
            metadata={"tags": ["game-input"], "jupyter": {"source_hidden": False}},
        ),
        nbf.v4.new_markdown_cell(
            """### 3.1 What to notice

- A click does not prove the chosen route maximized total utility; interruption cost also matters.
- Only the chosen action creates factual feedback. The other two user reactions are never sampled.
- The evaluator-only debrief makes regret measurable in the simulator, but none of it trains the learner.
- Today's feedback can improve the **next** decision; it cannot rewrite this round."""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Understand the setting

### 4.1 Contextual-bandit contract

| Moment | Available | Sealed away |
| --- | --- | --- |
| Student rollout | current context, parameters, past factual records | current feedback, future events, scoring oracle |
| Environment | outcome caused by the selected action | outcomes for both unchosen actions |
| Online update | post-decision teacher target, batch ≤4 | oracle action, ground-truth demonstration, retroactive correction |

The evaluator can grade the frozen action from hidden simulator state, but neither student nor teacher receives its utility vector."""
        ),
        nbf.v4.new_markdown_cell(
            """### 4.2 Why this is not batch learning

| Batch learning | This online stream |
| --- | --- |
| Labels exist before training | Feedback exists only after acting |
| Examples may be shuffled for many epochs | Events arrive once, in order |
| Evaluation follows training | Every live action is evaluated |
| Early errors disappear from test metrics | Cold-start and adaptation errors remain |

There is no train/test split. The stream drifts from weekday to on-call to off-hours behavior, and the objective is performance **during** adaptation."""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Compare the methods

### 5.1 Five methods on the same stream

| Method | Online adaptation |
| --- | --- |
| Base | Frozen Liquid LFM2.5-230M |
| ICL | Recent sampled teacher actions in the LFM prompt |
| RAG | Similar past sampled teacher actions in the LFM prompt |
| Online-SFT | LoRA update from one sampled teacher action |
| **Online-SDFT** | LoRA update from the teacher's full soft distribution |

All methods normally take the route with the highest LFM action-token
probability, use the same 6% uniform exploration, and never receive the
evaluator's preferred action. The LFM is the deployed student; the controlled
benchmark uses an explicit stochastic simulator policy as its auditable
post-decision teacher."""
        ),
        nbf.v4.new_markdown_cell(
            """### 5.2 Online-SFT versus Online-SDFT

| | Online-SFT | Online-SDFT |
| --- | --- | --- |
| Student rollout | From `x_t`, without `z_t` | From `x_t`, without `z_t` |
| Teacher timing | After factual feedback | After factual feedback |
| Retained target | One sampled hard action | Full soft distribution `q_t` |
| Online batch | Fresh + up to 3 recent | Fresh + up to 3 recent |

Neither method trains on a ground-truth demonstration. SDFT preserves the teacher's relative preference over all routes instead of reducing it to one noisy draw."""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Reproduce the experiment (optional)

The next four subsections install the LLM runtime, contain the complete embedded
implementation, execute 3 paired streams, and recompute the headline table.
Skip to [Section 7](#7-inspect-the-metrics-optional) if you only want the saved
outputs."""
        ),
        nbf.v4.new_markdown_cell(
            """### 6.1 Install the LLM runtime

In Colab, first choose **Runtime → Change runtime type → T4 GPU**. The setup
cell keeps Colab's CUDA-matched PyTorch, removes its stale optional `torchao`
package (unused here), pins the tested Transformers/PEFT versions, and prints
the detected GPU. Local CPU and Apple MPS runs remain supported."""
        ),
        reader_code_cell(INSTALL_DEPS, "Install the Liquid LFM runtime"),
        nbf.v4.new_markdown_cell(
            """### 6.2 Load the embedded simulator and policies

This long cell defines the stream, factual feedback, scoring oracle, teacher, five methods, and online update loop. It reads no external file."""
        ),
        reader_code_cell(embedded_core(), "Optional: embedded simulator and method implementation"),
        nbf.v4.new_markdown_cell(
            """### 6.3 Run the paired streams

All artifacts remain in memory: 3 seeds × 5 methods × 240 online decisions.
This executes real LFM inference and online LoRA updates. The cell prints one
line after every method so a long GPU run never looks stalled."""
        ),
        reader_code_cell(RUNNER, "Run the complete paired experiment in memory"),
        nbf.v4.new_markdown_cell("### 6.4 Confirm the recomputed metrics"),
        reader_code_cell(RESULTS, "Display the recomputed result table"),
        nbf.v4.new_markdown_cell(
            """## 7. Inspect the metrics (optional)

### 7.1 Aggregate online accuracy and regret

Accuracy is binary; regret weights mistakes by the evaluator's hidden utility gap. Both include cold start and exploration."""
        ),
        reader_code_cell(AGGREGATE_PLOT, "Plot aggregate accuracy and cumulative regret"),
        nbf.v4.new_markdown_cell(
            """### 7.2 Learning along the stream

Dashed boundaries mark weekday → on-call → off-hours drift. Nothing is filtered or rescored after learning."""
        ),
        reader_code_cell(LEARNING_CURVES, "Plot online learning curves"),
        nbf.v4.new_markdown_cell(
            """## 8. Audit the realism (optional)

### 8.1 Counterfactual boundary

This assertion verifies that archived actions never produce notification clicks."""
        ),
        reader_code_cell(AUDIT_INVARIANT, "Assert the ARCHIVE feedback invariant"),
        nbf.v4.new_markdown_cell(
            """### 8.2 Actions actually executed

The action mix is a diagnostic, not a relabeled counterfactual dataset."""
        ),
        reader_code_cell(ACTION_MIX, "Plot executed action counts"),
        nbf.v4.new_markdown_cell(
            """## 9. Inspect later-stream examples (optional)

These examples are selected after all rollouts finish. They show later decisions where Online-SDFT's actual pre-feedback action matches the evaluator and every comparison arm does not."""
        ),
        reader_code_cell(EXAMPLES, "Show qualitative Online-SDFT wins"),
        nbf.v4.new_markdown_cell(
            """## 10. Takeaway

This is not batch training followed by a clean test. Every method pays for
cold-start, exploration, and adaptation errors as they happen. Online-SFT keeps
one noisy teacher draw; Online-SDFT keeps the teacher's complete relative
preference. On three identical paired streams, that soft signal reaches
**62.5% online accuracy and 43.3 cumulative regret**, versus **39.2% and 102.8**
for Online-SFT. The sample is deliberately small, so treat it as a reproducible
demonstration rather than a production-scale claim."""
        ),
    ]


def main() -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    namespace = {}
    exec(GIF_FUNCTIONS, namespace)
    gif_bytes = namespace["make_online_sdft_gif"]()
    GIF_PATH.write_bytes(gif_bytes)

    notebook = nbf.v4.new_notebook(
        cells=markdown_cells(),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    # Keep the notebook standalone and make its key animation visible in GitHub
    # before any cell is executed.
    for cell in notebook.cells:
        if cell.cell_type == "code" and "gif_bytes = make_online_sdft_gif()" in cell.source:
            cell.outputs = [nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/gif": base64.b64encode(gif_bytes).decode("ascii"),
                    "text/plain": "<IPython.core.display.Image object>",
                },
                metadata={},
            )]
            break
    nbf.write(notebook, NOTEBOOK)
    print(f"wrote {NOTEBOOK}")
    print(f"wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
