import os
from dataclasses import dataclass, field
from pathlib import Path

# Short names that can be passed on the command line.
# Values are the full HuggingFace model IDs.
MODEL_PRESETS: dict[str, str] = {
    "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "granite": "ibm-granite/granite-vision-3.2-2b",
}


def _env_path(var: str, default: str) -> Path:
    """Return Path from environment variable *var*, falling back to *default*."""
    return Path(os.environ.get(var, default))


def _env_str(var: str, default: str) -> str:
    return os.environ.get(var, default)


def _env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    return int(raw) if raw is not None else default


def _env_float(var: str, default: float) -> float:
    raw = os.environ.get(var)
    return float(raw) if raw is not None else default


@dataclass
class ProjectPaths:
    # Override via WEATHER_DATA_DIR / WEATHER_OUTPUT_DIR / WEATHER_MODELS_DIR
    data_dir: Path = field(default_factory=lambda: _env_path("WEATHER_DATA_DIR", "data"))
    raw_images_dir: Path = field(default_factory=lambda: _env_path("WEATHER_DATA_DIR", "data") / "raw_images")
    annotations_dir: Path = field(default_factory=lambda: _env_path("WEATHER_DATA_DIR", "data") / "annotations")
    outputs_dir: Path = field(default_factory=lambda: _env_path("WEATHER_OUTPUT_DIR", "outputs"))
    models_dir: Path = field(default_factory=lambda: _env_path("WEATHER_MODELS_DIR", "models"))


@dataclass
class IngestConfig:
    # Override via WEATHER_IMAGES_DIR / WEATHER_TRANSCRIPTIONS_DIR / WEATHER_INGEST_OUTPUT_DIR
    images_dir: Path = field(
        default_factory=lambda: _env_path("WEATHER_IMAGES_DIR", "Daily_rainfall_sample/images")
    )
    transcriptions_dir: Path = field(
        default_factory=lambda: _env_path("WEATHER_TRANSCRIPTIONS_DIR", "Daily_rainfall_sample/transcriptions")
    )
    output_dir: Path = field(
        default_factory=lambda: _env_path("WEATHER_INGEST_OUTPUT_DIR", "data/dataset")
    )


@dataclass
class ModelConfig:
    # Override via WEATHER_MODEL / WEATHER_MAX_NEW_TOKENS / WEATHER_DEVICE
    model_name: str = field(
        default_factory=lambda: _env_str("WEATHER_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
    )
    max_new_tokens: int = field(default_factory=lambda: _env_int("WEATHER_MAX_NEW_TOKENS", 2048))
    temperature: float = 0.0
    device: str = field(default_factory=lambda: _env_str("WEATHER_DEVICE", "auto"))


@dataclass
class TrainingConfig:
    # Override via WEATHER_TRAINING_OUTPUT_DIR / WEATHER_EPOCHS / WEATHER_BATCH_SIZE
    output_dir: Path = field(
        default_factory=lambda: _env_path("WEATHER_TRAINING_OUTPUT_DIR", "outputs/checkpoints")
    )
    learning_rate: float = field(default_factory=lambda: _env_float("WEATHER_LEARNING_RATE", 2e-4))
    epochs: int = field(default_factory=lambda: _env_int("WEATHER_EPOCHS", 3))
    batch_size: int = field(default_factory=lambda: _env_int("WEATHER_BATCH_SIZE", 1))
    gradient_accumulation_steps: int = field(
        default_factory=lambda: _env_int("WEATHER_GRAD_ACCUM_STEPS", 8)
    )
    eval_split: float = 0.1
    # LoRA hyper-parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # None → "all-linear" (PEFT selects all linear layers automatically)
    lora_target_modules: list[str] | None = None
    # Experiment tracking backend: "none", "wandb", "tensorboard"
    report_to: str = field(default_factory=lambda: _env_str("WEATHER_REPORT_TO", "none"))
    extra_args: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
