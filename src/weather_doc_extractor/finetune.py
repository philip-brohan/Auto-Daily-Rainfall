"""Supervised fine-tuning of VLMs on daily-rainfall extraction pairs.

Public API
----------
build_training_example(record, family)
    Convert one DailyRainfallRecord into an SFT training-example dict.

build_training_examples(records, family)
    Build a list of SFT training-example dicts from paired records.

run_finetune(records, model_config, train_config)
    Load the model, apply LoRA, run SFTTrainer, and save the adapter.
    Returns the output directory path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weather_doc_extractor.config import ModelConfig, TrainingConfig
from weather_doc_extractor.inference import EXTRACTION_PROMPT, detect_model_family
from weather_doc_extractor.schemas import DailyRainfallRecord


# ---------------------------------------------------------------------------
# Ground-truth serialisation
# ---------------------------------------------------------------------------


def _ground_truth_json(record: DailyRainfallRecord) -> str:
    """Return the known-good grid as the canonical JSON string.

    Produces ``{"Day 1": [...], ..., "Day 31": [...], "Totals": [...]}``
    matching exactly the format that the extraction prompt requests.
    """
    assert record.grid is not None, "record must have a paired grid"
    data = {**record.grid.days, "Totals": record.grid.totals}
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Training-example builders
# ---------------------------------------------------------------------------


def build_training_example(
    record: DailyRainfallRecord,
    family: str,
) -> dict[str, Any]:
    """Convert one DailyRainfallRecord into an SFT training-example dict.

    The returned dict always has:
    - ``"image"`` – a ``PIL.Image.Image`` (RGB)
    - ``"messages"`` – a chat-format list understood by the family's collate_fn

    SmolVLM / generic
        ``{"type": "image"}`` placeholder in the user content; PIL stored
        separately under ``"image"``.  The SmolVLM collate_fn passes it via
        ``processor(text=..., images=[...])`` separately from the text prompt.

    Granite
        ``{"type": "image", "image": pil}`` embedded directly in the user
        content so that ``processor.apply_chat_template(tokenize=True, ...)``
        can resolve it in one call.
    """
    from PIL import Image as PILImage  # lazy import – no model needed for tests

    pil_image = PILImage.open(record.image_path).convert("RGB")
    answer = _ground_truth_json(record)

    if family == "granite":
        image_item: dict[str, Any] = {"type": "image", "image": pil_image}
    else:
        image_item = {"type": "image"}

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                image_item,
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return {"image": pil_image, "messages": messages}


def build_training_examples(
    records: list[DailyRainfallRecord],
    family: str,
) -> list[dict[str, Any]]:
    """Build SFT training examples for all *paired* records.

    Records without a transcription (``grid is None``) are silently skipped.
    """
    return [build_training_example(r, family) for r in records if r.grid is not None]


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def _make_collate_fn(processor: Any, family: str):
    """Return a collate function for SFTTrainer compatible with *family*.

    SmolVLM
        1. ``apply_chat_template(tokenize=False)`` → text prompt string
        2. ``processor(text=..., images=...)`` → tokenised batch
        3. Mask pad and image tokens in labels.

    Granite
        1. ``apply_chat_template(tokenize=True, return_dict=True)`` per example
        2. Pad and stack tensors across the batch.
        3. Mask pad tokens in labels.
    """

    def _collate_smolvlm(examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        images = [ex["image"] for ex in examples]
        texts = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in examples
        ]
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Don't compute loss on image-placeholder tokens
        if hasattr(processor, "image_token"):
            img_tok_id = processor.tokenizer.convert_tokens_to_ids(
                processor.image_token
            )
            labels[labels == img_tok_id] = -100
        batch["labels"] = labels
        return batch

    def _collate_granite(examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        batches = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            for ex in examples
        ]

        pad_id = processor.tokenizer.pad_token_id or 0
        input_ids = pad_sequence(
            [b["input_ids"][0] for b in batches],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = pad_sequence(
            [b["attention_mask"][0] for b in batches],
            batch_first=True,
            padding_value=0,
        )
        # pixel_values tensors may differ in spatial size; stack is valid when
        # images are resized to a fixed size by the processor (true for Granite).
        pixel_values = torch.cat([b["pixel_values"] for b in batches], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch

    return _collate_smolvlm if family != "granite" else _collate_granite


# ---------------------------------------------------------------------------
# Main fine-tuning entry point
# ---------------------------------------------------------------------------


def run_finetune(
    records: list[DailyRainfallRecord],
    model_config: ModelConfig,
    train_config: TrainingConfig,
) -> Path:
    """Fine-tune the model on paired records using TRL SFTTrainer + LoRA.

    Steps
    -----
    1. Split paired records into train / eval sets.
    2. Build per-family training examples (PIL images + chat messages).
    3. Load model and processor from HuggingFace.
    4. Wrap model in LoRA via PEFT.
    5. Run ``SFTTrainer.train()``.
    6. Save the adapter to ``train_config.output_dir / <model-slug>``.

    Returns
    -------
    Path
        Directory where the fine-tuned LoRA adapter was saved.
    """
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Install fine-tuning dependencies: pip install -e '.[train]'"
        ) from exc

    family = detect_model_family(model_config.model_name)
    paired = [r for r in records if r.grid is not None]

    if not paired:
        raise ValueError(
            "No paired records found; cannot fine-tune without ground truth."
        )

    # Train / eval split
    n_eval = max(1, int(len(paired) * train_config.eval_split))
    train_records = paired[:-n_eval]
    eval_records = paired[-n_eval:]

    train_examples = build_training_examples(train_records, family)
    eval_examples = build_training_examples(eval_records, family)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        model_config.model_name, trust_remote_code=True
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_config.model_name,
        torch_dtype=dtype,
        device_map=model_config.device,
        trust_remote_code=True,
    )

    # LoRA configuration
    target_modules = train_config.lora_target_modules or "all-linear"
    lora_cfg = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Output directory: one sub-directory per model so runs don't overwrite
    model_slug = model_config.model_name.replace("/", "--")
    output_dir = train_config.output_dir / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        # Disable TRL's built-in dataset preparation; our collate_fn does it.
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        logging_steps=10,
        fp16=False,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        report_to=train_config.report_to,
    )

    collate_fn = _make_collate_fn(processor, family)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_examples,
        eval_dataset=eval_examples,
        data_collator=collate_fn,
        peft_config=lora_cfg,
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    return output_dir
