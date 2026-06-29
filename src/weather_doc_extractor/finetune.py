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
import os
from pathlib import Path
from typing import Any

from weather_doc_extractor.config import ModelConfig, TrainingConfig
from weather_doc_extractor.inference import EXTRACTION_PROMPT, detect_model_family
from weather_doc_extractor.schemas import DailyRainfallRecord

_CONSENSUS_ROW_KEYS = [f"Day {i}" for i in range(1, 32)] + ["Totals"]

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


def _coerce_consensus_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "null":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _consensus_answer_and_char_mask(
    consensus: dict[str, Any],
) -> tuple[str, list[bool]]:
    """Serialize consensus JSON to assistant text with structure-aware masking.

    Loss is computed for:
    - All structural elements (braces, brackets, commas, keys)
    - Value characters from cells where ``correct=true``

    Loss is NOT computed for:
    - Values from cells where ``correct=false`` or ``correct`` is missing

    This ensures the model learns both format structure and correct values.
    """

    parts: list[str] = []
    char_mask: list[bool] = []

    def _append(text: str, include_in_loss: bool) -> None:
        parts.append(text)
        char_mask.extend([include_in_loss] * len(text))

    _append("{", True)

    for row_idx, row_key in enumerate(_CONSENSUS_ROW_KEYS):
        if row_idx > 0:
            _append(",", True)

        _append(json.dumps(row_key), True)
        _append(":[", True)

        row = consensus.get(row_key)
        row_cells = row if isinstance(row, list) else []

        for month_idx in range(12):
            if month_idx > 0:
                _append(",", True)

            cell = (
                row_cells[month_idx]
                if month_idx < len(row_cells) and isinstance(row_cells[month_idx], dict)
                else {}
            )
            value = _coerce_consensus_value(cell.get("value"))
            value_text = json.dumps(value)
            include_in_loss = bool(cell.get("correct", False))
            _append(value_text, include_in_loss)

        _append("]", True)

    _append("}", True)
    answer = "".join(parts)
    return answer, char_mask


def _load_consensus_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Training-example builders
# ---------------------------------------------------------------------------


def _remote_code_kwargs_for_family(family: str) -> dict[str, bool]:
    """Return trust_remote_code kwargs for the given model family.

    granite4 has native support in transformers>=5.8.0; forcing
    trust_remote_code=True can load stale cached custom modules.
    """
    return {} if family == "granite4" else {"trust_remote_code": True}


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

    if family.startswith("granite") or family == "gemma3":
        # These families embed the PIL image directly in the message so the
        # processor can resolve it during apply_chat_template.
        image_item: dict[str, Any] = {"type": "image", "image": pil_image}
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
    else:
        # SmolVLM / Gemma 4 / generic: placeholder only; PIL passed separately.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
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


def build_consensus_training_example(
    record: DailyRainfallRecord,
    family: str,
) -> dict[str, Any] | None:
    """Build one strict-consensus training example.

    Returns ``None`` when no valid consensus transcription exists.
    """
    if record.transcription_path is None:
        return None

    consensus = _load_consensus_json(record.transcription_path)
    if consensus is None:
        return None

    answer, char_mask = _consensus_answer_and_char_mask(consensus)

    # Check if there's at least one correct cell (ignore structural tokens)
    has_correct_cell = any(
        consensus.get(row_key, [{}])[month_idx].get("correct", False)
        for row_key in _CONSENSUS_ROW_KEYS
        for month_idx in range(12)
        if isinstance(consensus.get(row_key), list)
        and month_idx < len(consensus.get(row_key, []))
        and isinstance(consensus.get(row_key, [])[month_idx], dict)
    )
    if not has_correct_cell:
        return None

    from PIL import Image as PILImage

    pil_image = PILImage.open(record.image_path).convert("RGB")

    if family.startswith("granite") or family == "gemma3":
        # Granite/Gemma3 require embedded PIL image content for
        # apply_chat_template(tokenize=True, return_dict=True) to emit
        # vision tensors.
        image_item: dict[str, Any] = {"type": "image", "image": pil_image}
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
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
    return {
        "image": pil_image,
        "messages": messages,
        "assistant_text": answer,
        "assistant_char_mask": char_mask,
    }


def build_consensus_training_examples(
    records: list[DailyRainfallRecord],
    family: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        ex = build_consensus_training_example(record, family)
        if ex is not None:
            examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def _make_collate_fn(processor: Any, family: str):
    """Return a collate function for SFTTrainer compatible with *family*.

    SmolVLM
        1. ``apply_chat_template(tokenize=False)`` → text prompt string
        2. ``processor(text=..., images=...)`` → tokenised batch
        3. Mask pad and image tokens in labels.

    Granite / Gemma 3
        1. ``apply_chat_template(tokenize=True, return_dict=True)`` per example
        2. Pad and stack tensors across the batch.
        3. Mask pad tokens in labels.
        (Gemma 3 additionally passes ``do_pan_and_scan=True``.)

    Gemma 4
        Same pipeline as SmolVLM but passes ``enable_thinking=False`` to
        ``apply_chat_template`` so reasoning tokens are suppressed.
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
        # Granite uses the llava_next architecture which requires image_sizes
        # in the forward call to split multi-patch pixel_values back into per-image
        # feature grids.  The processor returns it; pass it through.
        if "image_sizes" in batches[0]:
            image_sizes = torch.cat([b["image_sizes"] for b in batches], dim=0)
            batch["image_sizes"] = image_sizes
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch

    def _collate_gemma3(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Gemma 3 collate: embed PIL image in message, use do_pan_and_scan."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        batches = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                do_pan_and_scan=True,
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

    def _collate_gemma4(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Gemma 4 collate: two-step tokenise; disable thinking tokens."""
        import torch

        images = [ex["image"] for ex in examples]
        texts = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
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
        if hasattr(processor, "image_token"):
            img_tok_id = processor.tokenizer.convert_tokens_to_ids(
                processor.image_token
            )
            labels[labels == img_tok_id] = -100
        batch["labels"] = labels
        return batch

    if family.startswith("granite"):
        return _collate_granite
    _dispatch = {
        "gemma3": _collate_gemma3,
        "gemma4": _collate_gemma4,
    }
    return _dispatch.get(family, _collate_smolvlm)


def _find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for idx in range(len(haystack) - len(needle), -1, -1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def _assistant_keep_flags(
    tokenizer: Any,
    assistant_text: str,
    assistant_char_mask: list[bool],
) -> tuple[list[int], list[bool]]:
    if len(assistant_char_mask) != len(assistant_text):
        raise ValueError("assistant_char_mask length must match assistant_text length")

    encoded = tokenizer(
        assistant_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "Tokenizer does not provide offset_mapping; strict consensus masking requires a fast tokenizer."
        )

    keep_flags: list[bool] = []
    for start, end in offsets:
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            keep_flags.append(False)
            continue
        keep_flags.append(any(assistant_char_mask[start_i:end_i]))
    return token_ids, keep_flags


def _make_consensus_collate_fn(processor: Any, family: str):
    """Return a consensus collate function for SFTTrainer compatible with *family*.

    All collators apply strict token-level masking:
    - Loss only on assistant tokens where consensus JSON has correct=true
    - Loss on all structural elements (brackets, commas, keys)
    - Preserves model-family-specific tokenization quirks.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def _collate_consensus_smolvlm(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """SmolVLM consensus: tokenize=False then processor."""
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

        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)
        pad_id = processor.tokenizer.pad_token_id
        n_total = input_ids.shape[1]

        for row_idx, ex in enumerate(examples):
            row = input_ids[row_idx]
            batch_offset = 0
            if pad_id is not None and row[0].item() == pad_id:
                for k in range(n_total):
                    if row[k].item() != pad_id:
                        batch_offset = k
                        break

            prompt_text = processor.apply_chat_template(
                [ex["messages"][0]],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_batch = processor(
                text=[prompt_text],
                images=[ex["image"]],
                return_tensors="pt",
            )
            n_prompt = prompt_batch["input_ids"].shape[1]

            assistant_start = batch_offset + n_prompt
            if assistant_start >= n_total:
                continue

            _, keep_flags = _assistant_keep_flags(
                processor.tokenizer,
                ex["assistant_text"],
                ex["assistant_char_mask"],
            )

            for offset, keep in enumerate(keep_flags):
                pos = assistant_start + offset
                if pos >= n_total:
                    break
                if keep:
                    labels[row_idx, pos] = input_ids[row_idx, pos]

        batch["labels"] = labels
        return batch

    def _collate_consensus_granite(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Granite consensus: tokenize=True per-example, pad_sequence, preserve image_sizes."""
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
        pixel_values = torch.cat([b["pixel_values"] for b in batches], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if "image_sizes" in batches[0]:
            image_sizes = torch.cat([b["image_sizes"] for b in batches], dim=0)
            batch["image_sizes"] = image_sizes

        labels = torch.full_like(input_ids, -100)
        n_total = input_ids.shape[1]

        for row_idx, ex in enumerate(examples):
            # For tokenize=True, find assistant start by tokenizing user message only
            user_msg_batch = processor.apply_chat_template(
                [ex["messages"][0]],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            n_prompt = user_msg_batch["input_ids"].shape[1]
            assistant_start = n_prompt
            if assistant_start >= n_total:
                continue

            _, keep_flags = _assistant_keep_flags(
                processor.tokenizer,
                ex["assistant_text"],
                ex["assistant_char_mask"],
            )

            for offset, keep in enumerate(keep_flags):
                pos = assistant_start + offset
                if pos >= n_total:
                    break
                if keep:
                    labels[row_idx, pos] = input_ids[row_idx, pos]

        batch["labels"] = labels
        return batch

    def _collate_consensus_gemma3(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Gemma 3 consensus: tokenize=True with do_pan_and_scan, preserve model quirks."""
        batches = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                do_pan_and_scan=True,
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
        pixel_values = torch.cat([b["pixel_values"] for b in batches], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        labels = torch.full_like(input_ids, -100)
        n_total = input_ids.shape[1]

        for row_idx, ex in enumerate(examples):
            user_msg_batch = processor.apply_chat_template(
                [ex["messages"][0]],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                do_pan_and_scan=True,
            )
            n_prompt = user_msg_batch["input_ids"].shape[1]
            assistant_start = n_prompt
            if assistant_start >= n_total:
                continue

            _, keep_flags = _assistant_keep_flags(
                processor.tokenizer,
                ex["assistant_text"],
                ex["assistant_char_mask"],
            )

            for offset, keep in enumerate(keep_flags):
                pos = assistant_start + offset
                if pos >= n_total:
                    break
                if keep:
                    labels[row_idx, pos] = input_ids[row_idx, pos]

        batch["labels"] = labels
        return batch

    def _collate_consensus_gemma4(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Gemma 4 consensus: tokenize=False with enable_thinking=False."""
        images = [ex["image"] for ex in examples]
        texts = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for ex in examples
        ]
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)
        pad_id = processor.tokenizer.pad_token_id
        n_total = input_ids.shape[1]

        for row_idx, ex in enumerate(examples):
            row = input_ids[row_idx]
            batch_offset = 0
            if pad_id is not None and row[0].item() == pad_id:
                for k in range(n_total):
                    if row[k].item() != pad_id:
                        batch_offset = k
                        break

            prompt_text = processor.apply_chat_template(
                [ex["messages"][0]],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_batch = processor(
                text=[prompt_text],
                images=[ex["image"]],
                return_tensors="pt",
            )
            n_prompt = prompt_batch["input_ids"].shape[1]

            assistant_start = batch_offset + n_prompt
            if assistant_start >= n_total:
                continue

            _, keep_flags = _assistant_keep_flags(
                processor.tokenizer,
                ex["assistant_text"],
                ex["assistant_char_mask"],
            )

            for offset, keep in enumerate(keep_flags):
                pos = assistant_start + offset
                if pos >= n_total:
                    break
                if keep:
                    labels[row_idx, pos] = input_ids[row_idx, pos]

        batch["labels"] = labels
        return batch

    # Ministral uses the same logic as SmolVLM (tokenize=False)
    _dispatch = {
        "granite": _collate_consensus_granite,
        "granite4": _collate_consensus_granite,
        "gemma3": _collate_consensus_gemma3,
        "gemma4": _collate_consensus_gemma4,
    }
    return _dispatch.get(family, _collate_consensus_smolvlm)


# ---------------------------------------------------------------------------
# LoRA configuration helpers
# ---------------------------------------------------------------------------


def _get_lora_task_type(family: str) -> str:
    """Return appropriate PEFT task type for the given model family.

    - Gemma4 uses AutoModelForCausalLM → task_type="CAUSAL_LM"
    - All others (SmolVLM, Granite, Gemma3, Ministral) use
      AutoModelForImageTextToText → task_type="SEQ_2_SEQ_LM"
    """
    return "CAUSAL_LM" if family == "gemma4" else "SEQ_2_SEQ_LM"


def _training_device_map(model_device: str) -> str | dict[str, int] | None:
    """Return a safe device_map for single-process and DDP fine-tuning.

    In distributed launches (WORLD_SIZE > 1), ``device_map='auto'`` can conflict
    with DDP process placement. Pin each process to its LOCAL_RANK device instead.
    """
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1

    if world_size <= 1:
        return model_device

    if model_device not in {"", "auto"}:
        return model_device

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        local_rank = 0

    return {"": local_rank}


def _effective_gradient_accumulation_steps(
    base_steps: int,
    world_size: int,
    auto_scale: bool,
) -> int:
    """Return gradient accumulation steps adjusted for distributed training.

    Keeps global batch approximately stable when scaling from 1 process to many:
    ``global_batch = per_device_batch * grad_accum * world_size``
    """
    base = max(1, int(base_steps))
    ws = max(1, int(world_size))
    if not auto_scale or ws <= 1:
        return base
    return max(1, (base + ws - 1) // ws)


def _world_size_from_env() -> int:
    """Return WORLD_SIZE from environment, defaulting to 1."""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _get_canonical_model_id(model_name: str) -> str:
    """Return the canonical HuggingFace model ID for the given model preset or full ID.

    Maps short presets (e.g., 'smolvlm2') to their full HuggingFace IDs.
    If an unknown preset is provided, returns the input as-is (assumed to be a full ID).

    This mapping matches the one in aml_submit.sh (RESOLVED_TRAINING_MODEL_NAME).
    """
    preset_map = {
        "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "granite": "ibm-granite/granite-vision-3.2-2b",
        "granite4": "ibm-granite/granite-vision-4.1-4b",
        "gemma3": "google/gemma-3-4b-it",
        "gemma4": "google/gemma-4-E4B-it",
        "ministral": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    }
    return preset_map.get(model_name, model_name)


def _fix_adapter_config_json(adapter_dir: Path, canonical_base_model_id: str) -> None:
    """Fix adapter_config.json to use canonical base_model_name_or_path.

    PEFT saves the base_model_name_or_path as whatever was passed to model loading,
    which could be a local HuggingFace cache path. When the adapter is loaded in a
    different environment (e.g., Azure ML with a different cache path), this fails.

    This function updates the adapter_config.json to use the canonical HuggingFace ID
    instead, ensuring the adapter can be loaded in any environment.

    Parameters
    ----------
    adapter_dir : Path
        Directory containing the saved adapter (with adapter_config.json).
    canonical_base_model_id : str
        The canonical HuggingFace model ID (e.g., "HuggingFaceTB/SmolVLM2-2.2B-Instruct").
    """
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        return

    try:
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        old_base = adapter_cfg.get("base_model_name_or_path", "")

        # Only update if it looks like a local path (contains '/' or is obviously a cache path)
        # and differs from the canonical ID
        if old_base and old_base != canonical_base_model_id:
            adapter_cfg["base_model_name_or_path"] = canonical_base_model_id
            adapter_cfg_path.write_text(
                json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[finetune] Fixed adapter_config.json base_model_name_or_path:\n"
                f"           {old_base}\n"
                f"        → {canonical_base_model_id}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[finetune] WARNING: Failed to fix adapter_config.json: {exc}",
            flush=True,
        )


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
        from peft import LoraConfig, PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Install fine-tuning dependencies: pip install -e '.[train]'"
        ) from exc

    family = detect_model_family(model_config.model_name)
    # Gemma 4 requires AutoModelForCausalLM; all others use
    # AutoModelForImageTextToText.
    use_causal_lm = family == "gemma4"
    model_cls = AutoModelForCausalLM if use_causal_lm else AutoModelForImageTextToText

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
    world_size = _world_size_from_env()
    grad_accum_steps = _effective_gradient_accumulation_steps(
        train_config.gradient_accumulation_steps,
        world_size,
        train_config.auto_scale_grad_accum,
    )
    if train_config.auto_scale_grad_accum and world_size > 1:
        print(
            "[finetune] Auto-scaled gradient accumulation steps for distributed training:\n"
            f"           base={train_config.gradient_accumulation_steps} world_size={world_size} "
            f"effective={grad_accum_steps}",
            flush=True,
        )

    # Resolve model path via HF cache (same logic as inference, avoids
    # re-downloading and bypasses the trust_remote_code Hub check)
    from weather_doc_extractor.inference import _local_hf_home, _resolve_model_path
    import os
    from pathlib import Path as _Path

    original_hf_home = os.environ.get("HF_HOME")
    local_cache = _local_hf_home()
    if local_cache is not None:
        os.environ["HF_HOME"] = str(local_cache)
        try:
            import huggingface_hub.constants as _hfc

            _hfc.HF_HOME = str(local_cache)
            _hfc.HUGGINGFACE_HUB_CACHE = str(local_cache / "hub")
            _hfc.HF_HUB_CACHE = str(local_cache / "hub")
        except Exception:
            pass
        if original_hf_home and _Path(original_hf_home) != local_cache:
            os.environ["_ORIGINAL_HF_HOME"] = original_hf_home

    hf_cache_dir: str | None = str(local_cache / "hub") if local_cache else None

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    train_device_map = _training_device_map(model_config.device)

    # Resume from an existing LoRA checkpoint when adapter_config.json is present.
    output_model_name = model_config.model_name
    adapter_root = train_config.checkpoint_dir
    adapter_cfg_path = (
        Path(adapter_root) / "adapter_config.json" if adapter_root else None
    )
    if adapter_cfg_path is not None and adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        base_model_name = adapter_cfg["base_model_name_or_path"]
        output_model_name = base_model_name
        resolved_base = _resolve_model_path(base_model_name)

        proc_kwargs: dict[str, Any] = _remote_code_kwargs_for_family(family)
        if hf_cache_dir:
            proc_kwargs["cache_dir"] = hf_cache_dir
        processor = AutoProcessor.from_pretrained(resolved_base, **proc_kwargs)

        base_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": train_device_map,
            **_remote_code_kwargs_for_family(family),
        }
        if hf_cache_dir:
            base_kwargs["cache_dir"] = hf_cache_dir
        base_model = model_cls.from_pretrained(resolved_base, **base_kwargs)
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_root),
            is_trainable=True,
        )
        lora_cfg = None
    else:
        resolved_name = _resolve_model_path(model_config.model_name)

        # Load processor and model
        proc_kwargs = _remote_code_kwargs_for_family(family)
        if hf_cache_dir:
            proc_kwargs["cache_dir"] = hf_cache_dir
        processor = AutoProcessor.from_pretrained(resolved_name, **proc_kwargs)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": train_device_map,
            **_remote_code_kwargs_for_family(family),
        }
        if hf_cache_dir:
            model_kwargs["cache_dir"] = hf_cache_dir
        model = model_cls.from_pretrained(resolved_name, **model_kwargs)

        # LoRA configuration
        target_modules = train_config.lora_target_modules or "all-linear"
        lora_cfg = LoraConfig(
            r=train_config.lora_r,
            lora_alpha=train_config.lora_alpha,
            lora_dropout=train_config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=_get_lora_task_type(family),
        )
        output_model_name = model_config.model_name

    model_slug = output_model_name.replace("/", "--")
    output_dir = train_config.output_dir / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        gradient_accumulation_steps=grad_accum_steps,
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

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_examples,
        "eval_dataset": eval_examples,
        "data_collator": collate_fn,
        "processing_class": processor,
    }
    if lora_cfg is not None:
        trainer_kwargs["peft_config"] = lora_cfg

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    # When resuming from a checkpoint, use its canonical base model ID.
    # When training from scratch, use the preset's canonical ID.
    # This ensures adapter_config.json always contains the correct HF Hub ID,
    # not a local datastore path.
    if adapter_cfg_path is not None and adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        loaded_base_model = adapter_cfg["base_model_name_or_path"]
        _fix_adapter_config_json(output_dir, _get_canonical_model_id(loaded_base_model))
    else:
        _fix_adapter_config_json(
            output_dir, _get_canonical_model_id(model_config.model_name)
        )

    return output_dir


def run_finetune_consensus(
    records: list[DailyRainfallRecord],
    model_config: ModelConfig,
    train_config: TrainingConfig,
) -> Path:
    """Fine-tune any model checkpoint using strict consensus-token masking."""
    try:
        import torch
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Install fine-tuning dependencies: pip install -e '.[train]'"
        ) from exc

    family = detect_model_family(model_config.model_name)

    transcribed = [r for r in records if r.transcription_path is not None]
    if len(transcribed) < 2:
        raise ValueError(
            "Need at least two records with consensus transcription files for train/eval split."
        )

    n_eval = max(1, int(len(transcribed) * train_config.eval_split))
    train_records = transcribed[:-n_eval]
    eval_records = transcribed[-n_eval:]

    train_examples = build_consensus_training_examples(train_records, family)
    eval_examples = build_consensus_training_examples(eval_records, family)
    world_size = _world_size_from_env()
    grad_accum_steps = _effective_gradient_accumulation_steps(
        train_config.gradient_accumulation_steps,
        world_size,
        train_config.auto_scale_grad_accum,
    )
    if train_config.auto_scale_grad_accum and world_size > 1:
        print(
            "[finetune-consensus] Auto-scaled gradient accumulation steps for distributed training:\n"
            f"                     base={train_config.gradient_accumulation_steps} world_size={world_size} "
            f"effective={grad_accum_steps}",
            flush=True,
        )
    if not train_examples:
        raise ValueError(
            "No valid consensus training examples with correct=true cells."
        )
    if not eval_examples:
        raise ValueError(
            "No valid consensus evaluation examples with correct=true cells."
        )

    # Reuse local HF cache path resolution from normal fine-tune path.
    from weather_doc_extractor.inference import _local_hf_home, _resolve_model_path
    import os
    from pathlib import Path as _Path

    original_hf_home = os.environ.get("HF_HOME")
    local_cache = _local_hf_home()
    if local_cache is not None:
        os.environ["HF_HOME"] = str(local_cache)
        try:
            import huggingface_hub.constants as _hfc

            _hfc.HF_HOME = str(local_cache)
            _hfc.HUGGINGFACE_HUB_CACHE = str(local_cache / "hub")
            _hfc.HF_HUB_CACHE = str(local_cache / "hub")
        except Exception:
            pass
        if original_hf_home and _Path(original_hf_home) != local_cache:
            os.environ["_ORIGINAL_HF_HOME"] = original_hf_home

    hf_cache_dir: str | None = str(local_cache / "hub") if local_cache else None

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    train_device_map = _training_device_map(model_config.device)

    # Continue from existing LoRA checkpoint when adapter_config.json is present.
    output_model_name = model_config.model_name
    adapter_root = train_config.checkpoint_dir or Path(model_config.model_name)
    adapter_cfg_path = Path(adapter_root) / "adapter_config.json"
    if adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        base_model_name = adapter_cfg["base_model_name_or_path"]
        output_model_name = base_model_name
        resolved_base = _resolve_model_path(base_model_name)

        load_family = detect_model_family(base_model_name)
        proc_kwargs: dict[str, Any] = _remote_code_kwargs_for_family(load_family)
        if hf_cache_dir:
            proc_kwargs["cache_dir"] = hf_cache_dir
        processor = AutoProcessor.from_pretrained(resolved_base, **proc_kwargs)

        base_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": train_device_map,
            **_remote_code_kwargs_for_family(load_family),
        }
        if hf_cache_dir:
            base_kwargs["cache_dir"] = hf_cache_dir
        base_model = AutoModelForImageTextToText.from_pretrained(
            resolved_base,
            **base_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_root),
            is_trainable=True,
        )
        lora_cfg = None
    else:
        resolved_name = _resolve_model_path(model_config.model_name)

        proc_kwargs = _remote_code_kwargs_for_family(family)
        if hf_cache_dir:
            proc_kwargs["cache_dir"] = hf_cache_dir
        processor = AutoProcessor.from_pretrained(resolved_name, **proc_kwargs)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": train_device_map,
            **_remote_code_kwargs_for_family(family),
        }
        if hf_cache_dir:
            model_kwargs["cache_dir"] = hf_cache_dir
        model = AutoModelForImageTextToText.from_pretrained(
            resolved_name, **model_kwargs
        )

        target_modules = train_config.lora_target_modules or "all-linear"
        lora_cfg = LoraConfig(
            r=train_config.lora_r,
            lora_alpha=train_config.lora_alpha,
            lora_dropout=train_config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=_get_lora_task_type(family),
        )

    model_slug = output_model_name.replace("/", "--")
    output_dir = train_config.output_dir / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=train_config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        logging_steps=10,
        fp16=False,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        report_to=train_config.report_to,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_examples,
        "eval_dataset": eval_examples,
        "data_collator": _make_consensus_collate_fn(processor, family),
        "processing_class": processor,
    }
    if lora_cfg is not None:
        trainer_kwargs["peft_config"] = lora_cfg

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    # When resuming from a checkpoint, use its canonical base model ID.
    # When training from scratch, use the preset's canonical ID.
    # This ensures adapter_config.json always contains the correct HF Hub ID,
    # not a local datastore path.
    if adapter_cfg_path is not None and adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        loaded_base_model = adapter_cfg["base_model_name_or_path"]
        _fix_adapter_config_json(output_dir, _get_canonical_model_id(loaded_base_model))
    else:
        _fix_adapter_config_json(
            output_dir, _get_canonical_model_id(model_config.model_name)
        )

    return output_dir
