"""Vision-LLM inference for daily-rainfall grid extraction.

Public API
----------
detect_model_family(model_name)
    Return the model family tag ("smolvlm", "granite", "granite4", or
    "generic").

build_messages(image_path, model_family)
    Build the chat-message list (with embedded image) for a VLM.

extract_grid(image_path, config)
    Load the configured model and extract a DailyRainfallGrid from an image.

parse_extraction_response(text)
    Parse a raw LLM text response into a DailyRainfallGrid (or None).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from weather_doc_extractor.config import ModelConfig
from weather_doc_extractor.ingest import _coerce_value
from weather_doc_extractor.schemas import DailyRainfallGrid

# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

#: Known model families and the substrings that identify them.
#: Order matters — more specific prefixes must come before broader ones.
_FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("gemma4", ["gemma-4"]),
    ("gemma3", ["gemma-3"]),
    ("smolvlm2", ["smolvlm2"]),
    ("smolvlm", ["smolvlm", "idefics"]),
    ("granite4", ["granite-vision-4.1"]),
    ("granite", ["granite"]),
    ("ministral", ["mistral-small-3", "pixtral"]),
]


def _extract_hf_model_id_from_snapshot_path(path_like: str) -> str | None:
    """Derive a HuggingFace model ID from a hub snapshot/cache path.

    Example:
    ``.../hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/<sha>``
    -> ``HuggingFaceTB/SmolVLM2-2.2B-Instruct``
    """
    marker = "models--"
    snapshots = "/snapshots/"
    norm = str(path_like).replace("\\", "/")

    start = norm.find(marker)
    if start < 0:
        return None
    after = norm[start + len(marker) :]
    end = after.find(snapshots)
    if end < 0:
        return None

    repo_token = after[:end]
    parts = repo_token.split("--", 1)
    if len(parts) != 2:
        return None
    org, repo = parts[0].strip(), parts[1].strip()
    if not org or not repo:
        return None
    return f"{org}/{repo}"


def _normalize_model_reference(model_ref: str) -> str:
    """Normalize model references from adapter configs across environments.

    PEFT adapters can store ``base_model_name_or_path`` as an absolute cache
    snapshot path from a different machine/job. When that path is not present
    in the current runtime, convert it to a canonical HF model ID.
    """
    raw = str(model_ref).strip()
    if not raw:
        return raw

    p = Path(raw)
    if p.exists():
        return raw

    hf_model_id = _extract_hf_model_id_from_snapshot_path(raw)
    if hf_model_id:
        print(
            f"[adapter] Normalized base model reference:\n"
            f"           {raw}\n"
            f"        -> {hf_model_id}",
            flush=True,
        )
        return hf_model_id

    return raw


def detect_model_family(model_name: str) -> str:
    """Return the model family for *model_name*.

    Returns one of ``"smolvlm"``, ``"smolvlm2"``, ``"granite"``,
    ``"granite4"``, or ``"generic"``.
    The family governs how messages are built and how the processor is called.

    If *model_name* is a local adapter directory, the base model name is read
    from ``adapter_config.json`` and used for family detection.
    """
    name = model_name
    adapter_cfg = Path(model_name) / "adapter_config.json"
    if adapter_cfg.exists():
        import json as _json

        name = _json.loads(adapter_cfg.read_text()).get(
            "base_model_name_or_path", model_name
        )
        name = _normalize_model_reference(str(name))

    lower = name.lower()
    for family, patterns in _FAMILY_PATTERNS:
        if any(p in lower for p in patterns):
            return family
    return "generic"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a data-entry assistant for historical meteorological documents.
The image shows a scanned daily-rainfall register page.
The table has:
  - Rows labelled "Day 1" through "Day 31" plus a "Totals" row.
  - 12 columns, one for each month in order:
    January, February, March, April, May, June,
    July, August, September, October, November, December.
  - Values are rainfall amounts in inches. Missing or illegible cells are blank.

Return ONLY a JSON object with exactly these 32 keys: "Day 1", "Day 2", ..., "Day 31", "Totals"
Each value is a JSON array of EXACTLY 12 numbers, one per month in the order above.
Use JSON null for any blank or illegible cell. Do not use strings for numbers.

Example row with all 12 months present:
"Day 1": [0.12, null, 0.05, 0.0, null, null, 0.38, null, null, 0.22, null, 0.07]
          Jan   Feb   Mar   Apr  May   Jun   Jul   Aug   Sep   Oct   Nov   Dec

Output ONLY the JSON object, no commentary, no markdown fences:"""


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------


def build_messages(
    image_path: Path | None = None,
    model_family: str = "smolvlm",
    pil_image: Any = None,
) -> list[dict[str, Any]]:
    """Return a chat-message list appropriate for *model_family*.

    SmolVLM / SmolVLM2 / Idefics3 / Gemma 4 / generic
        Uses ``{"type": "image"}`` as a placeholder; the PIL image is passed
        separately to ``processor(images=[...])``.

    Granite 3.2 / Granite 4.1 / Gemma 3
        Embeds the PIL image directly in the message content as
        ``{"type": "image", "image": pil_image}`` so that
        ``processor.apply_chat_template`` can resolve it.

    Granite (path-based fallback)
        When *pil_image* is ``None`` for the granite family, embeds the image
        URL/path as ``{"type": "image", "url": str(image_path)}``.

    Parameters
    ----------
    image_path:
        Path to the image file.  Required for Granite model families when
        *pil_image* is not provided; ignored for ``"smolvlm"``.
    model_family:
        One of ``"smolvlm"``, ``"smolvlm2"``, ``"granite"``, ``"granite4"``,
        ``"gemma3"``, ``"gemma4"``, or ``"generic"``.
    pil_image:
        Pre-loaded ``PIL.Image.Image``.  Required for the ``"gemma3"`` family;
        used instead of the URL for Granite families if provided.
    """
    if model_family == "gemma3":
        image_item: dict[str, Any] = {"type": "image", "image": pil_image}
        return [
            {
                "role": "user",
                "content": [
                    image_item,
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ]
    if model_family.startswith("granite"):
        if pil_image is not None:
            image_item = {"type": "image", "image": pil_image}
        else:
            image_item = {
                "type": "image",
                "url": str(image_path) if image_path is not None else "",
            }
        return [
            {
                "role": "user",
                "content": [
                    image_item,
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ]
    # SmolVLM / Gemma 4 / generic: placeholder only; PIL image passed separately
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_DAY_KEYS = {f"Day {i}" for i in range(1, 32)}
_MONTHS = 12


def _pad_or_trim(values: list[Any], length: int = _MONTHS) -> list[float | None]:
    """Normalise a list to exactly *length* coerced floats-or-None."""
    result = [_coerce_value(v) for v in values[:length]]
    result += [None] * max(0, length - len(result))
    return result


def _repair_truncated_json(text: str) -> str:
    """Attempt to close a truncated JSON object.

    If the model generation was cut off before the closing ``}``, this
    function closes any open array, then closes the object.  Returns the
    original text unchanged if it already ends with ``}``.
    """
    text = text.rstrip()
    if text.endswith("}"):
        return text

    # Close an open array if present, then close the object
    if not text.endswith("]"):
        # Drop any trailing partial token (e.g. a half-written number)
        text = re.sub(r",\s*$", "", text)  # trailing comma
        text = re.sub(r"[\d.]+\s*$", "", text)  # trailing partial number
        text = text.rstrip().rstrip(",")
        text += "]"
    text += "\n}"
    return text


def _extract_object(text: str) -> str | None:
    """Return the first JSON object literal found in *text*.

    Handles three common model output patterns:
    - Plain object: ``{...}``
    - Array-wrapped: ``[{...}]`` (Granite sometimes wraps in an array)
    - Prose + object: ``"Here is the data: {...}"``

    Returns the object string, or ``None`` if no ``{`` is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:]
    # Strip a trailing array bracket that some models (Granite) append
    # e.g. "{ ... }\n]" → "{ ... }"
    fragment = re.sub(r"\}\s*\]\s*$", "}", fragment.rstrip())
    return fragment


def parse_extraction_response(text: str) -> DailyRainfallGrid | None:
    """Attempt to parse *text* into a :class:`DailyRainfallGrid`.

    Tolerances applied:
    - Strips markdown code fences.
    - Handles array-wrapped JSON output ``[{...}]`` (Granite convention).
    - Searches for the first ``{…}`` block if prose surrounds the JSON.
    - Attempts to repair truncated JSON (missing closing ``}``).
    - Coerces string numbers to float.
    - Returns ``None`` only if no parseable JSON structure can be found.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    fragment = _extract_object(cleaned)
    if fragment is None:
        return None

    # Try to parse as-is first; if that fails, attempt repair
    raw: dict[str, list[Any]] | None = None
    for candidate in (fragment, _repair_truncated_json(fragment)):
        try:
            raw = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if raw is None:
        return None

    days: dict[str, list[float | None]] = {}
    totals: list[float | None] = [None] * _MONTHS

    for key, values in raw.items():
        if not isinstance(values, list):
            continue
        if key in _DAY_KEYS:
            days[key] = _pad_or_trim(values)
        elif key.lower() == "totals":
            totals = _pad_or_trim(values)

    if not days:
        return None

    # Fill in any missing day keys with all-None rows
    for i in range(1, 32):
        days.setdefault(f"Day {i}", [None] * _MONTHS)

    return DailyRainfallGrid(days=days, totals=totals)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def _is_adapter_path(name: str) -> bool:
    """Return True if *name* is a local directory containing a LoRA adapter.

    A path is considered an adapter path if:
    1. It's an existing directory with adapter_config.json, OR
    2. It looks like a checkpoint path (contains /outputs/checkpoints/) even if
       the directory doesn't exist yet (e.g., when passed as an Azure mount path).
    """
    p = Path(name)

    # Existing directory with adapter_config.json
    if p.is_dir() and (p / "adapter_config.json").exists():
        print(
            f"[adapter] Path recognized as adapter (has adapter_config.json): {name}",
            flush=True,
        )
        return True

    # Looks like a checkpoint path from the registry
    # (e.g., "Daily_rainfall_sample/outputs/checkpoints/...")
    is_checkpoint_pattern = "/outputs/checkpoints/" in str(
        name
    ) or "\\outputs\\checkpoints\\" in str(name)
    if is_checkpoint_pattern:
        print(
            f"[adapter] Path recognized as checkpoint (matches pattern): {name}",
            flush=True,
        )
        return True

    return False


def _gpu_dtype() -> "torch.dtype":
    """Return the best floating-point dtype for the available GPU.

    * ``bfloat16`` — Ampere and newer (compute capability >= 8.0, e.g. A100)
    * ``float16``  — older CUDA GPUs (e.g. V100, T4) that lack bfloat16 support
    * ``float32``  — CPU fallback (also used when CUDA is unavailable)
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _log_device_info() -> None:
    """Print GPU/CUDA diagnostics to stdout for AML job logs."""
    import torch

    print("[device] torch version:", torch.__version__)
    print("[device] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[device] CUDA device count: {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(
                f"[device]   GPU {i}: {props.name}  "
                f"CC={props.major}.{props.minor}  "
                f"VRAM={props.total_memory // 1024**3}GB"
            )
        print("[device] dtype selected:", _gpu_dtype())
    else:
        print("[device] WARNING: no CUDA device found — running on CPU")
        # Print CUDA version the build was compiled against (may differ from driver)
        print("[device] CUDA build version:", torch.version.cuda)
    print(flush=True)


def _local_hf_home() -> Path | None:
    """Return a writable local HF_HOME path, creating it if necessary.

    On Azure ML (when HF_HOME is a mounted datastore), use it directly since
    it persists across jobs and is shared with other jobs.  Do not use /tmp.

    On local systems (HF_HOME is a regular directory), also use HF_HOME directly
    for simplicity.  The /tmp optimization is only applied when explicitly needed.

    Returns None if HF_HOME is not set (HuggingFace uses its default cache).
    """
    import os

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None

    hf_path = Path(hf_home)

    # If HF_HOME is already a mounted datastore path (e.g., Azure ML mount),
    # use it directly — it persists across jobs.  Don't create /tmp cache.
    if (
        hf_home.startswith("/mnt/")
        or hf_home.startswith("${{")
        or "azureml" in hf_home.lower()
    ):
        print(f"[cache] Using Azure datastore cache directly: {hf_home}", flush=True)
        hf_path.mkdir(parents=True, exist_ok=True)
        return hf_path

    # For local or non-mount paths, use HF_HOME directly as well
    # (avoids complexity of /tmp cache coordination)
    hf_path.mkdir(parents=True, exist_ok=True)
    print(f"[cache] Using HF_HOME directly: {hf_home}", flush=True)
    return hf_path


def _resolve_model_path(model_name: str) -> str:
    """Return the local snapshot path for *model_name*, copying from blob store if needed.

    Strategy (avoids copying the entire HF cache to local disk):

    1. Check whether the model is already in the *local* HF cache
       (``/tmp/hf_cache``).  If yes, return that path directly.
    2. If not cached locally, check whether it is in the *remote* blob-store
       cache (the original ``HF_HOME``).  If yes, copy only that model's
       snapshot directory to the local cache, then return the local path.
    3. If not cached anywhere, return *model_name* so that transformers
       downloads it from HuggingFace (and writes to the local cache).
    """
    import os
    import shutil
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    model_name = _normalize_model_reference(model_name)
    if Path(model_name).exists():
        return model_name

    def _snapshot_is_complete(snapshot_path: Path) -> tuple[bool, str]:
        """Return (ok, reason) for a cached HF snapshot directory."""
        cfg = snapshot_path / "config.json"
        if not cfg.exists():
            return False, "missing config.json"

        broken = [
            p for p in snapshot_path.rglob("*") if p.is_symlink() and not p.exists()
        ]
        if broken:
            return False, f"{len(broken)} broken symlink(s)"

        try:
            cfg_obj = json.loads(cfg.read_text())
        except Exception:
            return False, "unreadable config.json"

        # AutoConfig requires model_type for model class dispatch.
        if not isinstance(cfg_obj, dict) or "model_type" not in cfg_obj:
            return False, "config.json missing model_type"

        return True, "ok"

    # Step 1: already in local cache?
    try:
        local_path = snapshot_download(model_name, local_files_only=True)
        snapshot_p = Path(local_path)
        ok, reason = _snapshot_is_complete(snapshot_p)
        if not ok:
            print(
                f"[cache] WARNING — local snapshot for {model_name} is incomplete "
                f"({reason}); treating as a cache miss and re-downloading.",
                flush=True,
            )
            # Remove the bad snapshot so a forced re-download does not keep
            # resolving to the same incomplete path.
            try:
                shutil.rmtree(snapshot_p, ignore_errors=True)
            except Exception:
                pass
            raise LocalEntryNotFoundError(local_path)

        size_mb = (
            sum(f.stat().st_size for f in snapshot_p.rglob("*") if f.is_file()) / 1e6
        )
        print(
            f"[cache] CACHE HIT (local) — {model_name}\n"
            f"        {local_path}  ({size_mb:.0f} MB)",
            flush=True,
        )
        return local_path
    except LocalEntryNotFoundError:
        pass

    # Step 2: in the remote blob-store cache?
    original_hf_home = os.environ.get("_ORIGINAL_HF_HOME")
    if original_hf_home:
        try:
            remote_path = snapshot_download(
                model_name,
                local_files_only=True,
                cache_dir=original_hf_home,
            )
            snapshot_p = Path(remote_path)
            # Validate the snapshot is complete before using it. A previous job
            # that crashed mid-download can leave a partial snapshot directory.
            ok, reason = _snapshot_is_complete(snapshot_p)
            if not ok:
                print(
                    f"[cache] WARNING — blob snapshot for {model_name} is incomplete "
                    f"({reason}); treating as a cache miss "
                    f"and re-downloading.",
                    flush=True,
                )
                raise LocalEntryNotFoundError(remote_path)
            # Copy only this model's snapshot to local cache.
            local_cache_dir = Path("/tmp/hf_cache")
            # Preserve the relative sub-path (hub/models--org--name/snapshots/...)
            rel = Path(remote_path).relative_to(Path(original_hf_home))
            local_dest = local_cache_dir / rel
            if not local_dest.exists():
                print(
                    f"[cache] CACHE HIT (blob) — copying {model_name} snapshot "
                    f"to local disk…",
                    flush=True,
                )
                t0 = time.monotonic()
                try:
                    shutil.copytree(remote_path, local_dest)
                except Exception as exc:
                    # Partial/corrupt snapshot on the blob store: some blob files
                    # referenced by symlinks in the snapshot dir are missing.
                    # Clean up the incomplete local copy and fall through to a
                    # fresh download from HuggingFace.
                    print(
                        f"[cache] WARNING — blob snapshot copy failed ({exc}); "
                        f"discarding partial copy and re-downloading.",
                        flush=True,
                    )
                    if local_dest.exists():
                        shutil.rmtree(local_dest, ignore_errors=True)
                    raise LocalEntryNotFoundError(remote_path) from exc
                elapsed = time.monotonic() - t0
                size_mb = (
                    sum(f.stat().st_size for f in local_dest.rglob("*") if f.is_file())
                    / 1e6
                )
                print(
                    f"[cache] Copied {size_mb:.0f} MB in {elapsed:.1f}s → {local_dest}",
                    flush=True,
                )
            else:
                print(
                    f"[cache] CACHE HIT (local, already copied) — {model_name}",
                    flush=True,
                )
            return str(local_dest)
        except LocalEntryNotFoundError:
            pass

    # Step 3: not cached anywhere — download from HuggingFace.
    print(
        f"[cache] CACHE MISS — {model_name} not cached; downloading from HuggingFace.",
        flush=True,
    )
    return model_name


def _uses_causal_lm(family: str) -> bool:
    """Return True for families that require ``AutoModelForCausalLM``.

    Gemma 4 uses ``AutoModelForCausalLM`` rather than
    ``AutoModelForImageTextToText``.
    """
    return family == "gemma4"


def _load_processor_with_fallback(
    auto_processor_cls: Any,
    model_ref: str,
    resolved_ref: str,
    proc_kwargs: dict[str, Any],
) -> Any:
    """Load an AutoProcessor, retrying from model ID if a cached snapshot is bad.

    When ``resolved_ref`` is a local snapshot path, a partial cache can trigger
    ``ValueError: Unrecognized processing class ...``. In that case we retry
    from ``model_ref`` with ``force_download=True`` so transformers can fetch
    any missing processor metadata.
    """
    try:
        return auto_processor_cls.from_pretrained(resolved_ref, **proc_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "Unrecognized processing class" not in msg or resolved_ref == model_ref:
            raise

        print(
            "[cache] WARNING — cached snapshot is missing processor metadata; "
            "retrying processor load from model ID with force_download=True.",
            flush=True,
        )
        retry_kwargs = dict(proc_kwargs)
        retry_kwargs["force_download"] = True
        return auto_processor_cls.from_pretrained(model_ref, **retry_kwargs)


def _load_model_with_fallback(
    auto_model_cls: Any,
    model_ref: str,
    resolved_ref: str,
    model_kwargs: dict[str, Any],
) -> Any:
    """Load an AutoModel, retrying from model ID when cached config is invalid.

    A partial snapshot can contain a truncated ``config.json`` without
    ``model_type``. In that case, retrying with ``force_download=True`` ensures
    a fresh config is fetched.
    """
    try:
        return auto_model_cls.from_pretrained(resolved_ref, **model_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if (
            "Unrecognized model in" not in msg
            or "model_type" not in msg
            or resolved_ref == model_ref
        ):
            raise

        print(
            "[cache] WARNING — cached snapshot has invalid model config; "
            "retrying model load from model ID with force_download=True.",
            flush=True,
        )
        retry_kwargs = dict(model_kwargs)
        retry_kwargs["force_download"] = True
        return auto_model_cls.from_pretrained(model_ref, **retry_kwargs)


def _load_model_and_processor(config: ModelConfig):  # type: ignore[return]
    """Load the processor and model from HuggingFace (or a local LoRA adapter).

    If *config.model_name* points to a local directory that contains
    ``adapter_config.json``, the base model is read from the adapter config
    and the LoRA weights are applied via ``peft.PeftModel``.

    Raises ``ImportError`` if the ``train`` extras are not installed.
    """
    import os

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: " "pip install -e '.[train]'"
        ) from exc

    _log_device_info()

    # Point HF_HOME at a local writable directory to avoid slow blob-mount reads.
    # Save the original blob-store path as _ORIGINAL_HF_HOME so _resolve_model_path
    # can copy only the required model snapshot across (not the whole cache).
    original_hf_home = os.environ.get("HF_HOME")
    local_cache = _local_hf_home()
    if local_cache is not None:
        os.environ["HF_HOME"] = str(local_cache)
        # huggingface_hub reads HF_HOME once at import time into the module-level
        # constant HUGGINGFACE_HUB_CACHE.  Setting os.environ afterwards has no
        # effect on that constant.  Patch it directly so snapshot_download and
        # from_pretrained use the local path even if the library was imported
        # before this function ran.
        try:
            import huggingface_hub.constants as _hfc

            _hfc.HF_HOME = str(local_cache)
            _hfc.HUGGINGFACE_HUB_CACHE = str(local_cache / "hub")
            _hfc.HF_HUB_CACHE = str(local_cache / "hub")
        except Exception:
            pass
        if original_hf_home and Path(original_hf_home) != local_cache:
            os.environ["_ORIGINAL_HF_HOME"] = original_hf_home
        print(f"[cache] HF_HOME set to local cache: {local_cache}", flush=True)

    # Cache dir to pass explicitly to from_pretrained — bypasses any stale
    # module-level constant that slipped through the patch above.
    hf_cache_dir: str | None = str(local_cache / "hub") if local_cache else None

    family = detect_model_family(config.model_name)
    use_causal = _uses_causal_lm(family)
    model_cls = AutoModelForCausalLM if use_causal else AutoModelForImageTextToText

    # granite4 has native transformers 5.8+ support; passing trust_remote_code=True
    # would load the stale custom modeling.py from hf_cache which lacks the
    # cache_position argument that transformers 5.x passes to create_causal_mask.
    extra_kwargs: dict[str, Any] = (
        {} if family == "granite4" else {"trust_remote_code": True}
    )
    if hf_cache_dir:
        extra_kwargs["cache_dir"] = hf_cache_dir

    model_device_map: str | None = config.device

    print(
        f"[load_model] Attempting to load: {config.model_name}",
        flush=True,
    )

    if _is_adapter_path(config.model_name):
        import json as _json

        from peft import PeftModel

        adapter_dir = Path(config.model_name)
        adapter_cfg_path = adapter_dir / "adapter_config.json"

        if not adapter_cfg_path.exists():
            print(
                f"[adapter] WARNING: adapter_config.json not found at {adapter_cfg_path}; "
                f"treating {config.model_name} as a base model name.",
                flush=True,
            )
            # Fall through to base model loading below
        else:
            try:
                adapter_cfg = _json.loads(adapter_cfg_path.read_text())
                base_model_name_raw = str(adapter_cfg["base_model_name_or_path"])
                base_model_name = _normalize_model_reference(base_model_name_raw)
                print(
                    f"[adapter] Loading checkpoint from {config.model_name}\n"
                    f"           Base model: {base_model_name}",
                    flush=True,
                )
                resolved_name = _resolve_model_path(base_model_name)

                proc_kwargs: dict[str, Any] = (
                    {} if family == "granite4" else {"trust_remote_code": True}
                )
                if hf_cache_dir:
                    proc_kwargs["cache_dir"] = hf_cache_dir
                processor = _load_processor_with_fallback(
                    AutoProcessor,
                    base_model_name,
                    resolved_name,
                    proc_kwargs,
                )
                model_kwargs: dict[str, Any] = {
                    "torch_dtype": _gpu_dtype(),
                    "device_map": model_device_map,
                    **extra_kwargs,
                }
                base = _load_model_with_fallback(
                    model_cls,
                    base_model_name,
                    resolved_name,
                    model_kwargs,
                )
                model = PeftModel.from_pretrained(base, str(adapter_dir))
                print(
                    f"[adapter] LoRA adapter loaded successfully.",
                    flush=True,
                )

                model.eval()
                _sync_cache_to_remote(local_cache)
                return processor, model
            except Exception as exc:
                print(
                    f"[adapter] WARNING: failed to load adapter ({exc}); "
                    f"falling back to base model loading.",
                    flush=True,
                )
                # Fall through to base model loading below

    # Base model loading (either no adapter was provided, or adapter loading failed)
    if _is_adapter_path(config.model_name):
        # This was marked as an adapter path but couldn't be loaded; try as a base model
        print(
            f"[model] **CRITICAL**: Checkpoint path provided but adapter not found/loaded!"
            f"\n         Path: {config.model_name}"
            f"\n         Falling back to base model - extraction will use UNTRAINED model!",
            flush=True,
        )

    resolved_name = _resolve_model_path(config.model_name)
    proc_kwargs: dict[str, Any] = (
        {} if family == "granite4" else {"trust_remote_code": True}
    )
    if hf_cache_dir:
        proc_kwargs["cache_dir"] = hf_cache_dir
    processor = _load_processor_with_fallback(
        AutoProcessor,
        config.model_name,
        resolved_name,
        proc_kwargs,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _gpu_dtype(),
        "device_map": model_device_map,
        **extra_kwargs,
    }
    model = _load_model_with_fallback(
        model_cls,
        config.model_name,
        resolved_name,
        model_kwargs,
    )

    model.eval()

    # Write any newly-downloaded files back to the persistent blob-store cache.
    _sync_cache_to_remote(local_cache)

    return processor, model


def _sync_cache_to_remote(local_cache: "Path | None") -> None:
    """Copy new files from *local_cache* back to the original HF_HOME mount.

    Only runs if HF_HOME was originally a different path (i.e. a blob mount).
    Skips files that already exist at the destination to avoid redundant I/O.
    """
    import os
    import shutil

    original_hf_home = os.environ.get("_ORIGINAL_HF_HOME")
    if local_cache is None or not original_hf_home:
        return

    dst = Path(original_hf_home)
    new_files = 0
    for src_file in local_cache.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(local_cache)
        dst_file = dst / rel
        if not dst_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            new_files += 1
    if new_files:
        print(f"[cache] Wrote {new_files} new file(s) back to {dst}", flush=True)
    else:
        print("[cache] No new files to sync back to remote cache.", flush=True)


def extract_grid_batch_with_model(
    image_paths: list[Path],
    config: ModelConfig,
    processor: Any,
    model: Any,
    family: str,
) -> list[tuple[DailyRainfallGrid | None, str]]:
    """Run inference on a batch of images in a single forward pass.

    Processing images together makes better use of GPU parallelism compared
    to calling :func:`extract_grid_with_model` in a loop.

    Parameters
    ----------
    image_paths:
        List of paths to document images.  All are processed together.
    config:
        Model generation parameters.
    processor:
        HuggingFace processor already loaded for this model.
    model:
        HuggingFace model already loaded and on the target device.
    family:
        Model family string from :func:`detect_model_family`.

    Returns
    -------
    list of (grid, raw_text) pairs, one per input image, in order.
    """
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: pip install -e '.[train]'"
        ) from exc

    do_sample = config.temperature > 0.0
    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        generate_kwargs["temperature"] = config.temperature

    images = [PILImage.open(p).convert("RGB") for p in image_paths]

    if family.startswith("granite") or family == "gemma3":
        # Granite 4 / Gemma 3: batch by passing one conversation per sample,
        # each with its own embedded PIL image.
        messages_batch = [
            build_messages(p, model_family=family, pil_image=img)
            for p, img in zip(image_paths, images)
        ]
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if family == "gemma3":
            template_kwargs["do_pan_and_scan"] = True

        inputs = processor.apply_chat_template(messages_batch, **template_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        raw_texts: list[str] = processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )
        return [(parse_extraction_response(raw), raw) for raw in raw_texts]

    if family == "gemma4":
        # Gemma 4: batch text prompts and provide one image list per prompt.
        text_prompts = [
            processor.apply_chat_template(
                build_messages(p, model_family=family),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in image_paths
        ]
        inputs = processor(
            text=text_prompts,
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        raw_texts: list[str] = processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )
        return [(parse_extraction_response(raw), raw) for raw in raw_texts]

    # SmolVLM / generic: batch text prompts and pass one image per sample.
    text_prompts = [
        processor.apply_chat_template(
            build_messages(p, model_family=family),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in image_paths
    ]
    # For SmolVLM, images must be nested per prompt: [[img1], [img2], ...]
    inputs = processor(
        text=text_prompts,
        images=[[img] for img in images],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    # Inputs are padded to a uniform length; strip that prefix from outputs.
    input_len = inputs["input_ids"].shape[1]
    raw_texts: list[str] = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )
    return [(parse_extraction_response(raw), raw) for raw in raw_texts]


def extract_grid(
    image_path: Path,
    config: ModelConfig,
) -> tuple[DailyRainfallGrid | None, str]:
    """Run the configured VLM over *image_path* and return ``(grid, raw_text)``.

    Loads the model and processor on every call.  When processing many images,
    use :func:`extract_grid_with_model` to load once and reuse.

    *grid* is ``None`` when the model response cannot be parsed.

    Parameters
    ----------
    image_path:
        Path to the document image (JPEG, PNG, …).
    config:
        :class:`~weather_doc_extractor.config.ModelConfig` controlling which
        model to load and its generation parameters.
    """
    processor, model = _load_model_and_processor(config)
    family = detect_model_family(config.model_name)
    return extract_grid_with_model(image_path, config, processor, model, family)


def extract_grid_with_model(
    image_path: Path,
    config: ModelConfig,
    processor: Any,
    model: Any,
    family: str,
) -> tuple[DailyRainfallGrid | None, str]:
    """Run inference using an already-loaded *model* and *processor*.

    Use this in batch loops to avoid reloading weights for every image.
    Obtain *processor*, *model*, and *family* once via::

        processor, model = _load_model_and_processor(config)
        family = detect_model_family(config.model_name)

    Parameters
    ----------
    image_path:
        Path to the document image.
    config:
        Model generation parameters (``max_new_tokens``, ``temperature``).
    processor:
        HuggingFace processor already loaded for this model.
    model:
        HuggingFace model already loaded and on the target device.
    family:
        Model family string from :func:`detect_model_family`.
    """
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: " "pip install -e '.[train]'"
        ) from exc

    image = PILImage.open(image_path).convert("RGB")
    messages = build_messages(image_path, model_family=family, pil_image=image)

    do_sample = config.temperature > 0.0
    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        generate_kwargs["temperature"] = config.temperature

    if family.startswith("granite") or family == "gemma3":
        # These families embed the image directly in the message; the processor
        # tokenises everything in one apply_chat_template call.
        # Gemma 3: pass do_pan_and_scan=True so the processor tiles wide/tall
        # scans into patches, preventing detail loss at fixed 896×896 encoding.
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if family == "gemma3":
            template_kwargs["do_pan_and_scan"] = True
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(
            model.device
        )
    elif family == "gemma4":
        # Gemma 4: two-step — tokenise text, then pass PIL image separately.
        # disable_thinking avoids outputting reasoning tokens.
        text_prompt: str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # SmolVLM / generic
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[1]
    raw_text: str = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )[0]

    grid = parse_extraction_response(raw_text)
    return grid, raw_text
