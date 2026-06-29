import json
import tempfile
import unittest
from pathlib import Path

from weather_doc_extractor.inference import (
    EXTRACTION_PROMPT,
    _extract_object,
    _extract_hf_model_id_from_snapshot_path,
    _load_model_with_fallback,
    _load_processor_with_fallback,
    _resolve_model_path,
    build_messages,
    detect_model_family,
    parse_extraction_response,
)
from weather_doc_extractor.schemas import DailyRainfallGrid

MONTHS = 12
_ALL_DAYS = [f"Day {i}" for i in range(1, 32)]


def _make_grid_json(day_value: float | None = 0.10, total: float = 1.0) -> str:
    data: dict[str, list] = {"Totals": [total] * MONTHS}
    for day in _ALL_DAYS:
        data[day] = [day_value] * MONTHS
    return json.dumps(data)


def _make_truncated_json(stop_at_day: int) -> str:
    """Build a JSON string that is cut off mid-way through the given day key."""
    lines = ["{"]
    for i in range(1, stop_at_day + 1):
        row = json.dumps([0.05] * MONTHS)
        lines.append(f'  "Day {i}": {row},')
    # Truncate after opening the next array
    lines.append(f'  "Day {stop_at_day + 1}": [0.05, 0.12')
    return "\n".join(lines)


class ParseExtractionResponseTests(unittest.TestCase):
    def test_clean_json_parses(self) -> None:
        raw = _make_grid_json(0.12)
        grid = parse_extraction_response(raw)
        self.assertIsInstance(grid, DailyRainfallGrid)
        self.assertEqual(grid.days["Day 1"][0], 0.12)
        self.assertEqual(len(grid.totals), MONTHS)

    def test_markdown_fence_stripped(self) -> None:
        raw = "```json\n" + _make_grid_json(0.05) + "\n```"
        grid = parse_extraction_response(raw)
        self.assertIsNotNone(grid)

    def test_prose_before_json(self) -> None:
        raw = "Sure! Here is the extracted data:\n\n" + _make_grid_json(0.20)
        grid = parse_extraction_response(raw)
        self.assertIsNotNone(grid)

    def test_null_values(self) -> None:
        raw = _make_grid_json(day_value=None)
        grid = parse_extraction_response(raw)
        assert grid is not None
        self.assertTrue(all(v is None for v in grid.days["Day 1"]))

    def test_string_number_coerced(self) -> None:
        data: dict[str, list] = {"Totals": ["1.5"] * MONTHS}
        for day in _ALL_DAYS:
            data[day] = ["0.10"] * MONTHS
        grid = parse_extraction_response(json.dumps(data))
        assert grid is not None
        self.assertAlmostEqual(grid.totals[0], 1.5)
        self.assertAlmostEqual(grid.days["Day 1"][0], 0.10)

    def test_missing_days_padded(self) -> None:
        """If the model only returns some day rows, missing ones are filled with None."""
        data = {"Totals": [1.0] * MONTHS, "Day 1": [0.05] * MONTHS}
        grid = parse_extraction_response(json.dumps(data))
        assert grid is not None
        self.assertEqual(len(grid.days), 31)
        self.assertTrue(all(v is None for v in grid.days["Day 2"]))

    def test_short_row_padded_to_12(self) -> None:
        data = {"Totals": [1.0] * MONTHS}
        for day in _ALL_DAYS:
            data[day] = [0.10] * 10  # only 10 months
        grid = parse_extraction_response(json.dumps(data))
        assert grid is not None
        self.assertEqual(len(grid.days["Day 1"]), MONTHS)
        self.assertIsNone(grid.days["Day 1"][11])

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(parse_extraction_response(""))

    def test_no_json_returns_none(self) -> None:
        self.assertIsNone(parse_extraction_response("Sorry, I cannot read this image."))

    def test_invalid_json_returns_none(self) -> None:
        self.assertIsNone(parse_extraction_response("{bad json:::"))

    def test_json_with_no_day_keys_returns_none(self) -> None:
        raw = json.dumps({"some_other_key": [1, 2, 3]})
        self.assertIsNone(parse_extraction_response(raw))

    def test_totals_case_insensitive(self) -> None:
        data: dict[str, list] = {"totals": [2.0] * MONTHS}
        for day in _ALL_DAYS:
            data[day] = [0.10] * MONTHS
        grid = parse_extraction_response(json.dumps(data))
        assert grid is not None
        self.assertAlmostEqual(grid.totals[0], 2.0)

    def test_all_31_days_present(self) -> None:
        grid = parse_extraction_response(_make_grid_json(0.07))
        assert grid is not None
        self.assertEqual(len(grid.days), 31)
        for day in _ALL_DAYS:
            self.assertIn(day, grid.days)

    def test_array_wrapped_json_parses(self) -> None:
        """Granite wraps its output in ``[{...}]``. The parser must unwrap it."""
        inner = json.loads(_make_grid_json(0.15))
        wrapped = json.dumps([inner])  # produces "[{...}]"
        grid = parse_extraction_response(wrapped)
        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertAlmostEqual(grid.days["Day 1"][0], 0.15)

    def test_array_wrapped_with_trailing_bracket(self) -> None:
        """Raw text ``{...}\\n]`` (partial array close) should still parse."""
        obj_str = _make_grid_json(0.20)
        raw = obj_str + "\n]"
        grid = parse_extraction_response(raw)
        self.assertIsNotNone(grid)

        """A JSON object cut off mid-stream should be repaired and parsed."""
        truncated = _make_truncated_json(stop_at_day=15)
        grid = parse_extraction_response(truncated)
        self.assertIsNotNone(grid)
        assert grid is not None
        # Days that were present before truncation should be parsed
        for i in range(1, 16):
            self.assertIn(f"Day {i}", grid.days)
        # Missing days filled with None
        self.assertTrue(all(v is None for v in grid.days["Day 31"]))

    def test_truncated_json_missing_closing_brace(self) -> None:
        """Complete arrays but missing closing ``}`` should still parse."""
        data = {"Day 1": [0.10] * MONTHS, "Day 2": [0.20] * MONTHS}
        incomplete = json.dumps(data)[:-1]  # remove trailing }
        grid = parse_extraction_response(incomplete)
        self.assertIsNotNone(grid)


class ExtractObjectTests(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(_extract_object('{"a": 1}'), '{"a": 1}')

    def test_array_wrapped(self) -> None:
        result = _extract_object('[{"a": 1}]')
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.strip().endswith("}"))
        self.assertNotIn("]", result)

    def test_prose_before_object(self) -> None:
        result = _extract_object('Here you go: {"a": 1}')
        self.assertIsNotNone(result)

    def test_no_object_returns_none(self) -> None:
        self.assertIsNone(_extract_object("no braces here"))

    def test_smolvlm_detected(self) -> None:
        self.assertEqual(
            detect_model_family("HuggingFaceTB/SmolVLM-500M-Instruct"), "smolvlm"
        )

    def test_smolvlm_case_insensitive(self) -> None:
        self.assertEqual(detect_model_family("HuggingFaceTB/smolvlm-256m"), "smolvlm")

    def test_smolvlm2_detected(self) -> None:
        self.assertEqual(
            detect_model_family("HuggingFaceTB/SmolVLM2-2.2B-Instruct"), "smolvlm2"
        )

    def test_smolvlm2_case_insensitive(self) -> None:
        self.assertEqual(detect_model_family("HuggingFaceTB/smolvlm2-256m"), "smolvlm2")

    def test_idefics_detected_as_smolvlm(self) -> None:
        self.assertEqual(
            detect_model_family("HuggingFaceM4/Idefics3-8B-Llama3"), "smolvlm"
        )

    def test_granite_detected(self) -> None:
        self.assertEqual(
            detect_model_family("ibm-granite/granite-vision-3.2-2b"), "granite"
        )

    def test_granite_case_insensitive(self) -> None:
        self.assertEqual(
            detect_model_family("ibm-granite/Granite-Vision-3.2-2B"), "granite"
        )

    def test_granite4_detected(self) -> None:
        self.assertEqual(
            detect_model_family("ibm-granite/granite-vision-4.1-4b"), "granite4"
        )

    def test_granite4_case_insensitive(self) -> None:
        self.assertEqual(
            detect_model_family("ibm-granite/Granite-Vision-4.1-4B"), "granite4"
        )

    def test_gemma3_detected(self) -> None:
        self.assertEqual(detect_model_family("google/gemma-3-4b-it"), "gemma3")

    def test_gemma3_case_insensitive(self) -> None:
        self.assertEqual(detect_model_family("google/Gemma-3-27B-IT"), "gemma3")

    def test_gemma4_detected(self) -> None:
        self.assertEqual(detect_model_family("google/gemma-4-E4B-it"), "gemma4")

    def test_gemma4_not_confused_with_gemma3(self) -> None:
        """gemma-4 must be detected as gemma4, not gemma3."""
        self.assertEqual(detect_model_family("google/gemma-4-27B-it"), "gemma4")

    def test_ministral_detected(self) -> None:
        self.assertEqual(
            detect_model_family("mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
            "ministral",
        )

    def test_pixtral_detected_as_ministral(self) -> None:
        self.assertEqual(detect_model_family("mistralai/Pixtral-12B-2409"), "ministral")

    def test_unknown_returns_generic(self) -> None:
        self.assertEqual(detect_model_family("some-org/some-unknown-model"), "generic")

    def test_extract_hf_model_id_from_snapshot_path(self) -> None:
        snapshot = (
            "/mnt/azureml/cr/j/old/wd/hf_cache/hub/"
            "models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/"
            "snapshots/482adb537c021c86670beed01cd58990d01e72e4"
        )
        self.assertEqual(
            _extract_hf_model_id_from_snapshot_path(snapshot),
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        )

    def test_detect_model_family_from_adapter_with_stale_snapshot_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            adapter_cfg = Path(td) / "adapter_config.json"
            adapter_cfg.write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": (
                            "/mnt/azureml/cr/j/old/wd/hf_cache/hub/"
                            "models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/"
                            "snapshots/482adb537c021c86670beed01cd58990d01e72e4"
                        )
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(detect_model_family(td), "smolvlm2")

    def test_resolve_model_path_returns_existing_local_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(_resolve_model_path(td), td)


class BuildMessagesTests(unittest.TestCase):
    def test_smolvlm_returns_one_user_message(self) -> None:
        msgs = build_messages(model_family="smolvlm")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_smolvlm_image_item_has_no_path(self) -> None:
        """SmolVLM expects ``{"type": "image"}`` with no embedded path."""
        msgs = build_messages(model_family="smolvlm")
        image_items = [i for i in msgs[0]["content"] if i["type"] == "image"]
        self.assertEqual(len(image_items), 1)
        self.assertNotIn("url", image_items[0])

    def test_granite_embeds_image_url(self) -> None:
        from pathlib import Path

        p = Path("/data/images/DRain_1871-1880_Cornwall-59.jpg")
        msgs = build_messages(image_path=p, model_family="granite")
        image_items = [i for i in msgs[0]["content"] if i["type"] == "image"]
        self.assertEqual(len(image_items), 1)
        self.assertEqual(image_items[0]["url"], str(p))

    def test_granite4_embeds_image_url(self) -> None:
        from pathlib import Path

        p = Path("/data/images/DRain_1871-1880_Cornwall-59.jpg")
        msgs = build_messages(image_path=p, model_family="granite4")
        image_items = [i for i in msgs[0]["content"] if i["type"] == "image"]
        self.assertEqual(len(image_items), 1)
        self.assertEqual(image_items[0]["url"], str(p))

    def test_gemma3_embeds_pil_image(self) -> None:
        """Gemma 3 should include the PIL image object in the message content."""
        sentinel = object()  # stand-in for a PIL image
        msgs = build_messages(model_family="gemma3", pil_image=sentinel)
        image_items = [i for i in msgs[0]["content"] if i["type"] == "image"]
        self.assertEqual(len(image_items), 1)
        self.assertIs(image_items[0]["image"], sentinel)
        self.assertNotIn("url", image_items[0])

    def test_gemma4_uses_placeholder(self) -> None:
        """Gemma 4 uses the same placeholder convention as SmolVLM."""
        msgs = build_messages(model_family="gemma4")
        image_items = [i for i in msgs[0]["content"] if i["type"] == "image"]
        self.assertEqual(len(image_items), 1)
        self.assertNotIn("url", image_items[0])
        self.assertNotIn("image", image_items[0])

    def test_generic_falls_back_to_placeholder(self) -> None:
        msgs = build_messages(model_family="generic")
        image_items = [i for i in msgs[0]["content"] if i["type"] == "image"]
        self.assertNotIn("url", image_items[0])

    def test_text_contains_extraction_instructions(self) -> None:
        msgs = build_messages(model_family="smolvlm")
        text_items = [i for i in msgs[0]["content"] if i["type"] == "text"]
        self.assertEqual(len(text_items), 1)
        prompt_text = text_items[0]["text"]
        self.assertIn("Day 1", prompt_text)
        # All 12 months should be named explicitly
        for month in [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]:
            self.assertIn(month, prompt_text)


class ProcessorFallbackTests(unittest.TestCase):
    def test_retries_from_model_id_on_unrecognized_processing_class(self) -> None:
        class _FakeAutoProcessor:
            calls: list[tuple[str, dict]] = []

            @classmethod
            def from_pretrained(cls, name: str, **kwargs):  # type: ignore[no-untyped-def]
                cls.calls.append((name, dict(kwargs)))
                if name == "/tmp/bad-snapshot":
                    raise ValueError(
                        "Unrecognized processing class in /tmp/bad-snapshot"
                    )
                return {"loaded": name, "kwargs": kwargs}

        proc = _load_processor_with_fallback(
            _FakeAutoProcessor,
            "google/gemma-4-E4B-it",
            "/tmp/bad-snapshot",
            {"trust_remote_code": True, "cache_dir": "/tmp/hf_cache/hub"},
        )

        self.assertEqual(_FakeAutoProcessor.calls[0][0], "/tmp/bad-snapshot")
        self.assertEqual(_FakeAutoProcessor.calls[1][0], "google/gemma-4-E4B-it")
        self.assertTrue(_FakeAutoProcessor.calls[1][1].get("force_download"))
        self.assertEqual(proc["loaded"], "google/gemma-4-E4B-it")

    def test_does_not_retry_for_other_value_errors(self) -> None:
        class _FakeAutoProcessor:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs):  # type: ignore[no-untyped-def]
                raise ValueError("Some other processor error")

        with self.assertRaisesRegex(ValueError, "Some other processor error"):
            _load_processor_with_fallback(
                _FakeAutoProcessor,
                "google/gemma-4-E4B-it",
                "/tmp/bad-snapshot",
                {"trust_remote_code": True},
            )


class ModelFallbackTests(unittest.TestCase):
    def test_retries_from_model_id_on_missing_model_type(self) -> None:
        class _FakeAutoModel:
            calls: list[tuple[str, dict]] = []

            @classmethod
            def from_pretrained(cls, name: str, **kwargs):  # type: ignore[no-untyped-def]
                cls.calls.append((name, dict(kwargs)))
                if name == "/tmp/bad-snapshot":
                    raise ValueError(
                        "Unrecognized model in /tmp/bad-snapshot. "
                        "Should have a `model_type` key in its config.json."
                    )
                return {"loaded": name, "kwargs": kwargs}

        mdl = _load_model_with_fallback(
            _FakeAutoModel,
            "google/gemma-4-E4B-it",
            "/tmp/bad-snapshot",
            {"trust_remote_code": True, "cache_dir": "/tmp/hf_cache/hub"},
        )

        self.assertEqual(_FakeAutoModel.calls[0][0], "/tmp/bad-snapshot")
        self.assertEqual(_FakeAutoModel.calls[1][0], "google/gemma-4-E4B-it")
        self.assertTrue(_FakeAutoModel.calls[1][1].get("force_download"))
        self.assertEqual(mdl["loaded"], "google/gemma-4-E4B-it")

    def test_does_not_retry_for_other_model_value_errors(self) -> None:
        class _FakeAutoModel:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs):  # type: ignore[no-untyped-def]
                raise ValueError("Some other model error")

        with self.assertRaisesRegex(ValueError, "Some other model error"):
            _load_model_with_fallback(
                _FakeAutoModel,
                "google/gemma-4-E4B-it",
                "/tmp/bad-snapshot",
                {"trust_remote_code": True},
            )


class Granite4TrustRemoteCodeTests(unittest.TestCase):
    """Granite4 has native transformers 5.8+ support so must not get trust_remote_code=True.

    The stale custom modeling.py downloaded via trust_remote_code in older envs
    doesn't accept the cache_position kwarg that transformers 5.x passes to
    create_causal_mask, causing a TypeError at inference time.
    """

    def test_granite4_not_in_trust_remote_code_families(self) -> None:
        """granite4 must be excluded from the trust_remote_code=True blanket."""
        import inspect

        from weather_doc_extractor.inference import _load_model_and_processor

        src = inspect.getsource(_load_model_and_processor)
        # The conditional should exclude granite4 from trust_remote_code=True.
        # Verify the source contains the exclusion pattern.
        self.assertIn('family == "granite4"', src)
        # Verify we do NOT apply trust_remote_code unconditionally.
        self.assertNotIn(
            'extra_kwargs: dict[str, Any] = {"trust_remote_code": True}',
            src,
        )


if __name__ == "__main__":
    unittest.main()
