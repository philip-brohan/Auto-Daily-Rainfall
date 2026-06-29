"""Tests for consensus variant functionality.

ENVIRONMENT: Run tests in the weather-doc-extractor conda environment:
  conda activate weather-doc-extractor
  pytest tests/test_consensus.py

Tests the multi-consensus dataset support, including:
- Config file creation and validation
- Variant directory structure
- Backwards compatibility with old-style single consensus
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset structure with images and dummy extractions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "consensus_dataset_1000"
        root.mkdir()

        # Create images/ and transcriptions/
        images_dir = root / "images"
        images_dir.mkdir()
        transcriptions_dir = root / "transcriptions"
        transcriptions_dir.mkdir()

        # Create dummy test images (empty files)
        for i in range(5):
            (images_dir / f"DRain_1871-1880_Test-{i}.jpg").touch()
            (transcriptions_dir / f"DRain_1871-1880_Test-{i}.json").write_text(
                json.dumps(
                    {f"Day {d}": [None] * 12 for d in range(1, 32)}
                    | {"Totals": [None] * 12}
                )
            )

        # Create dummy extraction directories with consistent output
        extractions_dir = root.parent / "extractions"
        extractions_dir.mkdir()

        model_names = [
            "model_A",
            "model_B",
            "model_C",
            "model_D",
            "model_E",
        ]
        extraction_dirs = []

        for model_name in model_names:
            model_dir = extractions_dir / model_name / "20260601-120000"
            model_dir.mkdir(parents=True)
            extraction_dirs.append(str(model_dir))

            # Create extraction JSONs with consistent values for testing
            for i in range(5):
                stem = f"DRain_1871-1880_Test-{i}"
                extraction_file = model_dir / f"{stem}.json"
                extraction_data = {
                    "grid": {
                        "days": {
                            f"Day {d}": [1.5 + 0.1 * ((d + i) % 12)] * 12
                            for d in range(1, 32)
                        },
                        "totals": [25.0 + 5 * ((i) % 5)] * 12,
                    }
                }
                extraction_file.write_text(json.dumps(extraction_data))

        yield root, extraction_dirs


class TestConfigCreation:
    """Test consensus config file creation."""

    def test_create_config_basic(self, temp_dataset):
        """Test creating a basic config file."""
        dataset_root, extraction_dirs = temp_dataset
        script_dir = Path(__file__).parent.parent / "scripts"

        result = subprocess.run(
            [
                "python3",
                str(script_dir / "create_consensus_config.py"),
                "--variant-name",
                "consensus_1000",
                "--dataset-root",
                str(dataset_root),
                "--agreement-threshold",
                "4",
                "--precision",
                "3",
                "--extraction-dirs",
                *extraction_dirs,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify config file was created
        config_file = dataset_root / "consensus_1000" / "consensus_config.json"
        assert config_file.exists()

        # Verify config contents
        config_data = json.loads(config_file.read_text())
        assert config_data["variant_name"] == "consensus_1000"
        assert config_data["agreement_threshold"] == 4
        assert config_data["null_threshold"] == 4
        assert config_data["precision"] == 3
        assert len(config_data["extraction_dirs"]) == 5

    def test_create_config_overwrite_protection(self, temp_dataset):
        """Test that overwriting requires --overwrite flag."""
        dataset_root, extraction_dirs = temp_dataset
        script_dir = Path(__file__).parent.parent / "scripts"

        # Create first config
        subprocess.run(
            [
                "python3",
                str(script_dir / "create_consensus_config.py"),
                "--variant-name",
                "consensus_1000",
                "--dataset-root",
                str(dataset_root),
                "--agreement-threshold",
                "4",
                "--precision",
                "3",
                "--extraction-dirs",
                *extraction_dirs,
            ],
            check=True,
            capture_output=True,
        )

        # Try to create again without --overwrite (should fail)
        result = subprocess.run(
            [
                "python3",
                str(script_dir / "create_consensus_config.py"),
                "--variant-name",
                "consensus_1000",
                "--dataset-root",
                str(dataset_root),
                "--agreement-threshold",
                "4",
                "--precision",
                "3",
                "--extraction-dirs",
                *extraction_dirs,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

        # With --overwrite should succeed
        result = subprocess.run(
            [
                "python3",
                str(script_dir / "create_consensus_config.py"),
                "--variant-name",
                "consensus_1000",
                "--dataset-root",
                str(dataset_root),
                "--agreement-threshold",
                "4",
                "--precision",
                "3",
                "--overwrite",
                "--extraction-dirs",
                *extraction_dirs,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestConsensusBuilding:
    """Test consensus building with config files."""

    def test_consensus_with_config(self, temp_dataset):
        """Test building consensus using config file."""
        dataset_root, extraction_dirs = temp_dataset
        script_dir = Path(__file__).parent.parent / "scripts"

        # Create config
        subprocess.run(
            [
                "python3",
                str(script_dir / "create_consensus_config.py"),
                "--variant-name",
                "consensus_1000",
                "--dataset-root",
                str(dataset_root),
                "--agreement-threshold",
                "3",
                "--precision",
                "3",
                "--extraction-dirs",
                *extraction_dirs,
            ],
            check=True,
            capture_output=True,
        )

        # Build consensus using config
        config_file = dataset_root / "consensus_1000" / "consensus_config.json"
        result = subprocess.run(
            [
                "python3",
                str(script_dir / "build_consensus_transcriptions.py"),
                "--config-file",
                str(config_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify consensus output was created
        consensus_dir = dataset_root / "consensus_1000" / "consensus_transcriptions"
        assert consensus_dir.exists()
        consensus_files = list(consensus_dir.glob("*.json"))
        assert len(consensus_files) > 0

        # Verify summary file
        summary_file = dataset_root / "consensus_1000" / "consensus_summary.json"
        assert summary_file.exists()
        summary = json.loads(summary_file.read_text())
        assert summary["agreement_threshold"] == 3
        assert summary["null_threshold"] == 3
        assert summary["precision"] == 3

    def test_non_null_precedence_when_both_thresholds_pass(self):
        """If non-null and null both meet thresholds, non-null consensus wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            script_dir = Path(__file__).parent.parent / "scripts"

            input_dirs = []
            stem = "DRain_1871-1880_Test-0"
            votes = [1.26, 1.26, None, None, None, None]

            for idx, vote in enumerate(votes):
                d = tmp / "extractions" / f"model_{idx}" / "20260601-120000"
                d.mkdir(parents=True)
                input_dirs.append(d)

                grid = {
                    "days": {f"Day {day}": [vote] * 12 for day in range(1, 32)},
                    "totals": [vote] * 12,
                }
                payload = {"parse_failed": False, "grid": grid}
                (d / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")

            output_dir = tmp / "consensus_out"
            summary_file = tmp / "summary.json"

            cmd = [
                "python3",
                str(script_dir / "build_consensus_transcriptions.py"),
                "--output-dir",
                str(output_dir),
                "--agreement-threshold",
                "2",
                "--null-threshold",
                "4",
                "--precision",
                "3",
                "--summary-file",
                str(summary_file),
            ]
            for d in input_dirs:
                cmd.extend(["--input-dir", str(d)])

            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Script failed: {result.stderr}"

            out = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
            cell = out["Day 1"][0]

            assert cell["value"] == 1.26
            assert cell["correct"] is True

    def test_consensus_backwards_compat_cli(self, temp_dataset):
        """Test that old CLI (5x --input-dir) still works."""
        dataset_root, extraction_dirs = temp_dataset
        script_dir = Path(__file__).parent.parent / "scripts"

        output_dir = dataset_root / "old_consensus_output"
        result = subprocess.run(
            [
                "python3",
                str(script_dir / "build_consensus_transcriptions.py"),
                "--agreement-threshold",
                "3",
                "--precision",
                "3",
                "--output-dir",
                str(output_dir),
                *[f"--input-dir={d}" for d in extraction_dirs],
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.json"))) > 0


class TestVariantDirectoryStructure:
    """Test variant directory creation and structure."""

    def test_prepare_dataset_with_variant_name(self, temp_dataset):
        """Test prepare_consensus_dataset.py with explicit variant name."""
        dataset_root, _ = temp_dataset
        script_dir = Path(__file__).parent.parent / "scripts"

        # Create sample manifest
        manifest_file = dataset_root / "sample_manifest.csv"
        manifest_file.write_text(
            "stem\nDRain_1871-1880_Test-0\nDRain_1871-1880_Test-1\n",
            encoding="utf-8",
        )

        # Run prepare script with variant name
        result = subprocess.run(
            [
                "python3",
                str(script_dir / "prepare_consensus_dataset.py"),
                "--manifest-csv",
                str(manifest_file),
                "--dataset-root",
                str(dataset_root),
                "--variant-name",
                "consensus_1000_v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify variant directory structure
        variant_dir = dataset_root / "consensus_1000_v2"
        assert variant_dir.exists()
        assert (variant_dir / "transcriptions").exists()
        assert len(list((variant_dir / "transcriptions").glob("*.json"))) > 0


class TestBackwardsCompatibility:
    """Test backwards compatibility with old single-consensus workflows."""

    def test_prepare_dataset_auto_variant_name(self, temp_dataset):
        """Test that auto-derived variant name creates both new and old paths."""
        dataset_root, _ = temp_dataset
        script_dir = Path(__file__).parent.parent / "scripts"

        # Create sample manifest
        manifest_file = dataset_root / "sample_manifest.csv"
        manifest_file.write_text(
            "stem\nDRain_1871-1880_Test-0\nDRain_1871-1880_Test-1\n",
            encoding="utf-8",
        )

        # Run prepare script WITHOUT variant name (auto-derive)
        result = subprocess.run(
            [
                "python3",
                str(script_dir / "prepare_consensus_dataset.py"),
                "--manifest-csv",
                str(manifest_file),
                "--dataset-root",
                str(dataset_root),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify BOTH old path and new path were created (compat mode)
        old_transcriptions = dataset_root / "transcriptions"
        new_transcriptions = dataset_root / "consensus_1000" / "transcriptions"

        assert old_transcriptions.exists(), "Old path should exist in compat mode"
        assert new_transcriptions.exists(), "New path should exist"
        assert len(list(old_transcriptions.glob("*.json"))) > 0
        assert len(list(new_transcriptions.glob("*.json"))) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
