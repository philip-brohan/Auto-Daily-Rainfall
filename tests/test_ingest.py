import json
import tempfile
import unittest
from pathlib import Path

from weather_doc_extractor.ingest import load_grid, parse_stem, scan_records
from weather_doc_extractor.schemas import DailyRainfallGrid, DailyRainfallRecord

MONTHS = 12


class ParseStemTests(unittest.TestCase):
    def test_standard_stem(self) -> None:
        result = parse_stem("DRain_1871-1880_Cornwall-59")
        self.assertEqual(result["decade"], "1871-1880")
        self.assertEqual(result["county"], "Cornwall")
        self.assertEqual(result["station_id"], "59")

    def test_county_with_part_suffix(self) -> None:
        result = parse_stem("DRain_1881-1890_Kent_Part2-173")
        self.assertEqual(result["decade"], "1881-1890")
        self.assertEqual(result["county"], "Kent_Part2")
        self.assertEqual(result["station_id"], "173")

    def test_invalid_stem_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_stem("not-a-valid-stem")

    def test_missing_station_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_stem("DRain_1871-1880_Cornwall")

    def test_collision_suffix_is_ignored(self) -> None:
        result = parse_stem("DRain_1871-1880_Cornwall-59__deadbeef")
        self.assertEqual(result["decade"], "1871-1880")
        self.assertEqual(result["county"], "Cornwall")
        self.assertEqual(result["station_id"], "59")

    def test_collision_suffix_with_index_is_ignored(self) -> None:
        result = parse_stem("DRain_1871-1880_Cornwall-59__deadbeef_2")
        self.assertEqual(result["decade"], "1871-1880")
        self.assertEqual(result["county"], "Cornwall")
        self.assertEqual(result["station_id"], "59")


class LoadGridTests(unittest.TestCase):
    def _make_transcription(self, tmp_dir: Path) -> Path:
        data = {"Totals": ["1.0"] * MONTHS}
        for day in range(1, 32):
            data[f"Day {day}"] = ["null" if day % 3 == 0 else "0.10"] * MONTHS
        path = tmp_dir / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_transcription(Path(tmp))
            grid = load_grid(path)
        self.assertIsInstance(grid, DailyRainfallGrid)
        self.assertEqual(len(grid.totals), MONTHS)
        self.assertEqual(grid.totals[0], 1.0)

    def test_loads_day_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_transcription(Path(tmp))
            grid = load_grid(path)
        self.assertEqual(len(grid.days), 31)
        self.assertEqual(len(grid.days["Day 1"]), MONTHS)

    def test_null_values_become_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_transcription(Path(tmp))
            grid = load_grid(path)
        # Day 3 is all "null" (3 % 3 == 0)
        self.assertTrue(all(v is None for v in grid.days["Day 3"]))

    def test_numeric_values_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_transcription(Path(tmp))
            grid = load_grid(path)
        self.assertEqual(grid.days["Day 1"][0], 0.10)

    def test_bare_decimal_value(self) -> None:
        """Values like '.15' (no leading zero) should parse correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            data = {"Totals": [".15"] * MONTHS, "Day 1": [".15"] * MONTHS}
            path = Path(tmp) / "bare.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            grid = load_grid(path)
        self.assertAlmostEqual(grid.totals[0], 0.15)
        self.assertAlmostEqual(grid.days["Day 1"][0], 0.15)


class ScanRecordsTests(unittest.TestCase):
    def _setup_dirs(self, tmp: Path) -> tuple[Path, Path]:
        images_dir = tmp / "images"
        transcriptions_dir = tmp / "transcriptions"
        images_dir.mkdir()
        transcriptions_dir.mkdir()
        return images_dir, transcriptions_dir

    def _make_image(self, images_dir: Path, stem: str) -> Path:
        img_path = images_dir / f"{stem}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG stub
        return img_path

    def _make_transcription(self, transcriptions_dir: Path, stem: str) -> Path:
        data = {"Totals": ["1.0"] * MONTHS}
        for day in range(1, 32):
            data[f"Day {day}"] = ["0.05"] * MONTHS
        path = transcriptions_dir / f"{stem}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _make_jpeg_image(self, images_dir: Path, stem: str) -> Path:
        img_path = images_dir / f"{stem}.jpeg"
        img_path.write_bytes(b"\xff\xd8\xff\xd9")
        return img_path

    def test_paired_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, transcriptions_dir = self._setup_dirs(Path(tmp))
            stem = "DRain_1871-1880_Cornwall-59"
            self._make_image(images_dir, stem)
            self._make_transcription(transcriptions_dir, stem)

            records = scan_records(images_dir, transcriptions_dir)

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertIsInstance(rec, DailyRainfallRecord)
        self.assertEqual(rec.county, "Cornwall")
        self.assertEqual(rec.station_id, "59")
        self.assertEqual(rec.decade, "1871-1880")
        self.assertIsNotNone(rec.grid)
        self.assertIsNotNone(rec.transcription_path)

    def test_unpaired_image_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, transcriptions_dir = self._setup_dirs(Path(tmp))
            stem = "DRain_1871-1880_Hampshire-42"
            self._make_image(images_dir, stem)

            records = scan_records(images_dir, transcriptions_dir)

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertIsNone(rec.grid)
        self.assertIsNone(rec.transcription_path)

    def test_invalid_filename_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, transcriptions_dir = self._setup_dirs(Path(tmp))
            (images_dir / "not-a-valid-name.jpg").write_bytes(b"")
            stem = "DRain_1871-1880_Cornwall-59"
            self._make_image(images_dir, stem)
            self._make_transcription(transcriptions_dir, stem)

            records = scan_records(images_dir, transcriptions_dir)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].stem, stem)

    def test_records_sorted_by_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, transcriptions_dir = self._setup_dirs(Path(tmp))
            stems = [
                "DRain_1881-1890_Berkshire-40",
                "DRain_1871-1880_Cornwall-59",
                "DRain_1871-1880_Hampshire-42",
            ]
            for s in stems:
                self._make_image(images_dir, s)

            records = scan_records(images_dir, transcriptions_dir)

        self.assertEqual([r.stem for r in records], sorted(stems))

    def test_collision_suffixed_stem_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, transcriptions_dir = self._setup_dirs(Path(tmp))
            stem = "DRain_1871-1880_Cornwall-59__deadbeef"
            self._make_image(images_dir, stem)

            records = scan_records(images_dir, transcriptions_dir)

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.county, "Cornwall")
        self.assertEqual(rec.station_id, "59")
        self.assertEqual(rec.decade, "1871-1880")

    def test_jpeg_extension_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, transcriptions_dir = self._setup_dirs(Path(tmp))
            stem = "DRain_1871-1880_Hampshire-42"
            self._make_jpeg_image(images_dir, stem)

            records = scan_records(images_dir, transcriptions_dir)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].stem, stem)

    def test_scan_sample_data(self) -> None:
        """Integration check against the real Daily_rainfall_sample data."""
        from pathlib import Path as P

        sample = P("Daily_rainfall_sample")
        if not sample.exists():
            self.skipTest("Daily_rainfall_sample directory not found")

        records = scan_records(sample / "images", sample / "transcriptions")

        self.assertGreater(len(records), 0)
        paired = [r for r in records if r.grid is not None]
        self.assertGreater(len(paired), 0)
        # Every paired record should have 31 day entries
        for rec in paired[:5]:
            self.assertEqual(len(rec.grid.days), 31)  # type: ignore[union-attr]
            self.assertEqual(len(rec.grid.totals), MONTHS)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
