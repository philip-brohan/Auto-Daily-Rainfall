# Data Ingest

The ingest stage scans the sample directory and pairs each document image with
its human transcription, building a dataset of labelled records.

## Directory layout

```
Daily_rainfall_sample/
├── images/
│   ├── DRain_1871-1880_Cornwall-59.jpg
│   └── ...
└── transcriptions/
    ├── DRain_1871-1880_Cornwall-59.json
    └── ...
```

Images and transcriptions are matched by filename stem.  An image with no
matching transcription is flagged as *unpaired* and excluded from fine-tuning
(but can still be used for inference).

## Running ingest

```bash
weather-extract ingest
```

Example output:

```
Scanned 20 records (18 paired, 2 unpaired)
{
  "total": 20,
  "paired": 18,
  "unpaired": 2
}
```

## Transcription format

Each `.json` transcription file is a `DailyRainfallGrid` object:

```json
{
  "days": {
    "Day 1":  [null, null, 0.3, 1.1, ...],
    "Day 2":  [0.0,  0.0,  0.0, 2.4, ...],
    ...
    "Day 31": [null, null, null, ...]
  },
  "totals": [12.4, 8.1, 31.2, ...]
}
```

- Each `days` entry is a list of 12 values (one per month).
- `null` means the day does not exist in that month (e.g. Feb 30) or the cell was blank.
- `totals` is a list of 12 monthly totals.

## Adding your own data

1. Place scanned images in `Daily_rainfall_sample/images/` (or point
   `IngestConfig.images_dir` at your own folder).
2. Create matching `.json` transcriptions for as many images as you have.
3. Re-run `weather-extract ingest` to pick up the new records.
