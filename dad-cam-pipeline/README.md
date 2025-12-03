# Dad Cam Pipeline

Automated video processing pipeline for legacy camcorder footage (TOD, MOI, MTS, AVCHD).

## Features

- **Automatic file ordering** - Parses MOI metadata for correct chronological sequence
- **Stuck pixel detection & removal** - Auto-detects hot/dead pixels across clips
- **Black frame trimming** - Removes camera start/stop artifacts
- **H.265/HEVC encoding** - Small files, universal playback, 4:2:2 10-bit color
- **Deinterlacing** - Converts interlaced 60i to progressive 29.97p
- **Audio processing** - 60Hz hum removal, 2-pass loudness normalization (-14 LUFS)
- **Multicam sync** - Audio cross-correlation for multi-camera shoots
- **FCPXML export** - Timeline-ready import for DaVinci Resolve, FCP X, Premiere

## Requirements

### System Dependencies

```bash
# macOS
brew install ffmpeg exiftool

# Ubuntu/Debian
apt install ffmpeg libimage-exiftool-perl
```

FFmpeg must be compiled with:
- libx265 (HEVC encoding)
- vidstab (stability analysis)

### Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

- Python 3.8+
- NumPy
- SciPy

## Usage

### Full Pipeline

```bash
python dad_cam_pipeline.py \
  --source "/path/to/raw/footage" \
  --output "/path/to/output"
```

### Options

| Option | Description |
|--------|-------------|
| `--source, -s` | Source directory with raw footage |
| `--output, -o` | Output directory for processed files |
| `--parallel, -j` | Number of parallel transcode jobs (default: CPU/2) |
| `--skip-transcode` | Skip transcoding (use existing clips) |
| `--skip-multicam` | Skip multicam synchronization |
| `--dry-run` | Analyze only, don't process files |
| `--verbose, -v` | Enable verbose logging |

### Examples

```bash
# Analyze footage without processing
python dad_cam_pipeline.py -s ./footage -o ./output --dry-run

# Process with 4 parallel jobs
python dad_cam_pipeline.py -s ./footage -o ./output -j 4

# Re-run assembly with existing clips
python dad_cam_pipeline.py -s ./footage -o ./output --skip-transcode
```

## Output Structure

```
output/
├── clips/                  # Individual processed clips
│   ├── dad_cam_001.mov
│   ├── dad_cam_002.mov
│   └── ...
├── master/                 # Concatenated master file
│   └── dad_cam_complete.mov
├── project/                # NLE project files
│   ├── dad_cam_timeline.fcpxml
│   └── dad_cam_multicam.fcpxml
├── analysis/               # Processing metadata
│   ├── inventory.json
│   ├── stuck_pixels.json
│   ├── black_frames.json
│   └── sync_offsets.json
└── logs/                   # Processing logs
    └── pipeline_*.log
```

## Pipeline Phases

### Phase 1: Discovery

- Scans source directories for TOD/MTS/WAV files
- Extracts timestamps from MOI sidecar files
- Establishes correct chronological order
- Detects hexadecimal filename sequences

### Phase 2: Analysis

- **Stuck pixel detection**: Samples frames across clips, identifies pixels with low variance
- **Black frame detection**: Finds camera start/stop artifacts at clip boundaries
- **Stability analysis**: Uses vidstab to score camera shake (for multicam switching)

### Phase 3: Transcoding

- Deinterlaces 60i → 29.97p (yadif)
- Applies stuck pixel removal (delogo)
- Trims black frames
- Encodes to H.265/HEVC CRF 18
- 4:2:2 10-bit, Apple QuickTime compatible

### Phase 4: Audio Processing

- Removes 60Hz hum + harmonics (bandreject filters)
- 2-pass loudness normalization to -14 LUFS
- Targets broadcast standard (EBU R128)

### Phase 5: Multicam Sync

- Extracts audio from all sources
- Cross-correlates against reference (external audio or longest clip)
- Detects clock drift on long recordings
- Generates sync offsets for timeline assembly

### Phase 6: Assembly

- Concatenates clips with audio crossfades
- Generates FCPXML timeline for NLE import
- Creates multicam FCPXML with synchronized angles

## Encoding Specifications

| Setting | Value |
|---------|-------|
| Codec | H.265/HEVC (libx265) |
| CRF | 18 (near-lossless) |
| Pixel Format | yuv422p10le (4:2:2 10-bit) |
| Preset | slow |
| Container | MOV |
| Audio | AAC 256kbps |

## Running Individual Phases

Each phase can be run independently:

```bash
# Discovery only
python scripts/01_discover.py -s ./footage -o ./output

# Analysis only
python scripts/02_analyze.py -o ./output

# Transcoding only
python scripts/03_transcode.py -o ./output -j 4

# Audio processing only
python scripts/04_audio_process.py -o ./output

# Multicam sync only
python scripts/05_sync_multicam.py -o ./output

# Assembly only
python scripts/06_assemble.py -o ./output
```

## Supported Formats

### Input
- TOD (JVC HD camcorder)
- MOD (JVC SD camcorder)
- MTS/M2TS (Sony/Panasonic AVCHD)
- MP4/MOV (generic)
- WAV (external audio)

### Output
- MOV (H.265/HEVC)
- FCPXML (timeline)

## License

MIT License
