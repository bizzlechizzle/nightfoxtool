# Dad Cam Pipeline - Implementation Guide

A comprehensive guide for developers to understand, extend, and maintain this pipeline.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Phases](#pipeline-phases)
3. [Configuration System](#configuration-system)
4. [Adding New Features](#adding-new-features)
5. [Troubleshooting](#troubleshooting)
6. [API Reference](#api-reference)

---

## Architecture Overview

### Directory Structure

```
dad-cam-pipeline/
├── dad_cam_pipeline.py   # Master orchestrator - entry point
├── run.sh                # Convenience shell script
├── requirements.txt      # Python dependencies
├── README.md             # User documentation
│
├── config/               # Configuration system
│   ├── __init__.py
│   └── settings.py       # All configurable parameters
│
├── lib/                  # Shared utilities
│   ├── __init__.py
│   ├── utils.py          # General utilities (progress bar, parsing)
│   ├── ffmpeg_utils.py   # FFmpeg wrapper functions
│   └── logging_utils.py  # Colored logging, phase tracking
│
├── scripts/              # Phase scripts (can run independently)
│   ├── __init__.py
│   ├── 01_discover.py    # Source discovery & metadata
│   ├── 02_analyze.py     # Technical analysis
│   ├── 03_transcode.py   # Video transcoding
│   ├── 04_audio_process.py # Audio processing
│   ├── 05_sync_multicam.py # Multicam sync
│   └── 06_assemble.py    # Timeline assembly
│
└── .venv/                # Python virtual environment
```

### Data Flow

```
Source Files (TOD/MTS/WAV)
         │
         ▼
┌─────────────────┐
│  01_discover    │ → inventory.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  02_analyze     │ → stuck_pixels.json, black_frames.json, stability_scores.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  03_transcode   │ → clips/dad_cam_###.mov
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  04_audio       │ → (modifies clips in-place)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  05_sync        │ → sync_offsets.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  06_assemble    │ → master/dad_cam_complete.mov, project/*.fcpxml
└─────────────────┘
```

---

## Pipeline Phases

### Phase 1: Discovery (`01_discover.py`)

**Purpose**: Find all source files and establish chronological order.

**Key Classes**:
- `SourceDiscovery` - Main discovery class
- `SourceFile` - Dataclass representing a discovered file

**Key Methods**:
```python
discovery = SourceDiscovery(source_dir, output_dir)
inventory = discovery.discover_all()
```

**Output**: `analysis/inventory.json`
```json
{
  "main_camera": [
    {
      "original_path": "/path/to/MOV008.TOD",
      "order": 1,
      "timestamp": "2025-08-23T08:16:21",
      "duration": 12.01
    }
  ],
  "tripod_camera": [...],
  "audio_sources": [...]
}
```

**Important Notes**:
- MOI files contain embedded timestamps (DateTimeOriginal)
- Hexadecimal file naming (MOV008→MOV009→MOV00A) is handled
- Sequence gaps are detected and logged

---

### Phase 2: Analysis (`02_analyze.py`)

**Purpose**: Detect technical issues requiring correction.

**Key Classes**:
- `VideoAnalyzer` - Main analysis class
- `StuckPixel` - Detected pixel info
- `BlackFrameRegion` - Black frame info
- `StabilitySegment` - Stability segment info
- `VignetteAnalysis` - Per-clip vignette analysis

**Stuck Pixel Detection**:
```python
# Algorithm:
# 1. Sample 50 frames across 10 clips
# 2. Compute per-pixel variance
# 3. Flag pixels with low variance + extreme brightness
# 4. Filter by visibility (contrast with neighbors)
```

**Black Frame Detection**:
- Uses FFmpeg's `blackdetect` filter
- Cross-references with `silencedetect` for trim decisions

**Stability Analysis**:
- Uses FFmpeg's `vidstabdetect` filter
- Parses transform file for motion vectors
- Classifies segments as stable/moderate/shaky

**Vignette Detection**:
```python
# Algorithm (per-clip because zoom affects intensity):
# 1. Sample 5 frames evenly across clip
# 2. Measure average brightness in 4 corners vs center
# 3. Compute ratio (lower = more vignette)
# 4. Classify: none (≥0.85), mild (0.70-0.85), moderate (0.55-0.70), severe (<0.55)
# 5. Generate crop filter (1-5% max)
```

**Output Files**:
- `analysis/stuck_pixels.json`
- `analysis/black_frames.json`
- `analysis/stability_scores.json`
- `analysis/vignette.json`

---

### Phase 3: Transcoding (`03_transcode.py`)

**Purpose**: Convert source files to H.265/HEVC.

**Key Classes**:
- `BatchTranscoder` - Manages parallel transcoding
- `TranscodeJob` - Single transcode job

**FFmpeg Pipeline**:
```
Input → Deinterlace (yadif) → Stuck Pixel Fix (delogo) →
Vignette Crop → Scale → Black Frame Trim → H.265 Encode → Output
```

**Encoding Settings** (from `config/settings.py`):
```python
video_codec = "libx265"
video_preset = "slow"
video_crf = 18              # Near-lossless
pixel_format = "yuv422p10le" # 4:2:2 10-bit
video_tag = "hvc1"          # Apple compatibility
```

**Parallel Processing**:
- Uses `ProcessPoolExecutor`
- Default: CPU cores / 2
- Resume support: skips existing valid files

---

### Phase 4: Audio Processing (`04_audio_process.py`)

**Purpose**: Clean and normalize audio.

**Key Classes**:
- `AudioProcessor` - Main processing class

**Processing Chain**:
```
Input → Highpass (60Hz) → Hum Removal (notch filters) →
Loudness Normalization (2-pass) → Output
```

**Hum Removal**:
```python
# Notch filters at 60Hz and harmonics (120, 180, 240, 300 Hz)
# Width increases with frequency for natural rolloff
```

**Loudness Normalization**:
```python
# Pass 1: Measure integrated loudness
# Pass 2: Apply with measured values (linear mode)
# Target: -14 LUFS, -1.5 dBTP
```

---

### Phase 5: Multicam Sync (`05_sync_multicam.py`)

**Purpose**: Synchronize multiple cameras via audio.

**Key Classes**:
- `MulticamSynchronizer` - Main sync class
- `SyncResult` - Sync offset info

**Algorithm**:
1. Extract 60s audio samples from each source
2. Downsample to 8kHz mono
3. Cross-correlate against reference source
4. Validate correlation confidence

**Cross-Correlation**:
```python
from scipy import signal
correlation = signal.correlate(ref_audio, src_audio, mode='full')
lag = np.argmax(correlation) - len(src_audio) + 1
offset_seconds = lag / sample_rate
```

---

### Phase 6: Assembly (`06_assemble.py`)

**Purpose**: Create final outputs.

**Key Classes**:
- `TimelineAssembler` - Main assembly class
- `ClipInfo` - Clip information

**Outputs**:
1. **Master File**: Concatenated video with all clips
2. **FCPXML Timeline**: Individual clips on timeline
3. **Multicam FCPXML**: Multi-angle project (if sync data exists)

**FCPXML Structure**:
```xml
<fcpxml version="1.10">
  <resources>
    <format id="r1" .../>
    <asset id="r2" src="file://..." .../>
  </resources>
  <library>
    <event>
      <project>
        <sequence>
          <spine>
            <asset-clip ref="r2" offset="0s" duration="12s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

---

## Configuration System

All settings are centralized in `config/settings.py`.

### Modifying Settings

```python
from config.settings import PipelineConfig

config = PipelineConfig()

# Override specific settings
config.transcode.video_crf = 20  # Lower quality, smaller files
config.audio.target_lufs = -16   # Quieter output
config.parallel_jobs = 4         # Limit CPU usage
```

### Available Configuration Classes

| Class | Purpose |
|-------|---------|
| `TranscodeSettings` | Video encoding parameters |
| `AudioSettings` | Audio processing parameters |
| `StuckPixelSettings` | Pixel detection thresholds |
| `BlackFrameSettings` | Black frame detection |
| `StabilitySettings` | Camera stability analysis |
| `SyncSettings` | Multicam synchronization |
| `AssemblySettings` | Timeline assembly options |

---

## Adding New Features

### Adding a New Phase Script

1. Create `scripts/07_new_phase.py`:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json
from lib.logging_utils import get_logger, PhaseLogger

class NewPhaseProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

    def process(self, inventory):
        with PhaseLogger("New Phase", self.logger) as phase:
            # Your processing logic here
            pass

def main():
    # CLI interface
    pass

if __name__ == "__main__":
    sys.exit(main())
```

2. Add to `scripts/__init__.py`
3. Import in `dad_cam_pipeline.py`

### Adding a New Configuration Setting

1. Add to appropriate class in `config/settings.py`:
```python
@dataclass
class TranscodeSettings:
    # Existing settings...
    new_setting: str = "default_value"
```

2. Use in your code:
```python
value = config.transcode.new_setting
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| "NumPy not available" | Not installed | `pip install numpy scipy` |
| Encoding error in TRF | Binary in text file | Fixed: uses `errors='ignore'` |
| FFmpeg not found | Not in PATH | `brew install ffmpeg` |
| Module import error | Wrong directory | Run from project root |

### Debug Mode

```bash
python dad_cam_pipeline.py -s /source -o /output --verbose
```

This enables DEBUG level logging with full tracebacks.

### Log Files

All operations are logged to `output/logs/pipeline_YYYYMMDD_HHMMSS.log`

---

## API Reference

### lib/utils.py

```python
run_command(cmd: List[str]) -> CompletedProcess
parse_timestamp(timestamp_str: str) -> datetime
format_timecode(seconds: float, fps: float) -> str
natural_sort_key(path: Path) -> Tuple
load_json(path: Path) -> dict
save_json(data: dict, path: Path) -> None
ProgressBar(total: int, prefix: str)
```

### lib/ffmpeg_utils.py

```python
probe_file(path: Path) -> MediaInfo
get_duration(path: Path) -> float
extract_audio(input_path, output_path, **kwargs) -> bool
measure_loudness(path: Path) -> dict
detect_black_frames(path: Path, threshold, min_duration) -> List[dict]
detect_silence(path: Path, threshold_db, min_duration) -> List[dict]
build_transcode_command(**kwargs) -> List[str]
run_ffmpeg(cmd: List[str]) -> Tuple[bool, str]
```

### lib/logging_utils.py

```python
setup_logger(name, log_dir, level) -> Logger
get_logger() -> Logger
PhaseLogger(phase_name, logger)  # Context manager
```

---

## Testing

### Running Individual Phases

```bash
# Activate venv first
source .venv/bin/activate

# Run specific phase
python scripts/01_discover.py -s /source -o /output
python scripts/02_analyze.py -o /output
python scripts/03_transcode.py -o /output
```

### Dry Run

```bash
python dad_cam_pipeline.py -s /source -o /output --dry-run
```

This runs discovery and analysis without processing.

---

## Performance Notes

### Transcode Estimation

- Source MPEG-2 @ 25Mbps → H.265 CRF 18: ~60% size reduction
- Speed: ~1-2 clips per minute (depends on duration)
- Parallel jobs: Diminishing returns above CPU/2

### Memory Usage

- Stuck pixel detection: ~500MB (frame buffer)
- Audio sync: ~100MB (downsampled audio)
- Transcoding: ~200MB per parallel job

### Disk Space

- Working directory needs ~2x source size
- Final output: ~40-60% of source size
