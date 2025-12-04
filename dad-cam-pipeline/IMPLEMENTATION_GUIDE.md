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

---

# CRITICAL FIX: Assembly Decision Application

**Date Added:** December 4, 2025
**Related:** AUDIT_REPORT.md, FIX_PLAN.md

## The Problem

The pipeline generates intelligent decision files (audio mix, multicam edits, sync offsets) but the assembly phase **NEVER USES THEM**. This causes:
- Audio gaps (lapel mic not used)
- No camera switching (multicam decisions ignored)
- Sync issues (offsets not applied)

See `AUDIT_REPORT.md` for full details.

---

## Quick Fix Guide for Developers

### Prerequisites
- Read `AUDIT_REPORT.md` to understand the issues
- Read `FIX_PLAN.md` for the detailed plan
- Back up `scripts/assemble.py` before making changes

### Step-by-Step Implementation

#### 1. Load Decision Files in `__init__` (5 min)

**Location:** `scripts/assemble.py`, inside `__init__`, after line 59

```python
# Load decision files for assembly
self.audio_mix_decisions = None
self.multicam_decisions = None
self.sync_offsets = None
self._source_to_order_map = None

audio_mix_path = config.analysis_dir / "audio_mix_decisions.json"
multicam_path = config.analysis_dir / "multicam_edit_decisions.json"
sync_path = config.analysis_dir / "sync_offsets.json"

if audio_mix_path.exists():
    self.audio_mix_decisions = load_json(audio_mix_path)
    self.logger.info(f"Loaded {len(self.audio_mix_decisions.get('segments', []))} audio mix segments")

if multicam_path.exists():
    self.multicam_decisions = load_json(multicam_path)
    self.logger.info(f"Loaded {len(self.multicam_decisions.get('decisions', []))} multicam decisions")

if sync_path.exists():
    self.sync_offsets = load_json(sync_path)
```

#### 2. Add Timeline Mapping Helper (5 min)

**Location:** After `_gather_clips` method

```python
def _build_timeline_clip_map(self, clips: List[ClipInfo]) -> Dict:
    """Build a mapping from timeline position to clip info."""
    boundaries = []
    timeline_pos = 0.0
    clips_by_order = {}

    for clip in clips:
        start = timeline_pos
        end = timeline_pos + clip.duration
        boundaries.append((start, end, clip))
        clips_by_order[clip.order] = clip
        timeline_pos = end

    return {
        "clip_boundaries": boundaries,
        "total_duration": timeline_pos,
        "clips_by_order": clips_by_order
    }

def _find_clip_at_time(self, time: float, timeline_map: Dict) -> Optional[ClipInfo]:
    """Find which clip is playing at a given timeline position."""
    for start, end, clip in timeline_map["clip_boundaries"]:
        if start <= time < end:
            return clip
    return None

def _get_clip_start_time(self, clip: ClipInfo, timeline_map: Dict) -> float:
    """Get the timeline start position for a clip."""
    for start, end, c in timeline_map["clip_boundaries"]:
        if c.order == clip.order:
            return start
    return 0.0
```

#### 3. Add Source-to-Clip Mapper (10 min)

This maps original source paths (e.g., `MOV008.TOD`) to transcoded clips (e.g., `dad_cam_001.mov`).

```python
def _find_transcoded_clip(self, original_path: str, clips: List[ClipInfo]) -> Optional[ClipInfo]:
    """Find the transcoded clip for an original source file."""
    # Build mapping on first call
    if self._source_to_order_map is None:
        self._source_to_order_map = {}
        inventory_path = self.config.analysis_dir / "inventory.json"
        if inventory_path.exists():
            inventory = load_json(inventory_path)
            for clip_data in inventory.get("main_camera", []):
                self._source_to_order_map[clip_data["original_path"]] = clip_data["order"]
            for clip_data in inventory.get("tripod_camera", []):
                self._source_to_order_map[clip_data["original_path"]] = clip_data["order"]

    order = self._source_to_order_map.get(original_path)
    if order is None:
        return None

    expected = f"{self.config.output_prefix}_{order:03d}.mov"
    for clip in clips:
        if clip.filename == expected:
            return clip

    expected_tripod = f"tripod_cam_{order:03d}.mov"
    for clip in clips:
        if clip.filename == expected_tripod:
            return clip

    return None
```

#### 4. Add Doc Edit Feature (15 min)

Creates per-camera files as a quick win.

```python
def _create_doc_edits(self, clips: List[ClipInfo]) -> Dict[str, Path]:
    """Create separate doc edit files for each camera."""
    results = {}

    main_clips = [c for c in clips if c.filename.startswith(self.config.output_prefix)]
    tripod_clips = [c for c in clips if c.filename.startswith("tripod_cam")]

    if main_clips:
        main_path = self.master_dir / f"{self.config.output_prefix}_main_camera_docedit.mov"
        if self._create_doc_edit_simple(main_clips, main_path):
            results["main_camera"] = main_path

    if tripod_clips:
        tripod_path = self.master_dir / f"{self.config.output_prefix}_tripod_camera_docedit.mov"
        if self._create_doc_edit_simple(tripod_clips, tripod_path):
            results["tripod_camera"] = tripod_path

    return results

def _create_doc_edit_simple(self, clips: List[ClipInfo], output_path: Path) -> bool:
    """Simple concat for doc edit."""
    concat_list = self.config.analysis_dir / "docedit_concat.txt"
    with open(concat_list, 'w') as f:
        for clip in clips:
            escaped = str(clip.path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', str(concat_list),
        '-c', 'copy',
        '-movflags', '+faststart',
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=7200)
    concat_list.unlink(missing_ok=True)
    return result.returncode == 0
```

#### 5. Update `assemble_all` Flow (10 min)

Add doc edits to the assembly flow:

```python
# After gathering clips:
clips = self._gather_clips()
timeline_map = self._build_timeline_clip_map(clips)

# Add doc edit creation:
with PhaseLogger("Creating Per-Camera Doc Edits", self.logger) as phase:
    doc_edits = self._create_doc_edits(clips)
    results["doc_edits"] = {k: str(v) for k, v in doc_edits.items()}
    if doc_edits:
        phase.success(f"Created {len(doc_edits)} doc edits")
```

---

## Testing Your Changes

```bash
cd /Volumes/Jay/Dad\ Cam/dad-cam-pipeline
source .venv/bin/activate

# Quick test with skip flags
python dad_cam_pipeline.py \
  --source "/Volumes/Jay/Dad Cam" \
  --output "/Volumes/Jay/Dad Cam/Output" \
  --skip-transcode --skip-audio

# Check for doc edits
ls -la "/Volumes/Jay/Dad Cam/Output/master/"
```

---

## Common Debug Patterns

### Check If Decision Files Are Loading
```python
# Add to assemble_all after loading:
self.logger.info(f"Audio decisions: {self.audio_mix_decisions is not None}")
self.logger.info(f"Multicam decisions: {self.multicam_decisions is not None}")
```

### Verify Clip Mapping
```python
# Test source-to-clip mapping:
test_path = "/Volumes/Jay/Dad Cam/Main Camera/PRG014/MOV008.TOD"
found = self._find_transcoded_clip(test_path, clips)
self.logger.info(f"Mapping test: {test_path} -> {found.filename if found else 'NOT FOUND'}")
```

### Check FFmpeg Errors
```python
result = subprocess.run(cmd, capture_output=True, text=True, ...)
if result.returncode != 0:
    self.logger.error(f"FFmpeg failed: {result.stderr[-500:]}")
```

---

## Reference: Decision File Formats

### audio_mix_decisions.json
```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 49.0,
      "primary_source": "/path/to/lapel.wav",
      "source_type": "lapel",
      "gain_db": 0.0
    }
  ]
}
```

### multicam_edit_decisions.json
```json
{
  "decisions": [
    {
      "timeline_start": 123.168,
      "timeline_end": 156.72,
      "source_camera": "tripod",
      "source_clip": "/path/to/00000.MTS",
      "source_start": 124.28
    }
  ]
}
```

### sync_offsets.json
```json
{
  "sources": {
    "main_camera": {
      "offset_seconds": 22.796625
    },
    "tripod_camera": {
      "offset_seconds": -1.1175
    }
  }
}
```

---

## Next Steps After Basic Fix

1. Implement `_apply_audio_mix` for full audio mixing
2. Implement `_apply_multicam_edits` for camera switching
3. Fix `_generate_multicam_fcpxml` to use actual decisions
4. Add AAC validation to catch corruption early

See `FIX_PLAN.md` for complete implementation details.
