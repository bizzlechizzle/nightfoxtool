# Dad Cam Pipeline

Legacy camcorder footage processing pipeline for TOD/MTS files.

## Quick Start (Recommended)

```bash
cd dad-cam-pipeline
source .venv/bin/activate
python pipeline_simple.py --source "/path/to/footage" --output "/path/to/output"
```

## Pipelines

### Simplified Pipeline (RECOMMENDED)

Use `pipeline_simple.py` - a streamlined 4-phase pipeline:

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `discover.py` | Scan source folders, build inventory.json |
| 2 | `transcode.py` | Convert TOD/MTS to H.265 MOV |
| 3 | `audio_process.py` | Two-pass loudnorm to -14 LUFS, remove hum |
| 4 | `assemble_jl.py` | Create timeline with J-L audio crossfades |

```bash
# Full pipeline
python pipeline_simple.py --source "/footage" --output "/output"

# Skip phases (use existing clips)
python pipeline_simple.py -s /footage -o /output --skip-transcode --skip-audio

# J-L Assembly only (with existing clips)
python scripts/assemble_jl.py -o "/output" --crossfade 0.75
```

### J-L Crossfade Assembly (Professional)

Use `assemble_jl.py` for professional-quality audio transitions:

```
CLIP A                    CLIP B
Video: [===========]      [============]
                    ↑ hard cut (frame-accurate)

Audio: [=============]
                 [=============]
              └────┬────┘
           crossfade zone (~0.75s)
           A fades out, B fades in
```

Features:
- Video: Hard cuts between clips (no transition effect)
- Audio: Smooth fade in/out at edit points
- A/V sync: <0.04s drift threshold
- Efficient O(n) processing (not O(n²))

### Original Pipeline (Complex)

The original `dad_cam_pipeline.py` has 9 phases with multicam support, but has known issues with audio sync and decision application.

## Output Structure

```
Output/
├── clips/           # Transcoded H.265 clips
│   ├── dad_cam_001.mov ... dad_cam_NNN.mov
│   └── tripod_cam_001.mov ...
├── timeline/        # Continuous edits per camera
│   ├── dad_cam_main_timeline.mov
│   └── dad_cam_tripod_timeline.mov
├── analysis/        # JSON decision files
│   ├── inventory.json
│   └── assembly_results.json
└── logs/            # Pipeline logs
```

## Implementation Guide

### Assembly Strategy

The simplified assembly uses a robust approach to handle clips with different audio sample rates (48kHz vs 96kHz):

1. **Detect sample rates** - Check all clips for audio sample rate
2. **If uniform** - Use FFmpeg concat demuxer with stream copy (fastest)
3. **If mixed** - Pre-process each clip to normalize audio to 48kHz, then concat

This avoids AAC decoder errors that occur when trying to decode/re-encode across clip boundaries with different audio formats.

### Key Code: `assemble_simple.py`

```python
# Detection
sample_rates = set(c.sample_rate for c in clips)

if len(sample_rates) == 1:
    # Stream copy - fast path
    success = self._create_timeline_simple(clips, output_path)
else:
    # Normalize each clip first, then concat
    success = self._create_timeline_normalized(clips, output_path)
```

### Verification

Every timeline is verified for audio/video duration match:

```python
def verify_timeline(path: Path) -> Dict:
    # Get video and audio durations separately
    # Gap < 1.0s is acceptable
    results["duration_match"] = gap < 1.0
```

## Technical Notes

### Audio Sample Rate Issue

Legacy camcorder clips may have mixed sample rates:
- Main camera: Some clips at 48kHz, some at 96kHz
- Tripod camera: Same issue

**Solution**: Pre-process each clip individually to 48kHz before concatenation.

### FFmpeg Concat Demuxer

The concat demuxer requires:
- Same codec for all streams being stream-copied
- Same sample rate for audio stream copy
- Proper escaping of file paths with spaces/special chars

```bash
# Create concat list
file '/path/to/clip1.mov'
file '/path/to/clip2.mov'

# Execute concat
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mov
```

### Audio Normalization

Clips are normalized to -14 LUFS using two-pass loudnorm:
1. First pass: Analyze (measure integrated, true peak, LRA)
2. Second pass: Apply normalization with measured values

60Hz hum detection and removal via highpass/lowpass notch filter.

## Dependencies

- Python 3.10+
- FFmpeg with libx265
- exiftool (for MOI metadata)
- numpy, scipy (for audio analysis)

## Files

| File | Purpose |
|------|---------|
| `pipeline_simple.py` | Simplified 4-phase pipeline (RECOMMENDED) |
| `scripts/assemble_jl.py` | J-L crossfade timeline assembly |
| `scripts/assemble_simple.py` | Simple concat assembly (fallback) |
| `scripts/discover.py` | Source file discovery |
| `scripts/transcode.py` | Video transcoding with drift correction |
| `scripts/audio_process.py` | Audio normalization |

## Transcode Drift Correction

The transcode step includes automatic A/V drift correction:

```python
# Key flags in transcode.py
cmd.extend(["-af", "aresample=48000:async=1"])  # Correct drift
cmd.extend(["-vsync", "cfr"])  # Constant frame rate
```

This ensures all clips have <0.04s A/V drift before assembly.

## Verified Results

Last run (Dec 4, 2025) - **J-L Crossfade Assembly**:

| Camera | Clips | Duration | A/V Sync | Size | Status |
|--------|-------|----------|----------|------|--------|
| Main | 115 | 121.0 min | 0.040s | 6.74 GB | OK |
| Tripod | 4 | 64.8 min | 0.040s | 3.78 GB | OK |

**Assembly Details:**
- Crossfade duration: 0.75s per transition
- Main timeline: 114 crossfades, 85.5s total reduction
- Processing time: ~50 minutes
- All clips verified <0.04s A/V drift threshold
