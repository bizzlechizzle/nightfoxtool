# Audio/Video Sync Fix - Implementation Guide

## Problem Statement

Transcoded clips have mismatched video and audio durations:
- Video: 12.012s
- Audio: 12.032s
- Drift: +20ms per clip

Over 115+ clips, this compounds to **several seconds of audio drift**.

## Root Cause Analysis

In `scripts/transcode.py` line 425-548, the `transcode_worker` function:
1. Encodes video with libx265 (frame-based timing: 30000/1001 fps)
2. Encodes audio with AAC (sample-based timing: 48000 Hz)
3. Does NOT force audio duration to match video

The issue: FFmpeg encodes audio and video streams independently. The audio encoder outputs slightly more samples than the exact video duration requires.

## Solution: Add `-shortest` Flag

The FFmpeg `-shortest` flag stops encoding when the shortest stream ends. Since video is always shorter (frame-aligned), this trims audio to match.

## Implementation

### Part 1: Fix `transcode.py` for Future Transcodes

**File:** `scripts/transcode.py`
**Location:** Line 483-486, in `transcode_worker` function

**Current code:**
```python
# Container options
cmd.extend(["-movflags", "+faststart"])

# Output
cmd.append(str(job.output_path))
```

**New code:**
```python
# Container options
cmd.extend(["-movflags", "+faststart"])

# Force audio to match video duration (prevents drift)
cmd.append("-shortest")

# Output
cmd.append(str(job.output_path))
```

### Part 2: Create Script to Fix Existing Clips

**File:** `scripts/fix_audio_sync.py` (NEW)

This script will:
1. Scan all clips in `Output/clips/`
2. For each clip, check if audio duration > video duration
3. If mismatched, remux with `-shortest` to fix
4. Preserve video stream (no re-encode)
5. Only re-encode audio (fast)

**Algorithm:**
```python
for each clip:
    video_dur = get_video_duration(clip)
    audio_dur = get_audio_duration(clip)

    if audio_dur > video_dur + 0.001:  # >1ms drift
        # Remux with audio trimmed to video length
        ffmpeg -i clip -c:v copy -c:a aac -b:a 256k -shortest output_temp
        replace clip with output_temp
```

### Part 3: Regenerate Doc Edits

After fixing clips, regenerate doc edits with:
- J/L crossfades
- Loudness normalization

## Code Changes

### Change 1: `scripts/transcode.py` (1 line addition)

```diff
@@ -483,6 +483,9 @@ def transcode_worker(job: TranscodeJob, config: PipelineConfig) -> Tuple[bool, s
     # Container options
     cmd.extend(["-movflags", "+faststart"])

+    # Force audio to match video duration (prevents drift)
+    cmd.append("-shortest")
+
     # Output
     cmd.append(str(job.output_path))
```

### Change 2: `scripts/fix_audio_sync.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""
fix_audio_sync.py - Fix Audio/Video Duration Mismatch
=====================================================

Remuxes clips to ensure audio duration matches video duration.
Prevents cumulative audio drift when concatenating clips.

Usage:
    python scripts/fix_audio_sync.py --clips-dir /path/to/clips
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.logging_utils import get_logger, setup_logger


def get_stream_duration(file_path: Path, stream_type: str) -> float:
    """Get duration of a specific stream (v:0 or a:0)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", f"{stream_type}:0",
        "-show_entries", "stream=duration",
        "-of", "csv=p=0",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def fix_clip(clip_path: Path, logger) -> Tuple[bool, float]:
    """
    Fix audio/video sync for a single clip.

    Returns:
        Tuple of (was_fixed, drift_ms)
    """
    video_dur = get_stream_duration(clip_path, "v")
    audio_dur = get_stream_duration(clip_path, "a")

    drift = audio_dur - video_dur
    drift_ms = drift * 1000

    if drift <= 0.001:  # <=1ms is acceptable
        return False, drift_ms

    # Create temp output
    temp_path = clip_path.with_suffix(".temp.mov")

    # Remux with -shortest to trim audio
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(clip_path),
        "-c:v", "copy",           # Don't re-encode video
        "-c:a", "aac",            # Re-encode audio
        "-b:a", "256k",
        "-shortest",              # Trim to shortest stream
        "-movflags", "+faststart",
        str(temp_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error(f"Failed to fix {clip_path.name}: {result.stderr}")
        temp_path.unlink(missing_ok=True)
        return False, drift_ms

    # Verify fix worked
    new_video_dur = get_stream_duration(temp_path, "v")
    new_audio_dur = get_stream_duration(temp_path, "a")
    new_drift = abs(new_audio_dur - new_video_dur)

    if new_drift > 0.005:  # Still >5ms drift
        logger.error(f"Fix failed for {clip_path.name}: still {new_drift*1000:.1f}ms drift")
        temp_path.unlink(missing_ok=True)
        return False, drift_ms

    # Replace original with fixed version
    shutil.move(temp_path, clip_path)
    return True, drift_ms


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix audio/video sync in clips")
    parser.add_argument("--clips-dir", required=True, help="Directory containing clips")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        print(f"Error: {clips_dir} not found")
        return 1

    setup_logger()
    logger = get_logger()

    clips = sorted(clips_dir.glob("*.mov"))
    logger.info(f"Checking {len(clips)} clips for audio sync issues...")

    fixed_count = 0
    total_drift = 0

    for i, clip in enumerate(clips, 1):
        was_fixed, drift_ms = fix_clip(clip, logger)
        total_drift += drift_ms

        if was_fixed:
            fixed_count += 1
            logger.info(f"[{i}/{len(clips)}] Fixed {clip.name} (was {drift_ms:.1f}ms drift)")
        elif drift_ms > 1:
            logger.warning(f"[{i}/{len(clips)}] {clip.name} has {drift_ms:.1f}ms drift (not fixed)")

    logger.info("")
    logger.info(f"Summary:")
    logger.info(f"  Total clips: {len(clips)}")
    logger.info(f"  Fixed: {fixed_count}")
    logger.info(f"  Total drift before fix: {total_drift:.0f}ms ({total_drift/1000:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Execution Plan

### Step 1: Add `-shortest` to transcode.py
- Edit line 483-486
- This fixes future transcodes

### Step 2: Create fix_audio_sync.py
- New script in scripts/
- Non-destructive (only fixes clips with drift)

### Step 3: Run fix on existing clips
```bash
cd dad-cam-pipeline
source .venv/bin/activate
python scripts/fix_audio_sync.py --clips-dir "/Volumes/Jay/Dad Cam/Output/clips"
```

### Step 4: Regenerate doc edits
```bash
python -c "
from pathlib import Path
from scripts.assemble import TimelineAssembler
from config.settings import PipelineConfig

config = PipelineConfig(
    source_dir=Path('/Volumes/Jay/Dad Cam'),
    output_dir=Path('/Volumes/Jay/Dad Cam/Output')
)

assembler = TimelineAssembler(config)
clips = assembler._gather_clips()
assembler._create_doc_edits(clips)
"
```

### Step 5: Verify
- Check doc edits for sync issues
- Spot check several timestamps

### Step 6: Commit and push
```bash
git add -A
git commit -m "Fix audio/video sync drift in transcoded clips"
git push
```

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Remux fails on some clips | Low | Script handles errors gracefully |
| Audio quality loss from re-encode | Minimal | AAC 256k matches original |
| Extended fix time | Medium | Only ~10s per clip, parallelizable |

## Estimated Time

- Code changes: 5 minutes
- Run fix script: ~20 minutes (119 clips × 10s)
- Regenerate doc edits: ~10 minutes
- Total: ~35 minutes
