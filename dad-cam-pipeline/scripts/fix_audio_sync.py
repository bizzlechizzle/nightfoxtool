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

    # Use atrim filter to force exact audio duration to match video
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(clip_path),
        "-c:v", "copy",           # Don't re-encode video
        "-af", f"atrim=0:{video_dur},asetpts=PTS-STARTPTS",  # Trim audio to exact video duration
        "-c:a", "aac",            # Re-encode audio
        "-b:a", "256k",
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
        total_drift += abs(drift_ms)

        if was_fixed:
            fixed_count += 1
            logger.info(f"[{i}/{len(clips)}] Fixed {clip.name} (was {drift_ms:.1f}ms drift)")
        elif drift_ms > 1:
            logger.debug(f"[{i}/{len(clips)}] {clip.name} has {drift_ms:.1f}ms drift")

    logger.info("")
    logger.info(f"Summary:")
    logger.info(f"  Total clips: {len(clips)}")
    logger.info(f"  Fixed: {fixed_count}")
    logger.info(f"  Total drift before fix: {total_drift:.0f}ms ({total_drift/1000:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
