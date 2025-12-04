#!/usr/bin/env python3
"""
03_transcode.py - Batch Video Transcoding
==========================================

Transcodes source video files to H.265/HEVC with:
- Deinterlacing (yadif) to 29.97p
- Stuck pixel removal (delogo filter)
- Black frame trimming
- 4:2:0 10-bit color (Mac hardware decode compatible)
- Apple QuickTime compatible tagging (hvc1)

Output: clips/dad_cam_###.mov
"""

import sys
import subprocess
import json
import os
import signal
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, ProgressBar, format_duration
from lib.ffmpeg_utils import build_transcode_command, run_ffmpeg, probe_file
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


# Maximum acceptable bitrate for HEVC (50 Mbps)
MAX_HEVC_BITRATE = 50_000_000


@dataclass
class TranscodeJob:
    """Represents a single transcode job."""
    input_path: Path
    output_path: Path
    order: int
    video_filters: List[str]
    trim_start: Optional[float]
    trim_end: Optional[float]
    duration: float


def validate_output(output_path: Path) -> Tuple[bool, str]:
    """
    Validate transcoded file meets quality requirements.

    Checks:
    - Codec is HEVC
    - Profile is Main 10
    - Bitrate is reasonable (< 50 Mbps)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=bit_rate:stream=codec_name,profile",
             "-of", "json", str(output_path)],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return False, "ffprobe failed"

        data = json.loads(result.stdout)

        # Check codec
        streams = data.get("streams", [])
        if not streams:
            return False, "No video stream found"

        codec = streams[0].get("codec_name", "")
        if codec != "hevc":
            return False, f"Wrong codec: {codec} (expected hevc)"

        # Check profile (Main or Main 10 both acceptable)
        profile = streams[0].get("profile", "")
        if profile not in ("Main", "Main 10"):
            return False, f"Wrong profile: {profile} (expected Main or Main 10)"

        # Check bitrate
        bitrate_str = data.get("format", {}).get("bit_rate", "0")
        try:
            bitrate = int(bitrate_str)
        except (ValueError, TypeError):
            bitrate = 0

        if bitrate > MAX_HEVC_BITRATE:
            return False, f"Bitrate too high: {bitrate/1_000_000:.1f} Mbps (max {MAX_HEVC_BITRATE/1_000_000:.0f} Mbps)"

        if bitrate == 0:
            return False, "Could not determine bitrate"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Validation timeout"
    except json.JSONDecodeError:
        return False, "Invalid ffprobe output"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


class BatchTranscoder:
    """Handles batch transcoding of video files."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.clips_dir = config.clips_dir
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Track progress
        self.completed = 0
        self.failed = 0
        self.total = 0

    def transcode_all(
        self,
        inventory: Dict,
        stuck_pixels: Dict,
        black_frames: Dict,
        vignette: Dict = None
    ) -> Dict:
        """
        Transcode all clips from inventory.

        Args:
            inventory: Source inventory
            stuck_pixels: Stuck pixel analysis
            black_frames: Black frame analysis
            vignette: Vignette analysis (per-clip crop recommendations)

        Returns:
            Transcode results
        """
        # Build job list
        jobs = self._build_jobs(inventory, stuck_pixels, black_frames, vignette)
        self.total = len(jobs)

        if not jobs:
            self.logger.warning("No clips to transcode")
            return {"clips": [], "summary": {}}

        self.logger.info(f"Transcoding {len(jobs)} clips...")
        self.logger.info(f"Output directory: {self.clips_dir}")
        self.logger.info(f"Parallel jobs: {self.config.parallel_jobs}")

        # Check for existing VALID files (resume support with validation)
        existing = self._check_existing(jobs)
        if existing:
            self.logger.info(f"  Skipping {len(existing)} already transcoded clips")
            jobs = [j for j in jobs if j.output_path not in existing]
            self.completed = len(existing)

        if not jobs:
            self.logger.info("All clips already transcoded")
            return self._build_results(inventory)

        # Estimate total duration
        total_source_duration = sum(j.duration for j in jobs)
        self.logger.info(f"  Total source duration: {format_duration(total_source_duration)}")

        start_time = datetime.now()

        # Run transcoding
        if self.config.parallel_jobs > 1:
            results = self._transcode_parallel(jobs)
        else:
            results = self._transcode_sequential(jobs)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Summary
        self.logger.info("")
        self.logger.info(f"Transcoding complete:")
        self.logger.info(f"  Completed: {self.completed}")
        self.logger.info(f"  Failed: {self.failed}")
        self.logger.info(f"  Time: {format_duration(elapsed)}")
        if elapsed > 0:
            self.logger.info(f"  Speed: {total_source_duration/elapsed:.1f}x realtime")

        return self._build_results(inventory)

    def _build_jobs(
        self,
        inventory: Dict,
        stuck_pixels: Dict,
        black_frames: Dict,
        vignette: Dict = None
    ) -> List[TranscodeJob]:
        """Build list of transcode jobs from inventory."""
        jobs = []
        settings = self.config.transcode

        # Get delogo filter from stuck pixel analysis
        delogo_filter = stuck_pixels.get("delogo_filter", "") if stuck_pixels else ""

        # Build black frame lookup
        black_frame_lookup = {}
        if black_frames and "clips" in black_frames:
            for clip_data in black_frames["clips"]:
                path = clip_data.get("clip_path", "")
                regions = clip_data.get("regions", [])
                # Find trim points
                trim_start = 0
                trim_end = None
                for region in regions:
                    if region.get("trim_recommended", False):
                        if region.get("position") == "start":
                            trim_start = max(trim_start, region.get("end_time", 0))
                        elif region.get("position") == "end":
                            end = region.get("start_time")
                            if end and (trim_end is None or end < trim_end):
                                trim_end = end
                black_frame_lookup[path] = (trim_start, trim_end)

        # Build vignette crop lookup (per-clip)
        vignette_lookup = {}
        if vignette and "clips" in vignette:
            for clip_data in vignette["clips"]:
                path = clip_data.get("clip_path", "")
                crop_filter = clip_data.get("crop_filter", "")
                if crop_filter:
                    vignette_lookup[path] = crop_filter

        # Process main camera clips
        for clip in inventory.get("main_camera", []):
            job = self._create_job(clip, delogo_filter, black_frame_lookup, vignette_lookup, "main")
            if job:
                jobs.append(job)

        # Process tripod camera clips
        for clip in inventory.get("tripod_camera", []):
            job = self._create_job(clip, "", black_frame_lookup, vignette_lookup, "tripod")
            if job:
                jobs.append(job)

        return sorted(jobs, key=lambda j: j.order)

    def _create_job(
        self,
        clip: Dict,
        delogo_filter: str,
        black_frame_lookup: Dict,
        vignette_lookup: Dict,
        camera_type: str
    ) -> Optional[TranscodeJob]:
        """Create a transcode job for a clip."""
        input_path = Path(clip["original_path"])
        if not input_path.exists():
            self.logger.warning(f"Source file not found: {input_path}")
            return None

        # Generate output filename based on camera type
        order = clip.get("order", 0)
        if camera_type == "tripod":
            output_name = f"tripod_cam_{order:03d}.mov"
        else:
            output_name = f"{self.config.output_prefix}_{order:03d}.mov"
        output_path = self.clips_dir / output_name

        # Build video filters
        video_filters = []

        # 1. Deinterlacing (if interlaced)
        if clip.get("is_interlaced", True):
            video_filters.append(self.config.transcode.deinterlace_filter)

        # 2. Stuck pixel removal (before crop)
        if delogo_filter and camera_type == "main":
            video_filters.append(delogo_filter)

        # 3. Vignette crop (per-clip, based on zoom level)
        crop_filter = vignette_lookup.get(clip["original_path"], "")
        if crop_filter:
            video_filters.append(crop_filter)
            # Scale back to original resolution after crop to maintain dimensions
            video_filters.append("scale=1440:1080:flags=lanczos")

        # 4. Timestamp reset
        video_filters.append("setpts=PTS-STARTPTS")

        # Get trim points
        trim_start, trim_end = black_frame_lookup.get(str(input_path), (0, None))

        return TranscodeJob(
            input_path=input_path,
            output_path=output_path,
            order=order,
            video_filters=video_filters,
            trim_start=trim_start if trim_start > 0 else None,
            trim_end=trim_end,
            duration=clip.get("duration", 0)
        )

    def _check_existing(self, jobs: List[TranscodeJob]) -> set:
        """Check for already transcoded files with full validation."""
        existing = set()
        for job in jobs:
            if job.output_path.exists():
                # Verify it's a valid video file with correct encoding
                info = probe_file(job.output_path)
                if info and info.duration > 0:
                    # Additional validation for correct encoding
                    valid, error = validate_output(job.output_path)
                    if valid:
                        existing.add(job.output_path)
                    else:
                        # Invalid file - delete it so it gets re-encoded
                        self.logger.warning(f"Removing invalid clip {job.output_path.name}: {error}")
                        try:
                            job.output_path.unlink()
                        except OSError:
                            pass
        return existing

    def _transcode_sequential(self, jobs: List[TranscodeJob]) -> List[Dict]:
        """Transcode jobs sequentially."""
        results = []
        progress = ProgressBar(len(jobs), prefix="Transcoding")

        for job in jobs:
            success, error = self._transcode_single(job)
            results.append({
                "input": str(job.input_path),
                "output": str(job.output_path),
                "success": success,
                "error": error
            })

            if success:
                self.completed += 1
            else:
                self.failed += 1

            progress.update(1, suffix=job.output_path.name)

        progress.finish()
        return results

    def _transcode_parallel(self, jobs: List[TranscodeJob]) -> List[Dict]:
        """Transcode jobs in parallel."""
        results = []

        # Use process pool for parallel encoding
        with ProcessPoolExecutor(max_workers=self.config.parallel_jobs) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(transcode_worker, job, self.config): job
                for job in jobs
            }

            # Progress tracking
            progress = ProgressBar(len(jobs), prefix="Transcoding")

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    success, error = future.result()
                except Exception as e:
                    success = False
                    error = str(e)

                results.append({
                    "input": str(job.input_path),
                    "output": str(job.output_path),
                    "success": success,
                    "error": error
                })

                if success:
                    self.completed += 1
                else:
                    self.failed += 1
                    self.logger.error(f"Failed: {job.output_path.name} - {error}")

                progress.update(1, suffix=job.output_path.name)

            progress.finish()

        return results

    def _transcode_single(self, job: TranscodeJob) -> Tuple[bool, str]:
        """Transcode a single file."""
        return transcode_worker(job, self.config)

    def _build_results(self, inventory: Dict) -> Dict:
        """Build final results dictionary."""
        clips = []

        # Scan output directory for completed clips
        for f in sorted(self.clips_dir.glob("*.mov")):
            info = probe_file(f)
            if info:
                clips.append({
                    "path": str(f),
                    "filename": f.name,
                    "duration": info.duration,
                    "size": f.stat().st_size,
                })

        return {
            "clips": clips,
            "output_dir": str(self.clips_dir),
            "summary": {
                "total_clips": len(clips),
                "total_duration": sum(c["duration"] for c in clips),
                "total_size": sum(c["size"] for c in clips),
            }
        }


def transcode_worker(job: TranscodeJob, config: PipelineConfig) -> Tuple[bool, str]:
    """
    Worker function for transcoding a single file.
    Runs in separate process for parallel execution.

    Uses H.265/HEVC with Main 10 profile for Mac hardware decode compatibility.
    """
    settings = config.transcode

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner"]

    # Input with optional trim
    if job.trim_start:
        cmd.extend(["-ss", str(job.trim_start)])

    cmd.extend(["-i", str(job.input_path)])

    if job.trim_end:
        duration = job.trim_end - (job.trim_start or 0)
        cmd.extend(["-t", str(duration)])

    # Video filters
    if job.video_filters:
        cmd.extend(["-vf", ",".join(job.video_filters)])

    # Video encoding - H.265 with fixed settings for Mac compatibility
    cmd.extend(["-c:v", "libx265"])

    # Preset - use faster preset for long clips
    preset = settings.video_preset if settings.video_preset else "medium"
    if job.duration > 1800:  # > 30 minutes
        preset = "fast"
    cmd.extend(["-preset", preset])

    # CRF - ALWAYS include this for quality control
    crf = settings.video_crf if settings.video_crf and settings.video_crf > 0 else 18
    cmd.extend(["-crf", str(crf)])

    # Pixel format - 4:2:0 8-bit (matches source)
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Profile - Main for 8-bit
    cmd.extend(["-profile:v", "main"])

    # Tag - hvc1 for Apple QuickTime compatibility
    cmd.extend(["-tag:v", "hvc1"])

    # Color metadata
    cmd.extend(["-color_primaries", "bt709"])
    cmd.extend(["-color_trc", "bt709"])
    cmd.extend(["-colorspace", "bt709"])

    # Audio encoding with drift correction
    # aresample=48000:async=1 resamples to 48kHz AND corrects A/V drift
    cmd.extend(["-af", "aresample=48000:async=1"])
    cmd.extend(["-c:a", "aac"])
    cmd.extend(["-b:a", "256k"])

    # Constant frame rate to prevent VFR drift
    cmd.extend(["-vsync", "cfr"])

    # Container options
    cmd.extend(["-movflags", "+faststart"])

    # Force audio to match video duration (prevents drift)
    cmd.append("-shortest")

    # Output
    cmd.append(str(job.output_path))

    # Run FFmpeg with 2-hour timeout
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per clip
        )

        if result.returncode != 0:
            # Clean up failed output
            if job.output_path.exists():
                try:
                    job.output_path.unlink()
                except OSError:
                    pass

            # Extract error message
            error_lines = result.stderr.split('\n')
            error_msg = next(
                (line for line in reversed(error_lines) if line.strip()),
                "Unknown error"
            )
            return False, error_msg

        # Verify output exists
        if not job.output_path.exists():
            return False, "Output file not created"

        if job.output_path.stat().st_size < 1000:
            job.output_path.unlink()
            return False, "Output file too small"

        # Validate output encoding
        valid, error = validate_output(job.output_path)
        if not valid:
            # Delete invalid file
            try:
                job.output_path.unlink()
            except OSError:
                pass
            return False, f"Validation failed: {error}"

        return True, ""

    except subprocess.TimeoutExpired:
        # Clean up incomplete file
        if job.output_path.exists():
            try:
                job.output_path.unlink()
            except OSError:
                pass
        return False, "Transcode timeout (2 hours)"
    except Exception as e:
        # Clean up on any error
        if job.output_path.exists():
            try:
                job.output_path.unlink()
            except OSError:
                pass
        return False, str(e)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcode video files to H.265"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU/2)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup
    import logging
    output_dir = Path(args.output)
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_dir=output_dir / "logs", level=level)
    logger = get_logger()

    # Load analysis data
    analysis_dir = output_dir / "analysis"

    inventory_path = analysis_dir / "inventory.json"
    if not inventory_path.exists():
        logger.error("Inventory not found - run 01_discover.py first")
        return 1
    inventory = load_json(inventory_path)

    stuck_pixels_path = analysis_dir / "stuck_pixels.json"
    stuck_pixels = load_json(stuck_pixels_path) if stuck_pixels_path.exists() else {}

    black_frames_path = analysis_dir / "black_frames.json"
    black_frames = load_json(black_frames_path) if black_frames_path.exists() else {}

    vignette_path = analysis_dir / "vignette.json"
    vignette = load_json(vignette_path) if vignette_path.exists() else {}

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir
    if args.parallel:
        config.parallel_jobs = args.parallel

    # Run transcoding
    transcoder = BatchTranscoder(config)

    with PhaseLogger("Batch Transcoding", logger):
        results = transcoder.transcode_all(inventory, stuck_pixels, black_frames, vignette)

    # Save results
    save_json(results, analysis_dir / "transcode_results.json")

    return 0 if transcoder.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
