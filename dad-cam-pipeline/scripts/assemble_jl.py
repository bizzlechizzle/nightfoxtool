#!/usr/bin/env python3
"""
assemble_jl.py - J-L Crossfade Timeline Assembly
=================================================

Creates timeline edits with J-L (split) crossfades:
- Video: Hard cut at edit point (no transition)
- Audio: Soft crossfade overlapping at edit point

This mimics professional film/TV editing where audio transitions
are smoother than video cuts, creating a more pleasant viewing experience.

J-L Crossfade Visualization:
    CLIP A                    CLIP B
    Video: [===========]      [============]
                        ↑ hard cut (frame-accurate)

    Audio: [=============]
                     [=============]
                  └────┬────┘
               crossfade zone (~0.75s)
               A fades out, B fades in

Output:
- timeline/dad_cam_main_timeline.mov     (main camera with J-L crossfades)
- timeline/dad_cam_tripod_timeline.mov   (tripod camera with J-L crossfades)
"""

import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json
from lib.ffmpeg_utils import probe_file
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


# Constants
DEFAULT_CROSSFADE = 0.75  # seconds - standard J-L crossfade duration
BATCH_SIZE = 10  # Process clips in batches to avoid FFmpeg filter limits


@dataclass
class ClipInfo:
    """Information about a clip for assembly."""
    path: Path
    filename: str
    order: int
    duration: float
    video_duration: float
    audio_duration: float


class JLAssembler:
    """
    J-L Crossfade Timeline Assembler.

    Strategy:
    1. Gather clips, verify A/V sync
    2. Process clips in batches with audio crossfades
    3. Combine batches into final timeline
    """

    def __init__(self, config: PipelineConfig, crossfade_duration: float = DEFAULT_CROSSFADE):
        self.config = config
        self.crossfade = crossfade_duration
        self.logger = get_logger()

        # Output directory
        self.timeline_dir = config.output_dir / "timeline"
        self.timeline_dir.mkdir(parents=True, exist_ok=True)

        # Temp directory for intermediate files
        self.temp_dir = config.analysis_dir / "temp_jl_assembly"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def assemble(self) -> Dict:
        """
        Assemble all clips into timeline edits with J-L crossfades.

        Returns:
            Results dictionary with paths to created files
        """
        results = {
            "main_timeline": None,
            "tripod_timeline": None,
            "clip_counts": {},
            "durations": {},
            "crossfade_duration": self.crossfade,
            "errors": []
        }

        # Gather all clips
        all_clips = self._gather_clips()

        if not all_clips:
            self.logger.error("No clips found in clips directory")
            results["errors"].append("No clips found")
            return results

        # Separate by camera
        main_clips = [c for c in all_clips if c.filename.startswith("dad_cam_")]
        tripod_clips = [c for c in all_clips if c.filename.startswith("tripod_cam_")]

        results["clip_counts"] = {
            "main": len(main_clips),
            "tripod": len(tripod_clips),
            "total": len(all_clips)
        }

        self.logger.info(f"Found {len(main_clips)} main camera clips, {len(tripod_clips)} tripod clips")
        self.logger.info(f"Using {self.crossfade}s J-L crossfade duration")

        # Create main camera timeline
        if main_clips:
            self.logger.info("Creating main camera timeline with J-L crossfades...")
            main_path = self.timeline_dir / "dad_cam_main_timeline.mov"

            success, duration = self._create_jl_timeline(main_clips, main_path, "main")

            if success:
                results["main_timeline"] = str(main_path)
                results["durations"]["main"] = duration

                # Calculate expected duration
                raw_duration = sum(c.duration for c in main_clips)
                crossfade_reduction = self.crossfade * (len(main_clips) - 1)
                expected_duration = raw_duration - crossfade_reduction

                self.logger.info(f"  Raw duration: {raw_duration/60:.1f} min")
                self.logger.info(f"  Crossfade reduction: {crossfade_reduction:.1f}s ({len(main_clips)-1} crossfades)")
                self.logger.info(f"  Final duration: {duration/60:.1f} min")
            else:
                results["errors"].append("Main timeline creation failed")

        # Create tripod camera timeline
        if tripod_clips:
            self.logger.info("Creating tripod camera timeline with J-L crossfades...")
            tripod_path = self.timeline_dir / "dad_cam_tripod_timeline.mov"

            success, duration = self._create_jl_timeline(tripod_clips, tripod_path, "tripod")

            if success:
                results["tripod_timeline"] = str(tripod_path)
                results["durations"]["tripod"] = duration
                self.logger.info(f"  Tripod timeline: {duration/60:.1f} minutes")
            else:
                results["errors"].append("Tripod timeline creation failed")

        # Cleanup temp files
        self._cleanup_temp()

        return results

    def _gather_clips(self) -> List[ClipInfo]:
        """Gather information about all transcoded clips."""
        clips = []
        clips_dir = self.config.clips_dir

        if not clips_dir.exists():
            return clips

        for clip_path in sorted(clips_dir.glob("*.mov")):
            info = probe_file(clip_path)

            if not info or not info.video_stream:
                self.logger.warning(f"Could not probe {clip_path.name}, skipping")
                continue

            # Extract order from filename
            try:
                order = int(clip_path.stem.split('_')[-1])
            except ValueError:
                order = len(clips) + 1

            # Get durations
            video_dur = info.duration
            audio_dur = info.duration

            if info.video_stream:
                video_dur = getattr(info.video_stream, 'duration', info.duration) or info.duration
            if info.audio_stream:
                audio_dur = getattr(info.audio_stream, 'duration', info.duration) or info.duration

            clips.append(ClipInfo(
                path=clip_path,
                filename=clip_path.name,
                order=order,
                duration=info.duration,
                video_duration=video_dur,
                audio_duration=audio_dur
            ))

        clips.sort(key=lambda c: c.order)
        return clips

    def _create_jl_timeline(
        self,
        clips: List[ClipInfo],
        output_path: Path,
        camera_name: str
    ) -> Tuple[bool, float]:
        """
        Create a timeline with J-L crossfades using efficient batch processing.

        Strategy for O(n) efficiency:
        1. Pre-process each clip to add audio fade in/out
        2. Trim video at clip boundaries by crossfade duration
        3. Concat all preprocessed clips (stream copy)

        This avoids O(n²) re-encoding of the pairwise approach.

        Returns:
            (success, duration) tuple
        """
        if not clips:
            return False, 0.0

        if len(clips) == 1:
            shutil.copy2(clips[0].path, output_path)
            return True, clips[0].duration

        self.logger.info(f"  Processing {len(clips)} clips with J-L crossfades...")

        # Step 1: Pre-process each clip (add fades, trim video)
        preprocessed = []
        total_expected_duration = 0

        for i, clip in enumerate(clips):
            is_first = (i == 0)
            is_last = (i == len(clips) - 1)

            prep_path = self.temp_dir / f"prep_{camera_name}_{i:04d}.mov"

            success = self._preprocess_clip_for_jl(
                clip, prep_path, is_first, is_last
            )

            if success:
                preprocessed.append(prep_path)
                # Calculate expected duration contribution
                clip_contrib = clip.duration
                if not is_first:
                    clip_contrib -= self.crossfade  # Trimmed at start
                total_expected_duration += clip_contrib
            else:
                # Fallback: use original clip
                self.logger.warning(f"  Failed to preprocess {clip.filename}, using original")
                preprocessed.append(clip.path)
                total_expected_duration += clip.duration

            # Progress every 20 clips
            if (i + 1) % 20 == 0:
                self.logger.info(f"  Preprocessed {i+1}/{len(clips)} clips...")

        self.logger.info(f"  Preprocessed {len(preprocessed)} clips")

        # Step 2: Concat all preprocessed clips
        self.logger.info(f"  Concatenating into final timeline...")

        concat_list = self.temp_dir / f"concat_{camera_name}.txt"
        with open(concat_list, 'w') as f:
            for prep_path in preprocessed:
                escaped = str(prep_path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            '-movflags', '+faststart',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Cleanup
        concat_list.unlink(missing_ok=True)
        for prep_path in preprocessed:
            if prep_path.parent == self.temp_dir:
                prep_path.unlink(missing_ok=True)

        if result.returncode != 0:
            self.logger.error(f"  Concat failed: {result.stderr[-300:]}")
            return False, 0.0

        if output_path.exists():
            info = probe_file(output_path)
            return True, info.duration if info else 0.0

        return False, 0.0

    def _preprocess_clip_for_jl(
        self,
        clip: ClipInfo,
        output_path: Path,
        is_first: bool,
        is_last: bool
    ) -> bool:
        """
        Preprocess a single clip for J-L assembly:
        - Trim video AND audio at start (except first clip) by crossfade duration
        - Add audio fade-in at start (except first clip)
        - Add audio fade-out at end (except last clip)

        Returns:
            True if successful
        """
        # Build filter components
        video_filters = []
        audio_filters = ["aresample=48000"]  # Normalize sample rate

        # Calculate effective duration after trimming
        effective_duration = clip.duration
        if not is_first:
            effective_duration -= self.crossfade

        # Video and Audio: trim at start if not first clip (must match!)
        if not is_first:
            video_filters.append(f"trim=start={self.crossfade}")
            video_filters.append("setpts=PTS-STARTPTS")
            # Audio must also be trimmed to match video
            audio_filters.append(f"atrim=start={self.crossfade}")
            audio_filters.append("asetpts=PTS-STARTPTS")

        # Audio fades (applied after trim)
        if not is_first:
            audio_filters.append(f"afade=t=in:st=0:d={self.crossfade/2}:curve=tri")

        if not is_last:
            fade_out_start = effective_duration - self.crossfade/2
            if fade_out_start > 0:
                audio_filters.append(f"afade=t=out:st={fade_out_start}:d={self.crossfade/2}:curve=tri")

        # Build command
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(clip.path),
        ]

        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
            # Must re-encode video when filtering
            cmd.extend([
                '-c:v', 'libx265', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1'
            ])
        else:
            cmd.extend(['-c:v', 'copy'])

        cmd.extend(['-af', ','.join(audio_filters)])
        cmd.extend(['-c:a', 'aac', '-b:a', '256k'])
        cmd.extend(['-movflags', '+faststart'])
        cmd.append(str(output_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            return result.returncode == 0 and output_path.exists()
        except Exception as e:
            self.logger.debug(f"Preprocess error: {e}")
            return False

    def _join_two_clips(
        self,
        clip_a: ClipInfo,
        clip_b: ClipInfo,
        output_path: Path
    ) -> bool:
        """
        Join two clips with J-L crossfade.

        - Video: Hard cut at edit point, properly trimmed
        - Audio: acrossfade for smooth transition, normalized to 48kHz

        The key is that both video and audio end up with the same duration:
        total = clip_a + clip_b - crossfade_duration

        Returns:
            True if successful
        """
        dur_a = clip_a.duration

        # Expected output duration
        expected_duration = dur_a + clip_b.duration - self.crossfade

        # Build filter_complex for J-L crossfade
        # Video: Trim clip_b at start by crossfade duration, then concat
        # This ensures video duration matches audio crossfade output
        # Audio: resample both to 48kHz first, then acrossfade
        filter_complex = (
            # Trim clip_b video by crossfade at start (J-L: video cuts where audio overlaps)
            f"[1:v]trim=start={self.crossfade},setpts=PTS-STARTPTS[v1trimmed];"
            f"[0:v][v1trimmed]concat=n=2:v=1:a=0[v];"
            # Audio: resample to 48kHz and crossfade
            f"[0:a]aresample=48000[a0];"
            f"[1:a]aresample=48000[a1];"
            f"[a0][a1]acrossfade=d={self.crossfade}:c1=tri:c2=tri[a]"
        )

        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(clip_a.path),
            '-i', str(clip_b.path),
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', '[a]',
            # Video encoding - fast re-encode (filter_complex requires it)
            '-c:v', 'libx265',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-tag:v', 'hvc1',
            # Audio encoding
            '-c:a', 'aac', '-b:a', '256k',
            '-movflags', '+faststart',
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout per join (re-encoding takes longer)
            )

            if result.returncode != 0:
                self.logger.debug(f"FFmpeg error: {result.stderr[-500:]}")
                return False

            return output_path.exists() and output_path.stat().st_size > 1000

        except subprocess.TimeoutExpired:
            self.logger.error("Join operation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Join error: {e}")
            return False

    def _cleanup_temp(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*"):
                f.unlink(missing_ok=True)


def verify_jl_timeline(path: Path, expected_clips: int, crossfade: float) -> Dict:
    """
    Verify a J-L timeline meets quality requirements.

    Returns dict with verification results.
    """
    results = {
        "exists": path.exists(),
        "video_duration": 0,
        "audio_duration": 0,
        "duration_match": False,
        "gap_seconds": 0,
        "expected_clips": expected_clips,
        "crossfade_duration": crossfade,
        "codec": "",
        "size_gb": 0
    }

    if not path.exists():
        return results

    info = probe_file(path)
    if not info:
        return results

    results["video_duration"] = info.duration
    results["size_gb"] = path.stat().st_size / (1024**3)

    if info.video_stream:
        results["codec"] = getattr(info.video_stream, 'codec_name', 'unknown')

    # Get precise audio duration
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=duration',
        '-of', 'csv=p=0',
        str(path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            results["audio_duration"] = float(result.stdout.strip())
    except Exception:
        results["audio_duration"] = info.duration

    # Check for A/V gap
    gap = abs(results["video_duration"] - results["audio_duration"])
    results["gap_seconds"] = gap
    results["duration_match"] = gap < 1.0

    return results


def main():
    """Main entry point for standalone execution."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Create timeline edits with J-L crossfades"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory (containing clips/)"
    )
    parser.add_argument(
        "--crossfade", "-x",
        type=float,
        default=DEFAULT_CROSSFADE,
        help=f"Crossfade duration in seconds (default: {DEFAULT_CROSSFADE})"
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        help="Specific clips to process (for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process first 3 clips only"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing timelines"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output)
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_dir=output_dir / "logs", level=level)
    logger = get_logger()

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir

    if args.verify_only:
        # Just verify existing timelines
        timeline_dir = output_dir / "timeline"

        for name in ["main", "tripod"]:
            path = timeline_dir / f"dad_cam_{name}_timeline.mov"
            if path.exists():
                results = verify_jl_timeline(path, 0, args.crossfade)
                logger.info(f"\n{name.upper()} TIMELINE:")
                logger.info(f"  Video duration: {results['video_duration']/60:.2f} min")
                logger.info(f"  Audio duration: {results['audio_duration']/60:.2f} min")
                logger.info(f"  Gap: {results['gap_seconds']:.2f}s")
                logger.info(f"  Status: {'OK' if results['duration_match'] else 'DRIFT'}")
                logger.info(f"  Size: {results['size_gb']:.2f} GB")

        return 0

    # Check clips directory
    if not config.clips_dir.exists():
        logger.error(f"Clips directory not found: {config.clips_dir}")
        return 1

    # Create assembler
    assembler = JLAssembler(config, crossfade_duration=args.crossfade)

    # Test mode: only process first 3 clips
    if args.test:
        logger.info("TEST MODE: Processing first 3 clips only")
        all_clips = assembler._gather_clips()
        main_clips = [c for c in all_clips if c.filename.startswith("dad_cam_")][:3]

        if len(main_clips) < 3:
            logger.error(f"Need at least 3 clips for test, found {len(main_clips)}")
            return 1

        test_output = output_dir / "timeline" / "test_jl_3clips.mov"

        logger.info(f"Creating test timeline with clips:")
        for c in main_clips:
            logger.info(f"  - {c.filename} ({c.duration:.1f}s)")

        success, duration = assembler._create_jl_timeline(main_clips, test_output, "test")

        if success:
            logger.info(f"\nTest timeline created: {test_output}")
            logger.info(f"Duration: {duration:.1f}s ({duration/60:.2f} min)")

            # Verify
            results = verify_jl_timeline(test_output, 3, args.crossfade)
            logger.info(f"A/V sync: {'OK' if results['duration_match'] else 'DRIFT'}")
            logger.info(f"Size: {results['size_gb']*1024:.1f} MB")

            return 0
        else:
            logger.error("Test timeline creation failed")
            return 1

    # Full assembly
    with PhaseLogger("J-L Crossfade Assembly", logger):
        results = assembler.assemble()

    # Save results
    save_json(results, config.analysis_dir / "jl_assembly_results.json")

    # Verify outputs
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION")
    logger.info("="*60)

    success = True

    for timeline_key, count_key in [("main_timeline", "main"), ("tripod_timeline", "tripod")]:
        if results.get(timeline_key):
            path = Path(results[timeline_key])
            clip_count = results["clip_counts"].get(count_key, 0)
            verification = verify_jl_timeline(path, clip_count, args.crossfade)

            name = timeline_key.replace("_timeline", "").upper()
            logger.info(f"\n{name}:")
            logger.info(f"  Clips: {clip_count}")
            logger.info(f"  Duration: {verification['video_duration']/60:.1f} minutes")
            logger.info(f"  Audio gap: {verification['gap_seconds']:.2f}s")
            logger.info(f"  Size: {verification['size_gb']:.2f} GB")

            if not verification["duration_match"]:
                logger.warning(f"  WARNING: Audio/video duration mismatch!")
                success = False
            else:
                logger.info(f"  Status: OK")

    if results.get("errors"):
        logger.error(f"\nErrors: {results['errors']}")
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
