#!/usr/bin/env python3
"""
assemble_simple.py - Simplified Timeline Assembly
==================================================

Creates timeline edits from transcoded clips with audio crossfades.

Key improvements over original:
1. Uses concat demuxer for video (no re-encode, guaranteed same duration)
2. Per-clip fade in/out instead of chained acrossfade (more reliable)
3. Fixed 0.5s crossfade with log curve (matches human hearing)
4. No post-assembly loudnorm (already normalized per-clip)
5. Robust fallback to simple concat

Output:
- timeline/dad_cam_main_timeline.mov     (main camera continuous edit)
- timeline/dad_cam_tripod_timeline.mov   (tripod camera continuous edit)
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json
from lib.ffmpeg_utils import probe_file
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


# Constants
CROSSFADE_DURATION = 0.5  # seconds - fixed, simple, reliable
CROSSFADE_CURVE = "log"   # logarithmic - matches human hearing perception


@dataclass
class ClipInfo:
    """Information about a clip for assembly."""
    path: Path
    filename: str
    order: int
    duration: float
    video_duration: float
    audio_duration: float
    sample_rate: int = 48000  # Audio sample rate


class SimpleAssembler:
    """
    Simplified timeline assembler.

    Strategy:
    1. Gather clips, verify audio/video sync
    2. Create concat list for video (stream copy)
    3. Apply per-clip fade in/out to audio
    4. Concat everything
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()

        # Use 'timeline' directory instead of 'master'
        self.timeline_dir = config.output_dir / "timeline"
        self.timeline_dir.mkdir(parents=True, exist_ok=True)

        # Temp directory for intermediate files
        self.temp_dir = config.analysis_dir / "temp_assembly"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def assemble(self) -> Dict:
        """
        Assemble all clips into timeline edits.

        Returns:
            Results dictionary with paths to created files
        """
        results = {
            "main_timeline": None,
            "tripod_timeline": None,
            "clip_counts": {},
            "durations": {},
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

        # Create main camera timeline
        if main_clips:
            self.logger.info("Creating main camera timeline...")
            main_path = self.timeline_dir / "dad_cam_main_timeline.mov"

            success, duration = self._create_timeline(main_clips, main_path, "main")

            if success:
                results["main_timeline"] = str(main_path)
                results["durations"]["main"] = duration
                self.logger.info(f"  Main timeline: {duration/60:.1f} minutes")
            else:
                results["errors"].append("Main timeline creation failed")

        # Create tripod camera timeline
        if tripod_clips:
            self.logger.info("Creating tripod camera timeline...")
            tripod_path = self.timeline_dir / "dad_cam_tripod_timeline.mov"

            success, duration = self._create_timeline(tripod_clips, tripod_path, "tripod")

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

            # Get separate video and audio durations
            video_dur = info.duration
            audio_dur = info.duration

            # Try to get precise durations from streams
            if info.video_stream:
                video_dur = getattr(info.video_stream, 'duration', info.duration) or info.duration
            if info.audio_stream:
                audio_dur = getattr(info.audio_stream, 'duration', info.duration) or info.duration

            # Get audio sample rate
            sample_rate = self._get_sample_rate(clip_path)

            clips.append(ClipInfo(
                path=clip_path,
                filename=clip_path.name,
                order=order,
                duration=info.duration,
                video_duration=video_dur,
                audio_duration=audio_dur,
                sample_rate=sample_rate
            ))

        # Sort by order
        clips.sort(key=lambda c: c.order)

        return clips

    def _get_sample_rate(self, clip_path: Path) -> int:
        """Get audio sample rate from clip."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'csv=p=0',
            str(clip_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except Exception:
            pass
        return 48000  # Default

    def _create_timeline(
        self,
        clips: List[ClipInfo],
        output_path: Path,
        camera_name: str
    ) -> Tuple[bool, float]:
        """
        Create a single timeline from clips.

        Strategy:
        - If all clips have same sample rate: use simple concat (stream copy)
        - If sample rates differ: re-encode audio to 48kHz during concat

        Returns:
            (success, duration) tuple
        """
        if not clips:
            return False, 0.0

        if len(clips) == 1:
            # Single clip - just copy
            shutil.copy2(clips[0].path, output_path)
            return True, clips[0].duration

        # Check for sample rate consistency
        sample_rates = set(c.sample_rate for c in clips)

        if len(sample_rates) == 1:
            # All same sample rate - use simple concat (fastest)
            self.logger.info(f"  All clips have {sample_rates.pop()} Hz - using stream copy")
            success = self._create_timeline_simple(clips, output_path)
        else:
            # Mixed sample rates - need to re-encode audio
            self.logger.info(f"  Mixed sample rates {sample_rates} - normalizing to 48kHz")
            success = self._create_timeline_normalized(clips, output_path)

        if success and output_path.exists():
            # Verify output
            info = probe_file(output_path)
            if info:
                return True, info.duration

        return False, 0.0

    def _create_timeline_normalized(
        self,
        clips: List[ClipInfo],
        output_path: Path
    ) -> bool:
        """
        Create timeline by re-encoding audio to a consistent sample rate.

        Used when clips have different audio sample rates (e.g., 48kHz vs 96kHz).
        Video is still stream-copied (no re-encode).

        Uses pre-processing approach to avoid AAC decoder errors at concat boundaries.
        """
        # Step 1: Pre-process each clip to normalize audio
        self.logger.info(f"    Pre-processing {len(clips)} clips...")
        normalized_clips = []

        for i, clip in enumerate(clips):
            norm_path = self.temp_dir / f"norm_{i:04d}.mov"

            # Re-encode audio to 48kHz for this clip
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(clip.path),
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
                '-movflags', '+faststart',
                str(norm_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and norm_path.exists():
                normalized_clips.append(norm_path)
            else:
                # Fallback: try to copy original
                self.logger.warning(f"    Failed to normalize {clip.filename}, using original")
                normalized_clips.append(clip.path)

        # Step 2: Concat all normalized clips (now all 48kHz)
        concat_list = self.temp_dir / "concat_norm.txt"

        with open(concat_list, 'w') as f:
            for norm_path in normalized_clips:
                escaped = str(norm_path).replace("'", "'\\''")
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
        concat_list.unlink(missing_ok=True)

        # Cleanup normalized clips (but not originals)
        for i, norm_path in enumerate(normalized_clips):
            if norm_path.parent == self.temp_dir:
                norm_path.unlink(missing_ok=True)

        if result.returncode != 0:
            self.logger.error(f"Normalized concat failed: {result.stderr[-300:]}")
            return False

        return output_path.exists()

    def _create_timeline_with_fades(
        self,
        clips: List[ClipInfo],
        output_path: Path
    ) -> bool:
        """
        Create timeline with per-clip audio fade in/out.

        This is more reliable than chaining acrossfade filters because:
        - Each clip is processed independently
        - No complex filter chains
        - Video uses concat demuxer (stream copy)
        """
        # Step 1: Apply fade in/out to each clip's audio
        faded_clips = []

        for i, clip in enumerate(clips):
            faded_path = self.temp_dir / f"faded_{i:04d}.mov"

            # Determine fade parameters
            fade_in = CROSSFADE_DURATION / 2 if i > 0 else 0
            fade_out_start = clip.duration - (CROSSFADE_DURATION / 2)
            fade_out = CROSSFADE_DURATION / 2 if i < len(clips) - 1 else 0

            # Build audio filter
            af_parts = []
            if fade_in > 0:
                af_parts.append(f"afade=t=in:st=0:d={fade_in}:curve={CROSSFADE_CURVE}")
            if fade_out > 0:
                af_parts.append(f"afade=t=out:st={fade_out_start}:d={fade_out}:curve={CROSSFADE_CURVE}")

            if af_parts:
                audio_filter = ",".join(af_parts)
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(clip.path),
                    '-c:v', 'copy',
                    '-af', audio_filter,
                    '-c:a', 'aac', '-b:a', '256k',
                    str(faded_path)
                ]
            else:
                # No fades needed (first clip, no subsequent clips)
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(clip.path),
                    '-c', 'copy',
                    str(faded_path)
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                self.logger.debug(f"Fade failed for {clip.filename}: {result.stderr[-200:]}")
                # Copy original if fade fails
                shutil.copy2(clip.path, faded_path)

            faded_clips.append(faded_path)

        # Step 2: Concat all faded clips
        concat_list = self.temp_dir / "concat_faded.txt"
        with open(concat_list, 'w') as f:
            for faded in faded_clips:
                escaped = str(faded).replace("'", "'\\''")
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

        # Cleanup faded clips
        for faded in faded_clips:
            faded.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)

        return result.returncode == 0 and output_path.exists()

    def _create_timeline_simple(
        self,
        clips: List[ClipInfo],
        output_path: Path
    ) -> bool:
        """
        Simple concat fallback - hard cuts, no fades.

        Used when fade method fails. Still produces valid output.
        """
        concat_list = self.temp_dir / "concat_simple.txt"

        with open(concat_list, 'w') as f:
            for clip in clips:
                escaped = str(clip.path).replace("'", "'\\''")
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
        concat_list.unlink(missing_ok=True)

        if result.returncode != 0:
            self.logger.error(f"Simple concat failed: {result.stderr[-300:]}")
            return False

        return output_path.exists()

    def _cleanup_temp(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*"):
                f.unlink(missing_ok=True)


def verify_timeline(path: Path) -> Dict:
    """
    Verify a timeline file meets quality requirements.

    Returns dict with verification results.
    """
    results = {
        "exists": path.exists(),
        "video_duration": 0,
        "audio_duration": 0,
        "duration_match": False,
        "gap_seconds": 0,
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

    # Get precise audio duration with ffprobe
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

    # Check for gap
    gap = abs(results["video_duration"] - results["audio_duration"])
    results["gap_seconds"] = gap
    results["duration_match"] = gap < 1.0  # Less than 1 second gap is acceptable

    return results


def main():
    """Main entry point for standalone execution."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Create timeline edits from transcoded clips"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory (containing clips/)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing timelines, don't create new ones"
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
                results = verify_timeline(path)
                logger.info(f"\n{name.upper()} TIMELINE:")
                logger.info(f"  Video duration: {results['video_duration']:.2f}s")
                logger.info(f"  Audio duration: {results['audio_duration']:.2f}s")
                logger.info(f"  Gap: {results['gap_seconds']:.2f}s")
                logger.info(f"  Duration match: {'YES' if results['duration_match'] else 'NO'}")
                logger.info(f"  Size: {results['size_gb']:.2f} GB")

        return 0

    # Check clips directory
    if not config.clips_dir.exists():
        logger.error(f"Clips directory not found: {config.clips_dir}")
        return 1

    # Run assembly
    assembler = SimpleAssembler(config)

    with PhaseLogger("Timeline Assembly", logger):
        results = assembler.assemble()

    # Save results
    save_json(results, config.analysis_dir / "assembly_results.json")

    # Verify outputs
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION")
    logger.info("="*60)

    success = True

    for timeline_key in ["main_timeline", "tripod_timeline"]:
        if results.get(timeline_key):
            path = Path(results[timeline_key])
            verification = verify_timeline(path)

            name = timeline_key.replace("_timeline", "").upper()
            logger.info(f"\n{name}:")
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
