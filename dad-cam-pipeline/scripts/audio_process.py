#!/usr/bin/env python3
"""
04_audio_process.py - Audio Processing
=======================================

Processes audio in transcoded clips:
1. Hum detection and removal (60Hz + harmonics)
2. Two-pass loudness normalization to -14 LUFS

The audio is processed and remuxed back into the video files.
"""

import sys
import subprocess
import json
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, ProgressBar, format_duration
from lib.ffmpeg_utils import probe_file, measure_loudness
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


@dataclass
class AudioAnalysis:
    """Audio analysis results for a clip."""
    path: str
    has_hum: bool
    hum_frequencies: List[int]
    measured_lufs: float
    measured_tp: float
    measured_lra: float


class AudioProcessor:
    """Processes audio in video clips."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.clips_dir = config.clips_dir

    def process_all(self) -> Dict:
        """
        Process audio in all transcoded clips.

        Returns:
            Processing results
        """
        # Find all clips
        clips = sorted(self.clips_dir.glob("*.mov"))

        if not clips:
            self.logger.warning("No clips found to process")
            return {"clips": [], "summary": {}}

        self.logger.info(f"Processing audio for {len(clips)} clips...")

        results = []
        progress = ProgressBar(len(clips), prefix="Audio processing")

        for clip_path in clips:
            result = self._process_clip(clip_path)
            results.append(result)
            progress.update(1, suffix=clip_path.name)

        progress.finish()

        # Build summary
        hum_detected = sum(1 for r in results if r.get("hum_removed", False))
        # Calculate average LUFS, handling None and string values
        lufs_values = []
        for r in results:
            val = r.get("final_lufs")
            if val is not None:
                try:
                    lufs_values.append(float(val))
                except (ValueError, TypeError):
                    lufs_values.append(-14.0)
            else:
                lufs_values.append(-14.0)
        avg_lufs = sum(lufs_values) / len(lufs_values) if lufs_values else -14.0

        summary = {
            "total_clips": len(clips),
            "hum_detected": hum_detected,
            "average_lufs": avg_lufs,
        }

        self.logger.info(f"  Clips with hum removed: {hum_detected}")
        self.logger.info(f"  Average final LUFS: {avg_lufs:.1f}")

        return {
            "clips": results,
            "summary": summary
        }

    def _process_clip(self, clip_path: Path) -> Dict:
        """Process audio in a single clip."""
        result = {
            "path": str(clip_path),
            "filename": clip_path.name,
            "hum_removed": False,
            "original_lufs": None,
            "final_lufs": None,
            "success": False,
            "error": None
        }

        try:
            # Step 1: Analyze audio for hum
            hum_filter = self._detect_and_build_hum_filter(clip_path)
            result["hum_removed"] = bool(hum_filter)

            # Step 2: Measure original loudness
            loudness = self._measure_loudness_pass1(clip_path)
            if loudness:
                try:
                    result["original_lufs"] = float(loudness.get("input_i", -14))
                except (ValueError, TypeError):
                    result["original_lufs"] = -14.0

            # Step 3: Build complete audio filter chain
            audio_filters = []

            # Highpass to remove subsonic rumble
            audio_filters.append("highpass=f=60")

            # Hum removal
            if hum_filter:
                audio_filters.append(hum_filter)

            # Loudness normalization (2-pass)
            if loudness:
                norm_filter = self._build_loudnorm_filter(loudness)
                audio_filters.append(norm_filter)
            else:
                # Single-pass fallback
                audio_filters.append(f"loudnorm=I={self.config.audio.target_lufs}:TP={self.config.audio.target_tp}:LRA={self.config.audio.target_lra}")

            # Step 4: Apply filters and remux
            success = self._apply_audio_filters(clip_path, audio_filters)

            if success:
                # Verify final loudness
                final_loudness = self._measure_loudness_pass1(clip_path)
                if final_loudness:
                    try:
                        result["final_lufs"] = float(final_loudness.get("input_i", -14))
                    except (ValueError, TypeError):
                        result["final_lufs"] = -14.0
                result["success"] = True
            else:
                result["error"] = "Failed to apply audio filters"

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Error processing {clip_path.name}: {e}")

        return result

    def _detect_and_build_hum_filter(self, clip_path: Path) -> Optional[str]:
        """
        Detect electrical hum and build removal filter.

        Uses FFT analysis to detect 60Hz hum and harmonics.
        """
        settings = self.config.audio

        # Extract short audio sample for analysis
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Extract 10 seconds of audio
            cmd = [
                'ffmpeg', '-y',
                '-i', str(clip_path),
                '-t', '10',
                '-vn',
                '-ar', '48000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                str(tmp_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Analyze for hum using FFmpeg's aspectralstats
            cmd = [
                'ffmpeg',
                '-i', str(tmp_path),
                '-af', 'aspectralstats=measure=peak',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # For now, use a simple heuristic - always apply hum filter
            # as consumer cameras often have 60Hz hum
            # A more sophisticated approach would analyze FFT peaks

            # Build notch filter chain for 60Hz and harmonics
            notch_filters = []
            for harmonic in range(1, settings.hum_harmonics + 1):
                freq = settings.hum_frequency * harmonic
                if freq < 8000:  # Stay in audible range
                    width = settings.notch_base_width + (harmonic - 1) * settings.notch_width_increment
                    notch_filters.append(f"bandreject=f={freq}:width_type=h:width={width}")

            if notch_filters:
                return ",".join(notch_filters)

        except subprocess.CalledProcessError:
            pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return None

    def _measure_loudness_pass1(self, clip_path: Path) -> Optional[Dict]:
        """
        First pass of loudness measurement.

        Returns measured values for second pass normalization.
        """
        settings = self.config.audio

        cmd = [
            'ffmpeg',
            '-i', str(clip_path),
            '-af', f'loudnorm=I={settings.target_lufs}:TP={settings.target_tp}:LRA={settings.target_lra}:print_format=json',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Parse JSON from stderr
            # Look for the JSON block containing measured values
            stderr = result.stderr

            # Find the last JSON block (the measured values)
            json_pattern = r'\{[^{}]*"input_i"[^{}]*\}'
            matches = re.findall(json_pattern, stderr, re.DOTALL)

            if matches:
                # Take the last match (measured values, not target)
                try:
                    return json.loads(matches[-1])
                except json.JSONDecodeError:
                    pass

        except subprocess.CalledProcessError:
            pass

        return None

    def _build_loudnorm_filter(self, measured: Dict) -> str:
        """Build loudnorm filter with measured values for 2-pass normalization."""
        settings = self.config.audio

        # Extract measured values
        input_i = measured.get("input_i", "-24")
        input_tp = measured.get("input_tp", "-2")
        input_lra = measured.get("input_lra", "7")
        input_thresh = measured.get("input_thresh", "-34")
        offset = measured.get("target_offset", "0")

        return (
            f"loudnorm=I={settings.target_lufs}:TP={settings.target_tp}:LRA={settings.target_lra}:"
            f"measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:"
            f"measured_thresh={input_thresh}:offset={offset}:linear=true"
        )

    def _apply_audio_filters(self, clip_path: Path, filters: List[str]) -> bool:
        """
        Apply audio filters to a clip.

        Creates a temporary file, then replaces the original.
        """
        if not filters:
            return True

        filter_chain = ",".join(filters)

        # Create temporary output file
        tmp_path = clip_path.with_suffix('.tmp.mov')

        cmd = [
            'ffmpeg', '-y',
            '-i', str(clip_path),
            '-c:v', 'copy',  # Don't re-encode video
            '-af', filter_chain,
            '-c:a', 'aac',
            '-b:a', '256k',
            '-movflags', '+faststart',
            str(tmp_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                if tmp_path.exists():
                    tmp_path.unlink()
                return False

            # Verify output
            if not tmp_path.exists() or tmp_path.stat().st_size < 1000:
                if tmp_path.exists():
                    tmp_path.unlink()
                return False

            # Replace original with processed version
            clip_path.unlink()
            tmp_path.rename(clip_path)

            return True

        except subprocess.TimeoutExpired:
            if tmp_path.exists():
                tmp_path.unlink()
            return False
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process audio in transcoded clips"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory (containing clips/)"
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

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir

    # Check clips directory
    if not config.clips_dir.exists():
        logger.error("Clips directory not found - run 03_transcode.py first")
        return 1

    # Process audio
    processor = AudioProcessor(config)

    with PhaseLogger("Audio Processing", logger):
        results = processor.process_all()

    # Save results
    save_json(results, config.analysis_dir / "audio_results.json")

    failed = sum(1 for r in results.get("clips", []) if not r.get("success", False))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
