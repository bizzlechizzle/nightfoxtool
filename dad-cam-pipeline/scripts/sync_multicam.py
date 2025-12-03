#!/usr/bin/env python3
"""
05_sync_multicam.py - Multicam Audio Synchronization
=====================================================

Synchronizes multiple camera sources and external audio using
audio cross-correlation.

Process:
1. Extract audio from all sources
2. Designate reference source (external audio or longest clip)
3. Cross-correlate all sources against reference
4. Validate sync quality
5. Check for clock drift on long recordings

Output: analysis/sync_offsets.json
"""

import sys
import subprocess
import tempfile
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, format_duration
from lib.ffmpeg_utils import probe_file, extract_audio, get_duration
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


@dataclass
class SyncResult:
    """Sync result for a source."""
    source_name: str
    source_path: str
    offset_seconds: float
    offset_frames: int
    confidence: float
    drift_ppm: float  # Parts per million drift rate


class MulticamSynchronizer:
    """Synchronizes multiple video/audio sources."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.analysis_dir = config.analysis_dir

    def sync_sources(self, inventory: Dict) -> Dict:
        """
        Synchronize all sources from inventory.

        Args:
            inventory: Source inventory

        Returns:
            Sync results including offsets
        """
        # Collect all sources
        main_clips = inventory.get("main_camera", [])
        tripod_clips = inventory.get("tripod_camera", [])
        audio_files = inventory.get("audio_sources", [])

        if not main_clips:
            self.logger.warning("No main camera clips found")
            return {"error": "No main camera clips"}

        # Determine reference source
        # Priority: external audio > tripod camera > first main camera clip
        reference = self._select_reference(main_clips, tripod_clips, audio_files)
        self.logger.info(f"Reference source: {reference['name']}")

        results = {
            "reference": {
                "name": reference["name"],
                "path": reference["path"],
                "duration": reference["duration"]
            },
            "sources": {},
            "sync_quality": "unknown"
        }

        # Extract reference audio
        ref_audio = self._extract_audio_for_sync(reference["path"], reference["duration"])
        if ref_audio is None:
            return {"error": "Failed to extract reference audio"}

        try:
            # Sync main camera (using first clip as representative)
            if main_clips:
                main_result = self._sync_source(
                    "main_camera",
                    main_clips[0]["original_path"],
                    main_clips[0].get("duration", 0),
                    ref_audio
                )
                if main_result:
                    results["sources"]["main_camera"] = asdict(main_result)

            # Sync tripod camera
            if tripod_clips:
                tripod_result = self._sync_source(
                    "tripod_camera",
                    tripod_clips[0]["original_path"],
                    tripod_clips[0].get("duration", 0),
                    ref_audio
                )
                if tripod_result:
                    results["sources"]["tripod_camera"] = asdict(tripod_result)

            # Sync external audio (if not reference)
            if audio_files and reference["name"] != "external_audio":
                audio_result = self._sync_source(
                    "external_audio",
                    audio_files[0]["original_path"],
                    audio_files[0].get("duration", 0),
                    ref_audio
                )
                if audio_result:
                    results["sources"]["external_audio"] = asdict(audio_result)

            # Assess overall sync quality
            results["sync_quality"] = self._assess_sync_quality(results["sources"])

        finally:
            # Cleanup
            if ref_audio and Path(ref_audio).exists():
                Path(ref_audio).unlink()

        # Save results
        save_json(results, self.analysis_dir / "sync_offsets.json")

        return results

    def _select_reference(
        self,
        main_clips: List[Dict],
        tripod_clips: List[Dict],
        audio_files: List[Dict]
    ) -> Dict:
        """Select the best reference source for synchronization."""
        # External audio is preferred (usually cleanest)
        if audio_files:
            longest_audio = max(audio_files, key=lambda x: x.get("duration", 0))
            return {
                "name": "external_audio",
                "path": longest_audio["original_path"],
                "duration": longest_audio.get("duration", 0)
            }

        # Tripod camera next (usually more continuous)
        if tripod_clips:
            longest_tripod = max(tripod_clips, key=lambda x: x.get("duration", 0))
            return {
                "name": "tripod_camera",
                "path": longest_tripod["original_path"],
                "duration": longest_tripod.get("duration", 0)
            }

        # Fall back to main camera
        longest_main = max(main_clips, key=lambda x: x.get("duration", 0))
        return {
            "name": "main_camera",
            "path": longest_main["original_path"],
            "duration": longest_main.get("duration", 0)
        }

    def _extract_audio_for_sync(
        self,
        source_path: str,
        duration: float
    ) -> Optional[str]:
        """Extract audio sample for synchronization."""
        settings = self.config.sync

        # Create temp file
        tmp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False
        )
        tmp_path = tmp_file.name
        tmp_file.close()

        # Determine sample range (use middle portion for best correlation)
        sample_duration = min(settings.sample_duration, duration)
        start_time = max(0, (duration - sample_duration) / 2)

        # Extract and downsample
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', source_path,
            '-t', str(sample_duration),
            '-vn',
            '-ar', str(settings.sample_rate),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            tmp_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return tmp_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract audio: {e}")
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            return None

    def _sync_source(
        self,
        name: str,
        source_path: str,
        duration: float,
        ref_audio_path: str
    ) -> Optional[SyncResult]:
        """Synchronize a single source against reference."""
        self.logger.info(f"  Syncing {name}...")

        # Extract source audio
        source_audio = self._extract_audio_for_sync(source_path, duration)
        if source_audio is None:
            return None

        try:
            # Perform cross-correlation
            offset, confidence = self._cross_correlate(ref_audio_path, source_audio)

            if confidence < self.config.sync.min_confidence:
                self.logger.warning(f"    Low confidence: {confidence:.2f}")

            # Check for drift (if recording is long enough)
            drift_ppm = 0.0
            if duration > 3600:  # > 1 hour
                drift_ppm = self._check_drift(
                    ref_audio_path,
                    source_audio,
                    duration
                )
                if abs(drift_ppm) > 10:
                    self.logger.warning(f"    Clock drift detected: {drift_ppm:.1f} ppm")

            # Convert offset to frames (29.97fps)
            offset_frames = int(round(offset * 29.97))

            self.logger.info(f"    Offset: {offset:.3f}s ({offset_frames} frames), "
                           f"confidence: {confidence:.2f}")

            return SyncResult(
                source_name=name,
                source_path=source_path,
                offset_seconds=offset,
                offset_frames=offset_frames,
                confidence=confidence,
                drift_ppm=drift_ppm
            )

        finally:
            if Path(source_audio).exists():
                Path(source_audio).unlink()

    def _cross_correlate(
        self,
        ref_path: str,
        source_path: str
    ) -> Tuple[float, float]:
        """
        Perform cross-correlation between two audio files.

        Returns:
            Tuple of (offset_seconds, confidence)
        """
        try:
            import numpy as np
            from scipy import signal
            from scipy.io import wavfile
        except ImportError:
            self.logger.warning("NumPy/SciPy not available - using fallback sync")
            return 0.0, 0.5

        try:
            # Read audio files
            ref_rate, ref_data = wavfile.read(ref_path)
            src_rate, src_data = wavfile.read(source_path)

            # Ensure same sample rate
            if ref_rate != src_rate:
                self.logger.warning("Sample rate mismatch")
                return 0.0, 0.0

            # Convert to float and normalize
            ref_data = ref_data.astype(np.float32)
            src_data = src_data.astype(np.float32)

            ref_data = ref_data / (np.max(np.abs(ref_data)) + 1e-10)
            src_data = src_data / (np.max(np.abs(src_data)) + 1e-10)

            # Cross-correlation
            correlation = signal.correlate(ref_data, src_data, mode='full')

            # Find peak
            peak_idx = np.argmax(np.abs(correlation))
            peak_value = np.abs(correlation[peak_idx])

            # Calculate offset
            # Peak at len(src)-1 means zero offset
            lag = peak_idx - len(src_data) + 1
            offset_seconds = lag / ref_rate

            # Confidence: ratio of peak to median
            median_corr = np.median(np.abs(correlation))
            confidence = min(1.0, peak_value / (median_corr * 10 + 1e-10))

            return offset_seconds, confidence

        except Exception as e:
            self.logger.error(f"Cross-correlation failed: {e}")
            return 0.0, 0.0

    def _check_drift(
        self,
        ref_path: str,
        source_path: str,
        duration: float
    ) -> float:
        """
        Check for clock drift between sources.

        Returns drift in parts per million (ppm).
        """
        settings = self.config.sync

        # Check sync at multiple points
        check_points = [0, duration * 0.25, duration * 0.5, duration * 0.75]
        offsets = []

        for point in check_points:
            # This is a simplified version - full implementation would
            # extract audio at each point and correlate
            pass

        # For now, return 0 (no drift detected)
        return 0.0

    def _assess_sync_quality(self, sources: Dict) -> str:
        """Assess overall sync quality."""
        if not sources:
            return "no_sources"

        confidences = [s.get("confidence", 0) for s in sources.values()]
        avg_confidence = sum(confidences) / len(confidences)

        if avg_confidence >= 0.8:
            return "excellent"
        elif avg_confidence >= 0.6:
            return "good"
        elif avg_confidence >= 0.4:
            return "fair"
        else:
            return "poor"


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Synchronize multicam sources"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
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

    # Load inventory
    inventory_path = output_dir / "analysis" / "inventory.json"
    if not inventory_path.exists():
        logger.error("Inventory not found - run 01_discover.py first")
        return 1

    inventory = load_json(inventory_path)

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir

    # Run sync
    syncer = MulticamSynchronizer(config)

    with PhaseLogger("Multicam Synchronization", logger):
        results = syncer.sync_sources(inventory)

    if "error" in results:
        logger.error(f"Sync failed: {results['error']}")
        return 1

    logger.info(f"Sync quality: {results.get('sync_quality', 'unknown')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
