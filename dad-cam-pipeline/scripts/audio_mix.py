#!/usr/bin/env python3
"""
audio_mix.py - Intelligent Audio Source Selection
==================================================

Analyzes multiple audio sources and determines which to use at each point
in the timeline.

Priority logic:
1. Lapel/speech mic when active (speeches, vows, ceremony)
2. DJ feed when music is playing (first dances, reception)
3. Best camera audio as fallback (highest SNR)

Output: analysis/audio_mix_decisions.json
"""

import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import struct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, ProgressBar, format_duration
from lib.ffmpeg_utils import probe_file
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class AudioSegment:
    """A segment of audio with source selection."""
    start_time: float
    end_time: float
    duration: float
    primary_source: str  # Path to primary audio source
    source_type: str  # "lapel", "dj", "camera_main", "camera_tripod"
    source_priority: int
    gain_db: float  # Gain adjustment for this source
    duck_others_db: float  # How much to duck other sources
    confidence: float  # Confidence in this selection


@dataclass
class AudioSourceAnalysis:
    """Analysis of an audio source."""
    path: str
    source_type: str
    duration: float
    segments: List[Dict]  # Active/silent segments
    avg_rms: float
    peak_db: float
    noise_floor_db: float
    snr_db: float


class AudioMixAnalyzer:
    """Analyzes audio sources and generates mix decisions."""

    # Thresholds
    SILENCE_THRESHOLD_DB = -50.0  # Below this = silence
    SPEECH_BAND_LOW = 300  # Hz
    SPEECH_BAND_HIGH = 3000  # Hz
    MUSIC_BASS_THRESHOLD = 200  # Hz

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()

    def analyze_and_decide(self, inventory: Dict, sync_data: Dict) -> Dict:
        """
        Analyze all audio sources and generate mix decisions.

        Args:
            inventory: Source inventory with audio files
            sync_data: Synchronization offsets

        Returns:
            Mix decisions for timeline assembly
        """
        results = {
            "sources": [],
            "segments": [],
            "summary": {}
        }

        # Gather all audio sources
        audio_files = inventory.get("audio_sources", [])
        main_clips = inventory.get("main_camera", [])
        tripod_clips = inventory.get("tripod_camera", [])

        if not audio_files and not main_clips:
            self.logger.warning("No audio sources found")
            return results

        self.logger.info("Analyzing audio sources...")

        # Analyze external audio sources
        external_analyses = []
        for audio in audio_files:
            analysis = self._analyze_audio_source(
                audio["original_path"],
                audio.get("audio_type", "unknown"),
                audio.get("audio_priority", 99)
            )
            if analysis:
                external_analyses.append(analysis)
                results["sources"].append(asdict(analysis))

        # Analyze camera audio (sample first clip of each)
        if main_clips:
            main_analysis = self._analyze_audio_source(
                main_clips[0]["original_path"],
                "camera_main",
                3  # Camera priority
            )
            if main_analysis:
                external_analyses.append(main_analysis)
                results["sources"].append(asdict(main_analysis))

        if tripod_clips:
            tripod_analysis = self._analyze_audio_source(
                tripod_clips[0]["original_path"],
                "camera_tripod",
                4  # Lowest priority
            )
            if tripod_analysis:
                external_analyses.append(tripod_analysis)
                results["sources"].append(asdict(tripod_analysis))

        # Get sync offsets
        offsets = self._get_sync_offsets(sync_data)

        # Generate mix decisions
        self.logger.info("Generating mix decisions...")
        segments = self._generate_mix_decisions(external_analyses, offsets, inventory)
        results["segments"] = [asdict(s) for s in segments]

        # Summary
        total_duration = sum(s.duration for s in segments)
        source_breakdown = {}
        for s in segments:
            source_breakdown[s.source_type] = source_breakdown.get(s.source_type, 0) + s.duration

        results["summary"] = {
            "total_duration": total_duration,
            "source_breakdown": source_breakdown,
            "segment_count": len(segments)
        }

        self.logger.info(f"Generated {len(segments)} mix segments")
        for src, dur in source_breakdown.items():
            pct = (dur / total_duration * 100) if total_duration > 0 else 0
            self.logger.info(f"  {src}: {format_duration(dur)} ({pct:.1f}%)")

        # Save results
        save_json(results, self.config.analysis_dir / "audio_mix_decisions.json")

        return results

    def _analyze_audio_source(
        self,
        path: str,
        source_type: str,
        priority: int
    ) -> Optional[AudioSourceAnalysis]:
        """Analyze a single audio source."""
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Audio source not found: {path}")
            return None

        self.logger.info(f"  Analyzing: {path.name} ({source_type})")

        # Get duration
        info = probe_file(path)
        duration = info.duration if info else 0

        # Analyze audio levels
        levels = self._measure_audio_levels(path)

        # Detect active/silent segments
        segments = self._detect_active_segments(path, duration)

        return AudioSourceAnalysis(
            path=str(path),
            source_type=source_type,
            duration=duration,
            segments=segments,
            avg_rms=levels.get("rms", -24),
            peak_db=levels.get("peak", -1),
            noise_floor_db=levels.get("noise_floor", -60),
            snr_db=levels.get("snr", 30)
        )

    def _measure_audio_levels(self, path: Path) -> Dict:
        """Measure audio levels using FFmpeg."""
        cmd = [
            'ffmpeg',
            '-i', str(path),
            '-af', 'astats=metadata=1:reset=1',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse astats output
            levels = {
                "rms": -24.0,
                "peak": -1.0,
                "noise_floor": -60.0,
                "snr": 30.0
            }

            for line in result.stderr.split('\n'):
                if 'RMS level dB' in line:
                    try:
                        levels["rms"] = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
                elif 'Peak level dB' in line:
                    try:
                        levels["peak"] = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass

            # Estimate noise floor from quiet sections
            levels["noise_floor"] = levels["rms"] - 20  # Rough estimate
            levels["snr"] = levels["peak"] - levels["noise_floor"]

            return levels

        except Exception as e:
            self.logger.warning(f"Level measurement failed: {e}")
            return {"rms": -24, "peak": -1, "noise_floor": -60, "snr": 30}

    def _detect_active_segments(self, path: Path, duration: float) -> List[Dict]:
        """
        Detect segments where audio is active (not silent).

        Returns list of {start, end, is_active, avg_level} dicts.
        """
        # Use silencedetect filter
        cmd = [
            'ffmpeg',
            '-i', str(path),
            '-af', f'silencedetect=noise={self.SILENCE_THRESHOLD_DB}dB:d=0.5',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Parse silence periods
            silence_starts = []
            silence_ends = []

            for line in result.stderr.split('\n'):
                if 'silence_start:' in line:
                    try:
                        time = float(line.split('silence_start:')[1].split()[0])
                        silence_starts.append(time)
                    except (IndexError, ValueError):
                        pass
                elif 'silence_end:' in line:
                    try:
                        time = float(line.split('silence_end:')[1].split()[0])
                        silence_ends.append(time)
                    except (IndexError, ValueError):
                        pass

            # Build segments
            segments = []
            current_time = 0.0

            # Pair up silence periods
            for i in range(len(silence_starts)):
                # Active segment before silence
                if silence_starts[i] > current_time:
                    segments.append({
                        "start": current_time,
                        "end": silence_starts[i],
                        "is_active": True
                    })

                # Silence segment
                end_time = silence_ends[i] if i < len(silence_ends) else duration
                segments.append({
                    "start": silence_starts[i],
                    "end": end_time,
                    "is_active": False
                })
                current_time = end_time

            # Final active segment if needed
            if current_time < duration:
                segments.append({
                    "start": current_time,
                    "end": duration,
                    "is_active": True
                })

            # If no silence detected, whole file is active
            if not segments:
                segments = [{"start": 0, "end": duration, "is_active": True}]

            return segments

        except Exception as e:
            self.logger.warning(f"Silence detection failed: {e}")
            return [{"start": 0, "end": duration, "is_active": True}]

    def _get_sync_offsets(self, sync_data: Dict) -> Dict[str, float]:
        """Extract sync offsets from sync data."""
        offsets = {}

        if not sync_data:
            return offsets

        sources = sync_data.get("sources", {})
        for name, data in sources.items():
            offsets[name] = data.get("offset_seconds", 0)

        return offsets

    def _generate_mix_decisions(
        self,
        analyses: List[AudioSourceAnalysis],
        offsets: Dict[str, float],
        inventory: Dict
    ) -> List[AudioSegment]:
        """
        Generate mix decisions based on source analysis.

        Logic:
        1. Find timeline extent (earliest start to latest end)
        2. For each time segment, determine best source:
           - If lapel is active → use lapel, duck cameras to -18dB
           - If DJ is active → use DJ, duck cameras to -24dB
           - Otherwise → use best camera (highest SNR)
        """
        segments = []

        # Sort analyses by priority
        analyses_by_priority = sorted(analyses, key=lambda a: self._get_priority(a.source_type))

        # Get total duration from main camera clips
        main_clips = inventory.get("main_camera", [])
        total_duration = sum(c.get("duration", 0) for c in main_clips)

        if total_duration == 0:
            return segments

        # Process in 1-second chunks
        chunk_duration = 1.0
        current_time = 0.0

        while current_time < total_duration:
            chunk_end = min(current_time + chunk_duration, total_duration)

            # Find best source for this chunk
            best_source = self._select_best_source(
                analyses_by_priority,
                current_time,
                chunk_end,
                offsets
            )

            # Extend previous segment if same source, otherwise create new
            if segments and segments[-1].primary_source == best_source["path"]:
                # Extend
                segments[-1].end_time = chunk_end
                segments[-1].duration = segments[-1].end_time - segments[-1].start_time
            else:
                # New segment
                segments.append(AudioSegment(
                    start_time=current_time,
                    end_time=chunk_end,
                    duration=chunk_end - current_time,
                    primary_source=best_source["path"],
                    source_type=best_source["type"],
                    source_priority=best_source["priority"],
                    gain_db=best_source["gain"],
                    duck_others_db=best_source["duck"],
                    confidence=best_source["confidence"]
                ))

            current_time = chunk_end

        # Merge very short segments (< 2 seconds) with neighbors
        segments = self._merge_short_segments(segments, min_duration=2.0)

        return segments

    def _get_priority(self, source_type: str) -> int:
        """Get numeric priority for source type."""
        priorities = {
            "lapel": 1,
            "dj": 2,
            "camera_main": 3,
            "camera_tripod": 4,
            "ambient": 5,
            "unknown": 99
        }
        return priorities.get(source_type, 99)

    def _select_best_source(
        self,
        analyses: List[AudioSourceAnalysis],
        start_time: float,
        end_time: float,
        offsets: Dict[str, float]
    ) -> Dict:
        """Select the best audio source for a time range."""
        best = {
            "path": "",
            "type": "camera_main",
            "priority": 99,
            "gain": 0.0,
            "duck": 0.0,
            "confidence": 0.5
        }

        for analysis in analyses:
            # Check if source is active at this time
            offset = offsets.get(analysis.source_type, 0)
            source_time = start_time - offset

            if source_time < 0 or source_time > analysis.duration:
                continue

            # Check if this time is in an active segment
            is_active = self._is_active_at_time(analysis.segments, source_time)

            if not is_active:
                continue

            # This source is active - check priority
            priority = self._get_priority(analysis.source_type)

            if priority < best["priority"]:
                best = {
                    "path": analysis.path,
                    "type": analysis.source_type,
                    "priority": priority,
                    "gain": self._get_gain_for_type(analysis.source_type),
                    "duck": self._get_duck_for_type(analysis.source_type),
                    "confidence": 0.8 if is_active else 0.5
                }

        # Fallback to first camera if nothing selected
        if not best["path"]:
            for analysis in analyses:
                if "camera" in analysis.source_type:
                    best = {
                        "path": analysis.path,
                        "type": analysis.source_type,
                        "priority": self._get_priority(analysis.source_type),
                        "gain": 0.0,
                        "duck": 0.0,
                        "confidence": 0.5
                    }
                    break

        return best

    def _is_active_at_time(self, segments: List[Dict], time: float) -> bool:
        """Check if audio is active at a given time."""
        for seg in segments:
            if seg["start"] <= time < seg["end"]:
                return seg.get("is_active", True)
        return False

    def _get_gain_for_type(self, source_type: str) -> float:
        """Get gain adjustment for source type."""
        gains = {
            "lapel": 0.0,  # Already normalized
            "dj": -3.0,  # DJ feeds are often hot
            "camera_main": 0.0,
            "camera_tripod": 0.0,
            "ambient": -6.0
        }
        return gains.get(source_type, 0.0)

    def _get_duck_for_type(self, source_type: str) -> float:
        """Get amount to duck other sources when this is primary."""
        ducks = {
            "lapel": -18.0,  # Duck cameras significantly for speech
            "dj": -24.0,  # Duck cameras more for music
            "camera_main": 0.0,
            "camera_tripod": 0.0,
            "ambient": 0.0
        }
        return ducks.get(source_type, 0.0)

    def _merge_short_segments(
        self,
        segments: List[AudioSegment],
        min_duration: float
    ) -> List[AudioSegment]:
        """Merge segments shorter than min_duration with neighbors."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]

            # If current segment is too short, merge with previous
            if seg.duration < min_duration:
                prev.end_time = seg.end_time
                prev.duration = prev.end_time - prev.start_time
            # If previous is too short and same type, merge
            elif prev.duration < min_duration and prev.source_type == seg.source_type:
                prev.end_time = seg.end_time
                prev.duration = prev.end_time - prev.start_time
            else:
                merged.append(seg)

        return merged


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate intelligent audio mix decisions"
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

    # Load required data
    inventory_path = output_dir / "analysis" / "inventory.json"
    if not inventory_path.exists():
        logger.error("Inventory not found - run 01_discover.py first")
        return 1
    inventory = load_json(inventory_path)

    sync_path = output_dir / "analysis" / "sync_offsets.json"
    sync_data = load_json(sync_path) if sync_path.exists() else {}

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir

    # Run analysis
    analyzer = AudioMixAnalyzer(config)

    with PhaseLogger("Audio Mix Analysis", logger):
        results = analyzer.analyze_and_decide(inventory, sync_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
