#!/usr/bin/env python3
"""
multicam_edit.py - Multicam Edit Decision Generator
====================================================

Generates edit decisions for multicam switching based on:
1. Stability analysis (prefer handheld, switch to tripod when shaky)
2. Shot boundaries (only switch at natural cut points)

Logic:
- Default to handheld camera (main)
- Switch to tripod when handheld stability drops below threshold
- Only switch at or near shot boundaries
- Minimum time between switches (hysteresis)

Output: analysis/multicam_edit_decisions.json
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, format_duration
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


@dataclass
class EditDecision:
    """A single edit decision (angle switch)."""
    timeline_start: float  # Start time on master timeline
    timeline_end: float  # End time on master timeline
    duration: float
    source_camera: str  # "main" or "tripod"
    source_clip: str  # Path to source clip
    source_start: float  # Start time within source clip
    reason: str  # Why this angle was chosen
    stability_score: float  # Stability at this point
    at_shot_boundary: bool  # Whether switch is at a shot boundary


class MulticamEditor:
    """Generates multicam edit decisions."""

    # Thresholds
    STABILITY_SWITCH_DOWN = 0.4  # Switch to tripod when below this
    STABILITY_SWITCH_UP = 0.5  # Switch back to handheld when above this
    MIN_SWITCH_INTERVAL = 5.0  # Minimum seconds between switches
    SHOT_BOUNDARY_WINDOW = 0.5  # Seconds to consider "near" a shot boundary

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()

    def generate_decisions(
        self,
        inventory: Dict,
        stability_data: Dict,
        shot_data: Dict,
        sync_data: Dict
    ) -> Dict:
        """
        Generate multicam edit decisions.

        Args:
            inventory: Source inventory
            stability_data: Stability analysis results
            shot_data: Shot boundary detection results
            sync_data: Synchronization offsets

        Returns:
            Edit decisions for assembly
        """
        results = {
            "decisions": [],
            "summary": {}
        }

        main_clips = inventory.get("main_camera", [])
        tripod_clips = inventory.get("tripod_camera", [])

        if not main_clips:
            self.logger.warning("No main camera clips found")
            return results

        self.logger.info(f"Generating multicam decisions for {len(main_clips)} clips...")

        # Build lookup tables
        stability_lookup = self._build_stability_lookup(stability_data)
        shot_lookup = self._build_shot_boundary_lookup(shot_data)
        sync_offsets = self._get_sync_offsets(sync_data)

        # Build timeline from main camera clips
        timeline = self._build_timeline(main_clips)

        # Determine tripod coverage
        tripod_coverage = self._get_tripod_coverage(tripod_clips, sync_offsets)

        # Generate decisions
        decisions = []
        current_camera = "main"  # Default to handheld
        last_switch_time = -self.MIN_SWITCH_INTERVAL  # Allow immediate first switch

        timeline_position = 0.0

        for clip_info in timeline:
            clip_path = clip_info["path"]
            clip_start = clip_info["timeline_start"]
            clip_duration = clip_info["duration"]
            clip_order = clip_info["order"]

            # Get stability segments for this clip
            stability_segments = stability_lookup.get(clip_path, [])

            # Get shot boundaries for this clip
            shot_boundaries = shot_lookup.get(clip_path, [])

            # Process each stability segment
            segment_start = 0.0

            for seg in stability_segments:
                seg_start = seg.get("start_time", 0)
                seg_end = seg.get("end_time", clip_duration)
                seg_stability = seg.get("stability_score", 0.7)

                timeline_seg_start = clip_start + seg_start
                timeline_seg_end = clip_start + seg_end

                # Check if we should switch cameras
                should_switch, switch_reason = self._should_switch(
                    current_camera,
                    seg_stability,
                    timeline_seg_start,
                    last_switch_time,
                    tripod_coverage
                )

                if should_switch:
                    # Find nearest shot boundary
                    switch_time, at_boundary = self._find_switch_point(
                        timeline_seg_start,
                        clip_start,
                        shot_boundaries
                    )

                    # Only switch if we're at or near a shot boundary
                    # (or if we've been on wrong camera too long)
                    time_since_switch = switch_time - last_switch_time
                    if at_boundary or time_since_switch > 15.0:
                        # Record the previous segment
                        if decisions:
                            decisions[-1]["timeline_end"] = switch_time
                            decisions[-1]["duration"] = switch_time - decisions[-1]["timeline_start"]

                        # Switch cameras
                        new_camera = "tripod" if current_camera == "main" else "main"

                        # Find source clip for new camera
                        source_clip, source_start = self._find_source_at_time(
                            new_camera,
                            switch_time,
                            main_clips if new_camera == "main" else tripod_clips,
                            sync_offsets
                        )

                        decisions.append({
                            "timeline_start": switch_time,
                            "timeline_end": None,  # Will be filled in
                            "duration": None,
                            "source_camera": new_camera,
                            "source_clip": source_clip,
                            "source_start": source_start,
                            "reason": switch_reason,
                            "stability_score": seg_stability,
                            "at_shot_boundary": at_boundary
                        })

                        current_camera = new_camera
                        last_switch_time = switch_time

            timeline_position = clip_start + clip_duration

        # Close final segment
        if decisions:
            decisions[-1]["timeline_end"] = timeline_position
            decisions[-1]["duration"] = timeline_position - decisions[-1]["timeline_start"]
        else:
            # No switches - entire timeline is main camera
            total_duration = sum(c["duration"] for c in timeline)
            decisions.append({
                "timeline_start": 0,
                "timeline_end": total_duration,
                "duration": total_duration,
                "source_camera": "main",
                "source_clip": main_clips[0]["original_path"] if main_clips else "",
                "source_start": 0,
                "reason": "default",
                "stability_score": 0.7,
                "at_shot_boundary": True
            })

        # Convert to dataclass for validation
        results["decisions"] = decisions

        # Summary
        total_duration = sum(d["duration"] or 0 for d in decisions)
        main_duration = sum(d["duration"] or 0 for d in decisions if d["source_camera"] == "main")
        tripod_duration = sum(d["duration"] or 0 for d in decisions if d["source_camera"] == "tripod")
        switch_count = len(decisions) - 1

        results["summary"] = {
            "total_duration": total_duration,
            "main_camera_duration": main_duration,
            "tripod_camera_duration": tripod_duration,
            "main_camera_pct": (main_duration / total_duration * 100) if total_duration > 0 else 0,
            "tripod_camera_pct": (tripod_duration / total_duration * 100) if total_duration > 0 else 0,
            "switch_count": switch_count,
            "avg_segment_duration": total_duration / len(decisions) if decisions else 0
        }

        self.logger.info(f"Generated {len(decisions)} edit segments with {switch_count} switches")
        self.logger.info(f"  Main camera: {format_duration(main_duration)} ({results['summary']['main_camera_pct']:.1f}%)")
        self.logger.info(f"  Tripod camera: {format_duration(tripod_duration)} ({results['summary']['tripod_camera_pct']:.1f}%)")

        # Save results
        save_json(results, self.config.analysis_dir / "multicam_edit_decisions.json")

        return results

    def _build_stability_lookup(self, stability_data: Dict) -> Dict[str, List]:
        """Build lookup table for stability by clip path."""
        lookup = {}
        for clip in stability_data.get("clips", []):
            path = clip.get("clip_path", "")
            segments = clip.get("segments", [])
            lookup[path] = segments
        return lookup

    def _build_shot_boundary_lookup(self, shot_data: Dict) -> Dict[str, List]:
        """Build lookup table for shot boundaries by clip path."""
        lookup = {}
        for clip in shot_data.get("clips", []):
            path = clip.get("clip_path", "")
            shots = clip.get("shots", [])
            # Extract boundary times (end of each shot)
            boundaries = [shot.get("end_time", 0) for shot in shots[:-1]]  # Exclude last
            lookup[path] = boundaries
        return lookup

    def _get_sync_offsets(self, sync_data: Dict) -> Dict[str, float]:
        """Extract sync offsets."""
        offsets = {"main": 0.0, "tripod": 0.0}
        if sync_data:
            sources = sync_data.get("sources", {})
            offsets["main"] = sources.get("main_camera", {}).get("offset_seconds", 0)
            offsets["tripod"] = sources.get("tripod_camera", {}).get("offset_seconds", 0)
        return offsets

    def _build_timeline(self, main_clips: List[Dict]) -> List[Dict]:
        """Build master timeline from main camera clips."""
        timeline = []
        position = 0.0

        for clip in sorted(main_clips, key=lambda c: c.get("order", 0)):
            duration = clip.get("duration", 0)
            timeline.append({
                "path": clip["original_path"],
                "order": clip.get("order", 0),
                "timeline_start": position,
                "duration": duration
            })
            position += duration

        return timeline

    def _get_tripod_coverage(
        self,
        tripod_clips: List[Dict],
        sync_offsets: Dict
    ) -> List[Tuple[float, float]]:
        """
        Get timeline ranges where tripod footage is available.

        Returns list of (start, end) tuples.
        """
        if not tripod_clips:
            return []

        coverage = []
        offset = sync_offsets.get("tripod", 0)

        for clip in tripod_clips:
            duration = clip.get("duration", 0)
            # Tripod clip starts at offset on timeline
            start = offset
            end = offset + duration
            coverage.append((start, end))
            offset = end

        return coverage

    def _should_switch(
        self,
        current_camera: str,
        stability_score: float,
        timeline_time: float,
        last_switch_time: float,
        tripod_coverage: List[Tuple[float, float]]
    ) -> Tuple[bool, str]:
        """
        Determine if we should switch cameras.

        Returns (should_switch, reason)
        """
        # Check minimum interval
        if timeline_time - last_switch_time < self.MIN_SWITCH_INTERVAL:
            return False, ""

        # Check tripod availability
        tripod_available = any(
            start <= timeline_time <= end
            for start, end in tripod_coverage
        )

        if current_camera == "main":
            # Switch to tripod if stability is low and tripod available
            if stability_score < self.STABILITY_SWITCH_DOWN and tripod_available:
                return True, f"stability_low ({stability_score:.2f})"
        else:
            # Switch back to main if stability recovered or tripod not available
            if stability_score > self.STABILITY_SWITCH_UP:
                return True, f"stability_recovered ({stability_score:.2f})"
            if not tripod_available:
                return True, "tripod_unavailable"

        return False, ""

    def _find_switch_point(
        self,
        desired_time: float,
        clip_start: float,
        shot_boundaries: List[float]
    ) -> Tuple[float, bool]:
        """
        Find the best switch point near desired time.

        Returns (switch_time, is_at_boundary)
        """
        clip_relative_time = desired_time - clip_start

        # Find nearest shot boundary
        for boundary in shot_boundaries:
            if abs(boundary - clip_relative_time) <= self.SHOT_BOUNDARY_WINDOW:
                return clip_start + boundary, True

        # No nearby boundary - switch at desired time anyway
        return desired_time, False

    def _find_source_at_time(
        self,
        camera: str,
        timeline_time: float,
        clips: List[Dict],
        sync_offsets: Dict
    ) -> Tuple[str, float]:
        """
        Find the source clip and start time for a camera at a timeline time.

        Returns (clip_path, source_start_time)
        """
        if camera == "main":
            # Main camera clips are sequential on timeline
            position = 0.0
            for clip in sorted(clips, key=lambda c: c.get("order", 0)):
                duration = clip.get("duration", 0)
                if position <= timeline_time < position + duration:
                    return clip["original_path"], timeline_time - position
                position += duration
            # Fallback to last clip
            if clips:
                return clips[-1]["original_path"], 0
        else:
            # Tripod clips - account for sync offset
            offset = sync_offsets.get("tripod", 0)
            position = offset
            for clip in clips:
                duration = clip.get("duration", 0)
                if position <= timeline_time < position + duration:
                    return clip["original_path"], timeline_time - position
                position += duration
            if clips:
                return clips[0]["original_path"], max(0, timeline_time - offset)

        return "", 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multicam edit decisions"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.4,
        help="Stability threshold for switching (default: 0.4)"
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
    analysis_dir = output_dir / "analysis"

    inventory_path = analysis_dir / "inventory.json"
    if not inventory_path.exists():
        logger.error("Inventory not found - run discover first")
        return 1
    inventory = load_json(inventory_path)

    stability_path = analysis_dir / "stability_scores.json"
    stability_data = load_json(stability_path) if stability_path.exists() else {"clips": []}

    shot_path = analysis_dir / "shot_boundaries.json"
    shot_data = load_json(shot_path) if shot_path.exists() else {"clips": []}

    sync_path = analysis_dir / "sync_offsets.json"
    sync_data = load_json(sync_path) if sync_path.exists() else {}

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir

    # Update threshold if specified
    editor = MulticamEditor(config)
    if args.stability_threshold:
        editor.STABILITY_SWITCH_DOWN = args.stability_threshold
        editor.STABILITY_SWITCH_UP = args.stability_threshold + 0.1

    # Generate decisions
    with PhaseLogger("Multicam Edit Decisions", logger):
        results = editor.generate_decisions(inventory, stability_data, shot_data, sync_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
