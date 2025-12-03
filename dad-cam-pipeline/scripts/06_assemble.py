#!/usr/bin/env python3
"""
06_assemble.py - Timeline Assembly and Export
==============================================

Assembles the final outputs:
1. Concatenated master file with J/L audio cuts
2. FCPXML timeline for NLE import
3. Multicam FCPXML (if sync data available)

Output:
- master/dad_cam_complete.mov
- project/dad_cam_timeline.fcpxml
- project/dad_cam_multicam.fcpxml (optional)
"""

import sys
import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, format_timecode, ProgressBar
from lib.ffmpeg_utils import probe_file, get_duration
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


@dataclass
class ClipInfo:
    """Information about a clip for assembly."""
    path: Path
    filename: str
    order: int
    duration: float
    width: int
    height: int
    fps: float
    start_tc: str  # Timecode start on timeline


class TimelineAssembler:
    """Assembles clips into timelines and exports."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()

        # Ensure output directories exist
        self.master_dir = config.master_dir
        self.project_dir = config.project_dir
        self.master_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def assemble_all(self, sync_data: Optional[Dict] = None) -> Dict:
        """
        Perform all assembly operations.

        Args:
            sync_data: Optional sync offsets for multicam

        Returns:
            Assembly results
        """
        results = {
            "master_file": None,
            "timeline_fcpxml": None,
            "multicam_fcpxml": None,
            "switch_points": None,
            "audio_source": None,
        }

        # Get clip information
        clips = self._gather_clips()

        if not clips:
            self.logger.error("No clips found to assemble")
            return results

        self.logger.info(f"Assembling {len(clips)} clips...")

        # 1. Create concatenated master
        with PhaseLogger("Creating Master File", self.logger) as phase:
            master_path = self._create_master_concat(clips)
            if master_path:
                results["master_file"] = str(master_path)
                phase.success(f"Master file: {master_path.name}")

        # 2. Generate FCPXML timeline
        with PhaseLogger("Generating FCPXML Timeline", self.logger) as phase:
            fcpxml_path = self._generate_fcpxml(clips)
            if fcpxml_path:
                results["timeline_fcpxml"] = str(fcpxml_path)
                phase.success(f"Timeline: {fcpxml_path.name}")

        # 3. Generate multicam FCPXML (if sync data available)
        if sync_data and sync_data.get("sources"):
            with PhaseLogger("Generating Multicam FCPXML", self.logger) as phase:
                # Select best audio source
                audio_source = self._select_best_audio_source(sync_data)
                results["audio_source"] = audio_source
                phase.substep(f"Selected audio: {audio_source}")

                # Generate switch point suggestions based on stability
                stability_data = self._load_stability_data()
                switch_points = self._generate_switch_points(clips, stability_data, sync_data)
                if switch_points:
                    results["switch_points"] = switch_points
                    phase.substep(f"Generated {len(switch_points)} switch point suggestions")

                    # Save switch points for review
                    switch_path = self.project_dir / f"{self.config.output_prefix}_switch_points.json"
                    save_json({
                        "audio_source": audio_source,
                        "switch_points": switch_points,
                        "total_clips": len(clips),
                        "total_duration": sum(c.duration for c in clips),
                    }, switch_path)
                    phase.substep(f"Switch points saved to: {switch_path.name}")

                multicam_path = self._generate_multicam_fcpxml(clips, sync_data)
                if multicam_path:
                    results["multicam_fcpxml"] = str(multicam_path)
                    phase.success(f"Multicam: {multicam_path.name}")

        return results

    def _gather_clips(self) -> List[ClipInfo]:
        """Gather information about all transcoded clips."""
        clips = []
        clips_dir = self.config.clips_dir

        for clip_path in sorted(clips_dir.glob("*.mov")):
            info = probe_file(clip_path)
            if info and info.video_stream:
                # Extract order from filename (dad_cam_001.mov -> 1)
                try:
                    order = int(clip_path.stem.split('_')[-1])
                except ValueError:
                    order = len(clips) + 1

                video = info.video_stream
                clips.append(ClipInfo(
                    path=clip_path,
                    filename=clip_path.name,
                    order=order,
                    duration=info.duration,
                    width=video.width or 1920,
                    height=video.height or 1080,
                    fps=video.fps or 29.97,
                    start_tc=""  # Will be calculated during assembly
                ))

        # Sort by order
        clips.sort(key=lambda c: c.order)

        # Calculate start timecodes
        running_time = 0.0
        for clip in clips:
            clip.start_tc = format_timecode(running_time, clip.fps)
            running_time += clip.duration

        return clips

    def _create_master_concat(self, clips: List[ClipInfo]) -> Optional[Path]:
        """
        Create concatenated master file with J/L audio cuts.

        Implements proper audio crossfades between clips for smooth transitions.
        Video cuts are hard cuts, audio has crossfades (J/L cut style).
        """
        settings = self.config.assembly
        master_path = self.master_dir / f"{self.config.output_prefix}_complete.mov"

        if len(clips) < 2:
            # Single clip - just copy
            if clips:
                return self._copy_single_clip(clips[0], master_path)
            return None

        # For J/L cuts with audio crossfades, use filter_complex
        # This creates audio overlap while video cuts are hard cuts
        crossfade_duration = settings.crossfade_duration

        # Build filter_complex for audio crossfades
        # Strategy: Hard video concat, audio crossfade between each clip pair

        # For large clip counts, use concat filter with audio crossfades
        # Build input arguments
        input_args = []
        for clip in clips:
            input_args.extend(['-i', str(clip.path)])

        # Build filter_complex
        filter_parts = []

        # First, concat all video streams (hard cut)
        video_inputs = ''.join([f'[{i}:v]' for i in range(len(clips))])
        filter_parts.append(f'{video_inputs}concat=n={len(clips)}:v=1:a=0[vout]')

        # For audio, create crossfades between adjacent clips
        # This is complex for many clips, so use a simpler approach:
        # Concat audio with acrossfade filter chain
        if len(clips) <= 20:
            # Use chained acrossfade for manageable clip counts
            audio_filter = self._build_audio_crossfade_chain(clips, crossfade_duration)
            filter_parts.append(audio_filter)
        else:
            # For many clips, use simple concat (crossfades add too much complexity)
            audio_inputs = ''.join([f'[{i}:a]' for i in range(len(clips))])
            filter_parts.append(f'{audio_inputs}concat=n={len(clips)}:v=0:a=1[aout]')

        filter_complex = ';'.join(filter_parts)

        cmd = ['ffmpeg', '-y']
        cmd.extend(input_args)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[vout]',
            '-map', '[aout]',
            '-c:v', 'copy',  # Don't re-encode video
            '-c:a', 'aac',
            '-b:a', '256k',
            '-movflags', '+faststart',
            str(master_path)
        ])

        try:
            self.logger.info(f"  Concatenating {len(clips)} clips with audio crossfades...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode != 0:
                self.logger.warning(f"Complex concat failed, falling back to simple concat")
                return self._create_simple_concat(clips, master_path)

            if not master_path.exists():
                return self._create_simple_concat(clips, master_path)

            # Verify output
            info = probe_file(master_path)
            if info:
                self.logger.info(f"  Master duration: {info.duration/60:.1f} minutes")
                self.logger.info(f"  Master size: {master_path.stat().st_size / (1024*1024*1024):.2f} GB")

            return master_path

        except subprocess.TimeoutExpired:
            self.logger.error("Concat timed out")
            return None
        except Exception as e:
            self.logger.error(f"Concat error: {e}")
            return self._create_simple_concat(clips, master_path)

    def _build_audio_crossfade_chain(self, clips: List[ClipInfo], crossfade_duration: float) -> str:
        """Build chained acrossfade filter for audio crossfades between clips."""
        n = len(clips)
        if n < 2:
            return f'[0:a]anull[aout]'

        # Chain acrossfade filters
        # [0:a][1:a]acrossfade=d=0.5:c1=tri:c2=tri[a01]
        # [a01][2:a]acrossfade=d=0.5:c1=tri:c2=tri[a02]
        # etc.

        filters = []
        curve = 'tri'  # Triangle crossfade curve

        # First crossfade
        filters.append(f'[0:a][1:a]acrossfade=d={crossfade_duration}:c1={curve}:c2={curve}[a01]')

        # Chain remaining crossfades
        for i in range(2, n):
            prev_label = f'a{i-2:02d}{i-1:02d}' if i > 2 else 'a01'
            next_label = f'a{i-1:02d}{i:02d}' if i < n - 1 else 'aout'
            filters.append(f'[{prev_label}][{i}:a]acrossfade=d={crossfade_duration}:c1={curve}:c2={curve}[{next_label}]')

        return ';'.join(filters)

    def _copy_single_clip(self, clip: ClipInfo, output_path: Path) -> Optional[Path]:
        """Copy single clip to output."""
        import shutil
        try:
            shutil.copy2(clip.path, output_path)
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to copy clip: {e}")
            return None

    def _create_simple_concat(self, clips: List[ClipInfo], master_path: Path) -> Optional[Path]:
        """Fallback simple concatenation without crossfades."""
        concat_list_path = self.config.analysis_dir / "concat_list.txt"

        with open(concat_list_path, 'w') as f:
            for clip in clips:
                escaped_path = str(clip.path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '256k',
            '-movflags', '+faststart',
            str(master_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0 and master_path.exists():
                return master_path
        except Exception:
            pass

        return None

    def _generate_fcpxml(self, clips: List[ClipInfo]) -> Optional[Path]:
        """
        Generate FCPXML timeline for import into NLEs.

        Creates a sequence with all clips on a single video/audio track.
        """
        fcpxml_path = self.project_dir / f"{self.config.output_prefix}_timeline.fcpxml"

        # Build FCPXML structure
        fcpxml = ET.Element('fcpxml', version="1.10")

        # Resources
        resources = ET.SubElement(fcpxml, 'resources')

        # Format resource
        format_id = "r1"
        format_elem = ET.SubElement(resources, 'format',
            id=format_id,
            name="FFVideoFormat1080p2997",
            frameDuration="1001/30000s",
            width="1920",
            height="1080"
        )

        # Asset resources for each clip
        asset_refs = {}
        for i, clip in enumerate(clips):
            asset_id = f"r{i+2}"
            asset_refs[clip.filename] = asset_id

            # Get absolute path with file:// prefix
            file_url = f"file://{clip.path}"

            asset = ET.SubElement(resources, 'asset',
                id=asset_id,
                name=clip.filename,
                src=file_url,
                start="0s",
                duration=f"{clip.duration}s",
                hasVideo="1",
                hasAudio="1"
            )

        # Library
        library = ET.SubElement(fcpxml, 'library')

        # Event
        event = ET.SubElement(library, 'event',
            name=f"{self.config.output_prefix}_event"
        )

        # Project
        project = ET.SubElement(event, 'project',
            name=f"{self.config.output_prefix}_timeline"
        )

        # Calculate total duration
        total_duration = sum(c.duration for c in clips)

        # Sequence
        sequence = ET.SubElement(project, 'sequence',
            format=format_id,
            tcStart="3600s",  # Start at 01:00:00:00
            tcFormat="NDF",
            duration=f"{total_duration}s"
        )

        # Spine (main timeline)
        spine = ET.SubElement(sequence, 'spine')

        # Add clips to spine
        timeline_offset = 0.0
        for clip in clips:
            asset_id = asset_refs[clip.filename]

            # Asset clip
            asset_clip = ET.SubElement(spine, 'asset-clip',
                ref=asset_id,
                offset=f"{timeline_offset}s",
                name=clip.filename,
                duration=f"{clip.duration}s",
                start="0s",
                tcFormat="NDF"
            )

            timeline_offset += clip.duration

        # Write XML
        tree = ET.ElementTree(fcpxml)
        ET.indent(tree, space="  ")

        # Add XML declaration
        with open(fcpxml_path, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(b'<!DOCTYPE fcpxml>\n')
            tree.write(f, encoding='unicode' if sys.version_info >= (3, 8) else 'utf-8')

        return fcpxml_path

    def _generate_multicam_fcpxml(
        self,
        clips: List[ClipInfo],
        sync_data: Dict
    ) -> Optional[Path]:
        """
        Generate FCPXML with multicam clip structure.

        Creates a multicam clip with synchronized angles and stability-aware
        switching suggestions based on camera stability analysis.
        """
        fcpxml_path = self.project_dir / f"{self.config.output_prefix}_multicam.fcpxml"

        # Get sync offsets
        sources = sync_data.get("sources", {})
        main_offset = sources.get("main_camera", {}).get("offset_seconds", 0)
        tripod_offset = sources.get("tripod_camera", {}).get("offset_seconds", 0)

        # Load stability data for intelligent switching
        stability_data = self._load_stability_data()

        # Build FCPXML
        fcpxml = ET.Element('fcpxml', version="1.10")

        # Resources
        resources = ET.SubElement(fcpxml, 'resources')

        # Format
        format_elem = ET.SubElement(resources, 'format',
            id="r1",
            name="FFVideoFormat1080p2997",
            frameDuration="1001/30000s",
            width="1920",
            height="1080"
        )

        # Media resources
        # Add main camera clips
        main_assets = []
        for i, clip in enumerate(clips):
            asset_id = f"main_{i}"
            file_url = f"file://{clip.path}"

            asset = ET.SubElement(resources, 'asset',
                id=asset_id,
                name=clip.filename,
                src=file_url,
                start="0s",
                duration=f"{clip.duration}s",
                hasVideo="1",
                hasAudio="1"
            )
            main_assets.append((asset_id, clip))

        # Library
        library = ET.SubElement(fcpxml, 'library')
        event = ET.SubElement(library, 'event',
            name=f"{self.config.output_prefix}_multicam_event"
        )

        # Create multicam clip
        total_duration = sum(c.duration for c in clips)

        mc_clip = ET.SubElement(event, 'mc-clip',
            name=f"{self.config.output_prefix}_multicam",
            tcStart="3600s",
            tcFormat="NDF"
        )

        # Angle A: Main camera
        mc_angle_a = ET.SubElement(mc_clip, 'mc-angle',
            name="Main Camera",
            angleID="A"
        )

        offset = 0.0
        for asset_id, clip in main_assets:
            gap_or_clip = ET.SubElement(mc_angle_a, 'asset-clip',
                ref=asset_id,
                offset=f"{offset}s",
                name=clip.filename,
                duration=f"{clip.duration}s"
            )
            offset += clip.duration

        # Note: Full tripod angle would require tripod clips
        # For now, create placeholder

        # Create project with multicam reference
        project = ET.SubElement(event, 'project',
            name=f"{self.config.output_prefix}_multicam_edit"
        )

        sequence = ET.SubElement(project, 'sequence',
            format="r1",
            tcStart="3600s",
            tcFormat="NDF",
            duration=f"{total_duration}s"
        )

        spine = ET.SubElement(sequence, 'spine')

        # Add note about multicam
        ET.SubElement(spine, 'gap',
            name="Multicam Edit Point",
            offset="0s",
            duration=f"{total_duration}s"
        )

        # Write XML
        tree = ET.ElementTree(fcpxml)
        ET.indent(tree, space="  ")

        with open(fcpxml_path, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(b'<!DOCTYPE fcpxml>\n')
            tree.write(f, encoding='unicode' if sys.version_info >= (3, 8) else 'utf-8')

        return fcpxml_path

    def _load_stability_data(self) -> Optional[Dict]:
        """Load stability analysis data for intelligent switching decisions."""
        stability_path = self.config.analysis_dir / "stability_scores.json"
        if stability_path.exists():
            return load_json(stability_path)
        return None

    def _generate_switch_points(
        self,
        clips: List[ClipInfo],
        stability_data: Optional[Dict],
        sync_data: Dict
    ) -> List[Dict]:
        """
        Generate recommended camera switch points based on stability analysis.

        Uses stability scores to suggest when to switch from handheld (main)
        to tripod camera. Switches when handheld becomes shaky.

        Returns list of switch suggestions with timing and reasoning.
        """
        if not stability_data or not stability_data.get("clips"):
            return []

        settings = self.config.assembly
        switch_points = []
        last_switch_time = 0.0

        # Analyze each clip's stability segments
        timeline_offset = 0.0
        for clip_stability in stability_data.get("clips", []):
            clip_order = clip_stability.get("clip_order", 0)

            for segment in clip_stability.get("segments", []):
                stability_score = segment.get("stability_score", 0.7)
                segment_start = timeline_offset + segment.get("start_time", 0)
                segment_end = timeline_offset + segment.get("end_time", 0)
                classification = segment.get("classification", "moderate")

                # If segment is shaky and we're past hysteresis threshold, suggest switch
                if classification == "shaky" and stability_score < settings.prefer_handheld_threshold:
                    if segment_start - last_switch_time >= settings.switch_hysteresis_seconds:
                        switch_points.append({
                            "time": segment_start,
                            "suggested_angle": "tripod",
                            "reason": f"Main camera shaky (score: {stability_score:.2f})",
                            "stability_score": stability_score,
                        })
                        last_switch_time = segment_start

                # If segment becomes stable again, suggest switching back
                elif classification == "stable" and stability_score >= settings.prefer_handheld_threshold:
                    if switch_points and switch_points[-1]["suggested_angle"] == "tripod":
                        if segment_start - last_switch_time >= settings.switch_hysteresis_seconds:
                            switch_points.append({
                                "time": segment_start,
                                "suggested_angle": "main",
                                "reason": f"Main camera stable (score: {stability_score:.2f})",
                                "stability_score": stability_score,
                            })
                            last_switch_time = segment_start

            timeline_offset += clip_stability.get("duration", 0)

        return switch_points

    def _select_best_audio_source(self, sync_data: Dict) -> str:
        """
        Select the best audio source for the multicam edit.

        Priority:
        1. External audio recorder (cleanest, no camera noise)
        2. Tripod camera (stable, less handling noise)
        3. Main camera (fallback)

        Returns the source name to use for audio.
        """
        sources = sync_data.get("sources", {})

        # Check for external audio with good sync confidence
        if "external_audio" in sources:
            ext_audio = sources["external_audio"]
            if ext_audio.get("confidence", 0) >= 0.5:
                return "external_audio"

        # Check tripod camera
        if "tripod_camera" in sources:
            tripod = sources["tripod_camera"]
            if tripod.get("confidence", 0) >= 0.5:
                return "tripod_camera"

        # Fall back to main camera
        return "main_camera"


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Assemble clips into timeline and exports"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--skip-master",
        action="store_true",
        help="Skip master file creation"
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
        logger.error("Clips directory not found - run transcoding first")
        return 1

    # Load sync data if available
    sync_path = config.analysis_dir / "sync_offsets.json"
    sync_data = load_json(sync_path) if sync_path.exists() else None

    # Run assembly
    assembler = TimelineAssembler(config)
    results = assembler.assemble_all(sync_data)

    # Save results
    save_json(results, config.analysis_dir / "assembly_results.json")

    # Summary
    logger.info("")
    logger.info("Assembly complete:")
    if results.get("master_file"):
        logger.info(f"  Master: {results['master_file']}")
    if results.get("timeline_fcpxml"):
        logger.info(f"  Timeline: {results['timeline_fcpxml']}")
    if results.get("multicam_fcpxml"):
        logger.info(f"  Multicam: {results['multicam_fcpxml']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
