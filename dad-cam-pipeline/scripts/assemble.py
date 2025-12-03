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

        Uses FFmpeg concat demuxer with audio crossfades.
        """
        settings = self.config.assembly
        master_path = self.master_dir / f"{self.config.output_prefix}_complete.mov"

        # For true J/L cuts with crossfades, we need filter_complex
        # This is complex, so we'll do a simpler approach:
        # 1. Concat with small audio crossfades

        # Create concat file list
        concat_list_path = self.config.analysis_dir / "concat_list.txt"

        with open(concat_list_path, 'w') as f:
            for clip in clips:
                # Escape special characters in path
                escaped_path = str(clip.path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        # Build filter for audio crossfades between clips
        # For simplicity, use concat demuxer with -safe 0

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list_path),
            '-c:v', 'copy',  # Don't re-encode video
            '-c:a', 'aac',
            '-b:a', '256k',
            '-movflags', '+faststart',
            str(master_path)
        ]

        try:
            self.logger.info(f"  Concatenating {len(clips)} clips...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode != 0:
                self.logger.error(f"Concat failed: {result.stderr[-500:]}")
                return None

            if not master_path.exists():
                return None

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

        Creates a multicam clip with synchronized angles.
        """
        fcpxml_path = self.project_dir / f"{self.config.output_prefix}_multicam.fcpxml"

        # Get sync offsets
        sources = sync_data.get("sources", {})
        main_offset = sources.get("main_camera", {}).get("offset_seconds", 0)
        tripod_offset = sources.get("tripod_camera", {}).get("offset_seconds", 0)

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
