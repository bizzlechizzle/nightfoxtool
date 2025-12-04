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
            "doc_edits": {},
        }

        # Get clip information
        clips = self._gather_clips()

        if not clips:
            self.logger.error("No clips found to assemble")
            return results

        self.logger.info(f"Assembling {len(clips)} clips...")

        # 0. Create per-camera doc edits (VHS-style continuous viewing)
        with PhaseLogger("Creating Per-Camera Doc Edits", self.logger) as phase:
            doc_edits = self._create_doc_edits(clips)
            results["doc_edits"] = {k: str(v) for k, v in doc_edits.items()}
            if doc_edits:
                phase.success(f"Created {len(doc_edits)} doc edits")
            else:
                phase.success("No doc edits created (no camera-specific clips found)")

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
        Create concatenated master file with J/L audio crossfades.

        Uses FFmpeg filter_complex for proper audio crossfades:
        - Video: Hard cuts (no transitions)
        - Audio: 0.25s linear crossfade at each cut point

        For very long timelines (>20 clips), falls back to simple concat
        to avoid filter_complex limitations.
        """
        settings = self.config.assembly
        master_path = self.master_dir / f"{self.config.output_prefix}_complete.mov"
        crossfade_duration = 0.25  # 8 frames at 29.97fps

        # For many clips, use batched approach to avoid filter_complex limits
        if len(clips) > 20:
            return self._create_master_batched(clips, master_path, crossfade_duration)

        # Build filter_complex for audio crossfades
        self.logger.info(f"  Creating master with {len(clips)} clips and audio crossfades...")

        # Build input arguments
        input_args = []
        for clip in clips:
            input_args.extend(['-i', str(clip.path)])

        # Build filter_complex string
        # Strategy: Chain audio crossfades while copying video
        filter_parts = []

        # For N clips, we need N-1 crossfades
        # Each crossfade takes two audio streams and produces one

        if len(clips) == 1:
            # Single clip - just copy
            filter_complex = "[0:v]copy[vout];[0:a]acopy[aout]"
        else:
            # Build video concat (no transitions - hard cuts)
            v_inputs = "".join(f"[{i}:v]" for i in range(len(clips)))
            filter_parts.append(f"{v_inputs}concat=n={len(clips)}:v=1:a=0[vout]")

            # Build audio crossfade chain
            # Start with first clip's audio
            current_audio = "[0:a]"

            for i in range(1, len(clips)):
                next_audio = f"[{i}:a]"

                if i < len(clips) - 1:
                    # Intermediate crossfade - output to temporary
                    out_label = f"[a{i}]"
                else:
                    # Final crossfade - output to aout
                    out_label = "[aout]"

                # Calculate duration for crossfade
                # Crossfade happens at the END of current clip
                prev_duration = clips[i-1].duration

                # acrossfade filter: duration is crossfade length, curve is crossfade shape
                filter_parts.append(
                    f"{current_audio}{next_audio}acrossfade=d={crossfade_duration}:c1=tri:c2=tri{out_label}"
                )

                current_audio = out_label

            filter_complex = ";".join(filter_parts)

        # Build full command
        cmd = ['ffmpeg', '-y']
        cmd.extend(input_args)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[vout]',
            '-map', '[aout]',
            '-c:v', 'copy',  # Don't re-encode video (already H.265)
            '-c:a', 'aac',
            '-b:a', '256k',
            '-movflags', '+faststart',
            str(master_path)
        ])

        try:
            self.logger.info(f"  Running FFmpeg with audio crossfades...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode != 0:
                self.logger.warning(f"Crossfade concat failed, trying simple concat...")
                self.logger.debug(f"Error: {result.stderr[-1000:]}")
                # Fall back to simple concat
                return self._create_master_simple(clips, master_path)

            if not master_path.exists():
                return self._create_master_simple(clips, master_path)

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
            self.logger.warning(f"Crossfade concat error: {e}, trying simple concat...")
            return self._create_master_simple(clips, master_path)

    def _create_master_batched(
        self,
        clips: List[ClipInfo],
        master_path: Path,
        crossfade_duration: float
    ) -> Optional[Path]:
        """
        Create master by processing clips in batches.

        For long timelines, process in batches of 20 clips,
        then concat the batches.
        """
        import tempfile
        batch_size = 20
        temp_files = []

        try:
            # Process in batches
            for i in range(0, len(clips), batch_size):
                batch = clips[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(clips) + batch_size - 1) // batch_size

                self.logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} clips)...")

                # Create temp file for batch
                temp_path = self.config.analysis_dir / f"batch_{batch_num:03d}.mov"
                temp_files.append(temp_path)

                # Process batch with crossfades
                batch_result = self._create_batch_with_crossfades(batch, temp_path, crossfade_duration)
                if not batch_result:
                    # Fall back to simple concat for this batch
                    self._create_master_simple(batch, temp_path)

            # Now concat all batches (simple concat is fine between batches)
            self.logger.info(f"  Concatenating {len(temp_files)} batches...")

            concat_list = self.config.analysis_dir / "batch_concat.txt"
            with open(concat_list, 'w') as f:
                for temp in temp_files:
                    f.write(f"file '{temp}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                '-movflags', '+faststart',
                str(master_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                self.logger.error(f"Batch concat failed: {result.stderr[-500:]}")
                return None

            # Verify output
            info = probe_file(master_path)
            if info:
                self.logger.info(f"  Master duration: {info.duration/60:.1f} minutes")
                self.logger.info(f"  Master size: {master_path.stat().st_size / (1024*1024*1024):.2f} GB")

            return master_path

        finally:
            # Cleanup temp files
            for temp in temp_files:
                if temp.exists():
                    temp.unlink()

    def _create_batch_with_crossfades(
        self,
        clips: List[ClipInfo],
        output_path: Path,
        crossfade_duration: float
    ) -> bool:
        """Create a single batch with audio crossfades."""
        if len(clips) == 1:
            # Just copy single clip
            cmd = [
                'ffmpeg', '-y',
                '-i', str(clips[0].path),
                '-c', 'copy',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0

        # Build filter_complex for batch
        input_args = []
        for clip in clips:
            input_args.extend(['-i', str(clip.path)])

        # Video concat
        v_inputs = "".join(f"[{i}:v]" for i in range(len(clips)))
        filter_parts = [f"{v_inputs}concat=n={len(clips)}:v=1:a=0[vout]"]

        # Audio crossfade chain
        current_audio = "[0:a]"
        for i in range(1, len(clips)):
            next_audio = f"[{i}:a]"
            out_label = "[aout]" if i == len(clips) - 1 else f"[a{i}]"
            filter_parts.append(
                f"{current_audio}{next_audio}acrossfade=d={crossfade_duration}:c1=tri:c2=tri{out_label}"
            )
            current_audio = out_label

        filter_complex = ";".join(filter_parts)

        cmd = ['ffmpeg', '-y']
        cmd.extend(input_args)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[vout]',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '256k',
            str(output_path)
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return result.returncode == 0

    def _create_master_simple(self, clips: List[ClipInfo], master_path: Path) -> Optional[Path]:
        """
        Simple concat fallback without crossfades.

        Used when filter_complex approach fails.
        """
        self.logger.info(f"  Using simple concat for {len(clips)} clips...")

        concat_list = self.config.analysis_dir / "concat_list.txt"
        with open(concat_list, 'w') as f:
            for clip in clips:
                escaped_path = str(clip.path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '256k',
            '-movflags', '+faststart',
            str(master_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            if result.returncode != 0:
                self.logger.error(f"Simple concat failed: {result.stderr[-500:]}")
                return None

            info = probe_file(master_path)
            if info:
                self.logger.info(f"  Master duration: {info.duration/60:.1f} minutes")

            return master_path

        except Exception as e:
            self.logger.error(f"Simple concat error: {e}")
            return None

    def _create_doc_edits(self, clips: List[ClipInfo]) -> Dict[str, Path]:
        """
        Create separate 'doc edit' files for each camera source.

        Doc edits are simple J/L cut concatenations of all clips from a single
        camera, with original camera audio. Provides a VHS-style continuous
        viewing experience per camera.

        Returns dict of camera_name -> output_path
        """
        results = {}

        # Separate clips by camera source using filename prefix
        # Transcoder names files: dad_cam_XXX.mov (main) vs tripod_cam_XXX.mov (tripod)
        main_clips = [c for c in clips if c.filename.startswith(self.config.output_prefix + "_")
                      and not c.filename.startswith("tripod_cam")]
        tripod_clips = [c for c in clips if c.filename.startswith("tripod_cam")]

        self.logger.info(f"  Found {len(main_clips)} main camera clips, {len(tripod_clips)} tripod camera clips")

        # Create main camera doc edit
        if main_clips:
            main_path = self.master_dir / f"{self.config.output_prefix}_main_camera_docedit.mov"
            self.logger.info(f"  Creating main camera doc edit ({len(main_clips)} clips)...")
            if self._create_doc_edit_for_camera(main_clips, main_path):
                results["main_camera"] = main_path
                info = probe_file(main_path)
                if info:
                    self.logger.info(f"    Duration: {info.duration/60:.1f} minutes")
                    self.logger.info(f"    Size: {main_path.stat().st_size / (1024*1024*1024):.2f} GB")

        # Create tripod camera doc edit
        if tripod_clips:
            tripod_path = self.master_dir / f"{self.config.output_prefix}_tripod_camera_docedit.mov"
            self.logger.info(f"  Creating tripod camera doc edit ({len(tripod_clips)} clips)...")
            if self._create_doc_edit_for_camera(tripod_clips, tripod_path):
                results["tripod_camera"] = tripod_path
                info = probe_file(tripod_path)
                if info:
                    self.logger.info(f"    Duration: {info.duration/60:.1f} minutes")
                    self.logger.info(f"    Size: {tripod_path.stat().st_size / (1024*1024*1024):.2f} GB")

        return results

    def _create_doc_edit_for_camera(self, clips: List[ClipInfo], output_path: Path) -> bool:
        """
        Create a single doc edit file from clips with J/L audio crossfades.

        Strategy:
        1. Concat demuxer for video (stream copy - fast, no re-encode)
        2. Extract and crossfade audio separately
        3. Mux video + crossfaded audio together
        """
        if not clips:
            return False

        if len(clips) == 1:
            # Single clip - just copy
            import shutil
            shutil.copy2(clips[0].path, output_path)
            return True

        # Dynamic crossfade based on clip duration
        # Short clips (<30s): 0.5s, Medium (30s+): 1s, Long (60s+): 2s
        def get_crossfade(duration: float) -> float:
            if duration < 30:
                return 0.5
            elif duration < 60:
                return 1.0
            else:
                return 2.0

        # Step 1: Concat video only (stream copy)
        concat_list = self.config.analysis_dir / "docedit_concat.txt"
        video_only = self.config.analysis_dir / "docedit_video.mov"

        with open(concat_list, 'w') as f:
            for clip in clips:
                escaped = str(clip.path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd_video = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list),
            '-c:v', 'copy',
            '-an',  # No audio
            str(video_only)
        ]

        try:
            result = subprocess.run(cmd_video, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                self.logger.error(f"Video concat failed: {result.stderr[-500:]}")
                concat_list.unlink(missing_ok=True)
                return False

            # Step 2: Build audio crossfade filter
            # For N clips, chain N-1 acrossfade filters
            input_args = []
            for clip in clips:
                input_args.extend(['-i', str(clip.path)])

            # Build audio crossfade chain with dynamic durations
            if len(clips) == 2:
                # Use shorter clip's duration for crossfade
                fade = min(get_crossfade(clips[0].duration), get_crossfade(clips[1].duration))
                filter_complex = f"[0:a][1:a]acrossfade=d={fade}:c1=esin:c2=esin[aout]"
            else:
                filter_parts = []
                current = "[0:a]"
                for i in range(1, len(clips)):
                    next_audio = f"[{i}:a]"
                    if i == len(clips) - 1:
                        out_label = "[aout]"
                    else:
                        out_label = f"[a{i}]"
                    # Use shorter of two clips for crossfade duration
                    fade = min(get_crossfade(clips[i-1].duration), get_crossfade(clips[i].duration))
                    filter_parts.append(f"{current}{next_audio}acrossfade=d={fade}:c1=esin:c2=esin{out_label}")
                    current = out_label
                filter_complex = ";".join(filter_parts)

            # Step 3: Create crossfaded audio
            audio_only = self.config.analysis_dir / "docedit_audio.m4a"
            cmd_audio = ['ffmpeg', '-y']
            cmd_audio.extend(input_args)
            cmd_audio.extend([
                '-filter_complex', filter_complex,
                '-map', '[aout]',
                '-c:a', 'aac',
                '-b:a', '256k',
                str(audio_only)
            ])

            result = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                self.logger.error(f"Audio crossfade failed: {result.stderr[-500:]}")
                # Fallback: just concat without crossfades
                video_only.unlink(missing_ok=True)
                return self._create_doc_edit_simple(clips, output_path, concat_list)

            # Step 4: Mux video + audio, then normalize audio
            # First mux to temp file
            temp_muxed = self.config.analysis_dir / "docedit_temp_muxed.mov"
            cmd_mux = [
                'ffmpeg', '-y',
                '-i', str(video_only),
                '-i', str(audio_only),
                '-c:v', 'copy',
                '-c:a', 'copy',
                str(temp_muxed)
            ]

            result = subprocess.run(cmd_mux, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                self.logger.error(f"Mux failed: {result.stderr[-500:]}")
                video_only.unlink(missing_ok=True)
                audio_only.unlink(missing_ok=True)
                concat_list.unlink(missing_ok=True)
                return False

            # Step 5: Normalize audio loudness across entire doc edit
            # Use EBU R128 loudness normalization (-14 LUFS is standard for streaming)
            self.logger.info("    Normalizing audio loudness...")
            cmd_normalize = [
                'ffmpeg', '-y',
                '-i', str(temp_muxed),
                '-c:v', 'copy',
                '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
                '-c:a', 'aac',
                '-b:a', '256k',
                '-movflags', '+faststart',
                str(output_path)
            ]

            result = subprocess.run(cmd_normalize, capture_output=True, text=True, timeout=3600)

            # Cleanup temp files
            video_only.unlink(missing_ok=True)
            audio_only.unlink(missing_ok=True)
            concat_list.unlink(missing_ok=True)
            temp_muxed.unlink(missing_ok=True)

            if result.returncode != 0:
                self.logger.error(f"Normalize failed: {result.stderr[-500:]}")
                return False

            return output_path.exists()

        except Exception as e:
            self.logger.error(f"Doc edit failed: {e}")
            concat_list.unlink(missing_ok=True)
            video_only.unlink(missing_ok=True) if video_only.exists() else None
            return False

    def _create_doc_edit_simple(self, clips: List[ClipInfo], output_path: Path, concat_list: Path) -> bool:
        """Fallback: simple concat without crossfades."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            '-movflags', '+faststart',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        concat_list.unlink(missing_ok=True)
        return result.returncode == 0 and output_path.exists()

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

        # Add XML declaration - use text mode for compatibility
        with open(fcpxml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE fcpxml>\n')
            tree.write(f, encoding='unicode')

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

        with open(fcpxml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE fcpxml>\n')
            tree.write(f, encoding='unicode')

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
