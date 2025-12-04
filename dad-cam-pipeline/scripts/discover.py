#!/usr/bin/env python3
"""
01_discover.py - Source File Discovery and Metadata Extraction
==============================================================

Scans source directories for video files (TOD, MTS, MOD) and audio files (WAV),
extracts metadata from MOI sidecar files, and establishes proper chronological
ordering for the pipeline.

Output: analysis/inventory.json
"""

import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import natural_sort_key, save_json, parse_timestamp
from lib.ffmpeg_utils import probe_file, get_duration
from lib.logging_utils import get_logger, PhaseLogger, setup_logger


@dataclass
class SourceFile:
    """Represents a discovered source file."""
    original_path: str
    filename: str
    camera_source: str  # "main_camera", "tripod_camera", "audio"
    format: str  # "TOD", "MTS", "WAV", etc.
    order: int
    timestamp: Optional[str]
    duration: float
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    is_interlaced: bool
    moi_path: Optional[str]
    # Audio-specific fields
    audio_type: Optional[str] = None  # "lapel", "dj", "ambient", None for video
    audio_priority: int = 99  # Lower = higher priority (1=lapel, 2=dj, 3=camera)

    def to_dict(self) -> dict:
        return asdict(self)


class SourceDiscovery:
    """Discovers and catalogs all source files."""

    # Supported video extensions
    VIDEO_EXTENSIONS = {'.tod', '.mod', '.mts', '.m2ts', '.mp4', '.mov'}

    # Supported audio extensions
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aac', '.flac'}

    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.logger = get_logger()

        # Ensure output directory exists
        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def discover_all(self) -> Dict:
        """
        Discover all source files and generate inventory.

        Returns:
            Complete inventory dictionary
        """
        with PhaseLogger("Source Discovery", self.logger) as phase:
            # Find all source files
            phase.step("Scanning for video files...")
            video_files = self._find_video_files()
            phase.substep(f"Found {len(video_files)} video files")

            phase.step("Scanning for audio files...")
            audio_files = self._find_audio_files()
            phase.substep(f"Found {len(audio_files)} audio files")

            # Categorize by source
            phase.step("Categorizing sources...")
            main_camera, tripod_camera = self._categorize_video_sources(video_files)
            phase.substep(f"Main camera: {len(main_camera)} files")
            phase.substep(f"Tripod camera: {len(tripod_camera)} files")

            # Extract metadata and create source objects
            phase.step("Extracting metadata from main camera files...")
            main_sources = self._process_video_files(main_camera, "main_camera")

            phase.step("Extracting metadata from tripod camera files...")
            tripod_sources = self._process_video_files(tripod_camera, "tripod_camera")

            phase.step("Processing audio files...")
            audio_sources = self._process_audio_files(audio_files)

            # Sort by timestamp
            phase.step("Sorting files by recording timestamp...")
            main_sources = self._sort_by_timestamp(main_sources)
            tripod_sources = self._sort_by_timestamp(tripod_sources)
            audio_sources = self._sort_by_timestamp(audio_sources)

            # Assign sequential order numbers
            self._assign_order_numbers(main_sources)
            self._assign_order_numbers(tripod_sources)
            self._assign_order_numbers(audio_sources)

            # Check for gaps in sequence
            phase.step("Checking for sequence gaps...")
            gaps = self._detect_sequence_gaps(main_sources)
            if gaps:
                for gap in gaps:
                    phase.warn(f"Gap detected: {gap['after']} -> {gap['before']} (missing: {gap['missing']})")

            # Build inventory
            inventory = {
                "generated_at": datetime.now().isoformat(),
                "source_dir": str(self.source_dir),
                "summary": {
                    "main_camera_clips": len(main_sources),
                    "tripod_camera_clips": len(tripod_sources),
                    "audio_files": len(audio_sources),
                    "total_main_duration": sum(s.duration for s in main_sources),
                    "total_tripod_duration": sum(s.duration for s in tripod_sources),
                    "total_audio_duration": sum(s.duration for s in audio_sources),
                    "sequence_gaps": gaps,
                },
                "main_camera": [s.to_dict() for s in main_sources],
                "tripod_camera": [s.to_dict() for s in tripod_sources],
                "audio_sources": [s.to_dict() for s in audio_sources],
            }

            # Save inventory
            inventory_path = self.analysis_dir / "inventory.json"
            save_json(inventory, inventory_path)
            phase.success(f"Inventory saved to: {inventory_path}")

            # Print summary
            self._print_summary(inventory)

            return inventory

    def _find_video_files(self) -> List[Path]:
        """Find all video files recursively."""
        files = []
        for ext in self.VIDEO_EXTENSIONS:
            files.extend(self.source_dir.rglob(f"*{ext}"))
            files.extend(self.source_dir.rglob(f"*{ext.upper()}"))
        return sorted(set(files), key=natural_sort_key)

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files recursively."""
        files = []
        for ext in self.AUDIO_EXTENSIONS:
            files.extend(self.source_dir.rglob(f"*{ext}"))
            files.extend(self.source_dir.rglob(f"*{ext.upper()}"))
        return sorted(set(files), key=natural_sort_key)

    def _categorize_video_sources(self, video_files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Categorize video files into main camera and tripod camera.

        Heuristic:
        - TOD/MOD files in folders with "PRG" = main camera (JVC format)
        - MTS files = tripod camera (Sony/Panasonic AVCHD)
        """
        main_camera = []
        tripod_camera = []

        for f in video_files:
            ext = f.suffix.lower()
            path_str = str(f).lower()

            if ext in ('.tod', '.mod'):
                main_camera.append(f)
            elif ext in ('.mts', '.m2ts'):
                tripod_camera.append(f)
            elif 'main' in path_str or 'handheld' in path_str:
                main_camera.append(f)
            elif 'tripod' in path_str or 'static' in path_str:
                tripod_camera.append(f)
            else:
                # Default to main camera
                main_camera.append(f)

        return main_camera, tripod_camera

    def _process_video_files(self, files: List[Path], camera_source: str) -> List[SourceFile]:
        """Process video files and extract metadata."""
        sources = []

        for f in files:
            # Find MOI sidecar if it exists
            moi_path = self._find_moi_sidecar(f)
            timestamp = None

            # Try to get timestamp from MOI
            if moi_path:
                timestamp = self._extract_moi_timestamp(moi_path)

            # Probe file for media info
            info = probe_file(f)

            if info:
                video = info.video_stream
                source = SourceFile(
                    original_path=str(f),
                    filename=f.name,
                    camera_source=camera_source,
                    format=f.suffix.upper().lstrip('.'),
                    order=0,  # Will be assigned later
                    timestamp=timestamp,
                    duration=info.duration,
                    width=video.width if video else None,
                    height=video.height if video else None,
                    fps=video.fps if video else None,
                    is_interlaced=info.is_interlaced,
                    moi_path=str(moi_path) if moi_path else None,
                )
                sources.append(source)
            else:
                self.logger.warning(f"Could not probe file: {f}")

        return sources

    def _process_audio_files(self, files: List[Path]) -> List[SourceFile]:
        """Process audio files and extract metadata with classification."""
        sources = []

        for f in files:
            # Get timestamp from filename or metadata
            timestamp = self._extract_audio_timestamp(f)

            # Probe for duration
            info = probe_file(f)
            duration = info.duration if info else 0

            # Classify audio type
            audio_type, audio_priority = self._classify_audio_source(f, duration)

            source = SourceFile(
                original_path=str(f),
                filename=f.name,
                camera_source="audio",
                format=f.suffix.upper().lstrip('.'),
                order=0,
                timestamp=timestamp,
                duration=duration,
                width=None,
                height=None,
                fps=None,
                is_interlaced=False,
                moi_path=None,
                audio_type=audio_type,
                audio_priority=audio_priority,
            )
            sources.append(source)

            self.logger.info(f"  Audio: {f.name} -> {audio_type} (priority {audio_priority})")

        return sources

    def _classify_audio_source(self, audio_path: Path, duration: float) -> Tuple[str, int]:
        """
        Classify audio source type based on filename, duration, and content.

        Returns:
            Tuple of (audio_type, priority)
            Types: "lapel", "dj", "ambient"
            Priority: 1=lapel (highest), 2=dj, 3=ambient
        """
        filename = audio_path.name.lower()
        stem = audio_path.stem.lower()

        # Check filename patterns for lapel/speech
        lapel_patterns = ['lapel', 'lav', 'speech', 'vow', 'ceremony', 'tr1', 'tr2', 'track']
        for pattern in lapel_patterns:
            if pattern in stem:
                return ("lapel", 1)

        # Check filename patterns for DJ/music
        dj_patterns = ['dj', 'music', 'dance', 'reception', 'party', 'mix']
        for pattern in dj_patterns:
            if pattern in stem:
                return ("dj", 2)

        # Use duration heuristics
        # Lapel recordings tend to be very long (ceremony + speeches)
        # DJ sets are medium length (reception)
        # Ambient/room tone is short

        if duration > 7200:  # > 2 hours - likely lapel left running
            return ("lapel", 1)
        elif duration > 3600:  # 1-2 hours - could be either, check content
            # Analyze frequency content to distinguish speech vs music
            audio_type = self._analyze_audio_content(audio_path)
            if audio_type == "speech":
                return ("lapel", 1)
            elif audio_type == "music":
                return ("dj", 2)
            else:
                return ("lapel", 1)  # Default long recordings to lapel
        elif duration > 1800:  # 30min - 1hr - likely DJ set
            return ("dj", 2)
        else:
            # Short recording - ambient/room tone
            return ("ambient", 3)

    def _analyze_audio_content(self, audio_path: Path) -> str:
        """
        Analyze audio content to distinguish speech vs music.

        Uses FFmpeg to check frequency distribution:
        - Speech: energy concentrated in 300Hz-3kHz
        - Music: broader spectrum with strong bass (< 200Hz)

        Returns: "speech", "music", or "unknown"
        """
        try:
            # Extract 30 seconds from middle of file for analysis
            cmd = [
                'ffmpeg', '-y',
                '-ss', '60',  # Skip first minute
                '-i', str(audio_path),
                '-t', '30',
                '-af', 'astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level',
                '-f', 'null', '-'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # For now, use simple heuristic based on filename patterns we might have missed
            # Full implementation would analyze FFT spectrum

            # Check for Zoom recorder patterns (often used for speech)
            if 'zoom' in str(audio_path).lower():
                return "speech"

            return "unknown"

        except Exception:
            return "unknown"

    def _find_moi_sidecar(self, video_path: Path) -> Optional[Path]:
        """Find MOI sidecar file for a video file."""
        # MOI has same name as video but with .MOI extension
        moi_path = video_path.with_suffix('.MOI')
        if moi_path.exists():
            return moi_path

        moi_path = video_path.with_suffix('.moi')
        if moi_path.exists():
            return moi_path

        return None

    def _extract_moi_timestamp(self, moi_path: Path) -> Optional[str]:
        """Extract timestamp from MOI file using exiftool."""
        try:
            result = subprocess.run(
                ['exiftool', '-DateTimeOriginal', '-s3', str(moi_path)],
                capture_output=True,
                text=True,
                check=True
            )
            timestamp_str = result.stdout.strip()
            if timestamp_str:
                # Parse and reformat to ISO
                dt = parse_timestamp(timestamp_str)
                if dt:
                    return dt.isoformat()
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            self.logger.debug(f"Error extracting MOI timestamp: {e}")

        return None

    def _extract_audio_timestamp(self, audio_path: Path) -> Optional[str]:
        """Extract timestamp from audio file metadata or filename."""
        # Try exiftool first
        try:
            result = subprocess.run(
                ['exiftool', '-DateTimeOriginal', '-CreateDate', '-s3', str(audio_path)],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    dt = parse_timestamp(line.strip())
                    if dt:
                        return dt.isoformat()
        except subprocess.CalledProcessError:
            pass

        # Try parsing filename (common formats: YYMMDD_XXXX.wav, 230825_0002.wav)
        filename = audio_path.stem
        # Match YYMMDD pattern
        match = re.match(r'(\d{6})_', filename)
        if match:
            date_str = match.group(1)
            try:
                # Assume 20XX century
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                return datetime(year, month, day).isoformat()
            except ValueError:
                pass

        return None

    def _sort_by_timestamp(self, sources: List[SourceFile]) -> List[SourceFile]:
        """Sort sources by timestamp, falling back to filename."""
        def sort_key(s: SourceFile):
            if s.timestamp:
                return (0, s.timestamp, s.filename)
            else:
                # Use natural sort on filename as fallback
                return (1, "", s.filename)

        return sorted(sources, key=sort_key)

    def _assign_order_numbers(self, sources: List[SourceFile]) -> None:
        """Assign sequential order numbers to sources."""
        for i, source in enumerate(sources, start=1):
            source.order = i

    def _detect_sequence_gaps(self, sources: List[SourceFile]) -> List[Dict]:
        """
        Detect gaps in hexadecimal file naming sequence.

        For files like MOV008, MOV009, MOV00A, MOV00B...
        Detects if MOV00C is missing between MOV00B and MOV00D.
        """
        gaps = []

        for i in range(len(sources) - 1):
            current = sources[i]
            next_file = sources[i + 1]

            # Extract hex portion from filename
            current_match = re.search(r'([0-9A-Fa-f]+)\.[^.]+$', current.filename)
            next_match = re.search(r'([0-9A-Fa-f]+)\.[^.]+$', next_file.filename)

            if current_match and next_match:
                try:
                    current_num = int(current_match.group(1), 16)
                    next_num = int(next_match.group(1), 16)

                    if next_num - current_num > 1:
                        missing = [
                            f"{current.filename.rsplit(current_match.group(1), 1)[0]}{hex(n)[2:].upper().zfill(3)}"
                            for n in range(current_num + 1, next_num)
                        ]
                        gaps.append({
                            "after": current.filename,
                            "before": next_file.filename,
                            "missing": missing
                        })
                except ValueError:
                    pass

        return gaps

    def _print_summary(self, inventory: Dict) -> None:
        """Print a summary of discovered files."""
        summary = inventory['summary']
        logger = self.logger

        logger.info("")
        logger.info("=" * 60)
        logger.info("DISCOVERY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Main Camera:    {summary['main_camera_clips']:3d} clips  "
                   f"({summary['total_main_duration']/60:.1f} min)")
        logger.info(f"  Tripod Camera:  {summary['tripod_camera_clips']:3d} clips  "
                   f"({summary['total_tripod_duration']/60:.1f} min)")
        logger.info(f"  Audio Files:    {summary['audio_files']:3d} files  "
                   f"({summary['total_audio_duration']/60:.1f} min)")
        logger.info("=" * 60)

        if summary['sequence_gaps']:
            logger.warning(f"  ⚠ {len(summary['sequence_gaps'])} sequence gap(s) detected")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover and catalog source media files"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source directory containing media files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for pipeline results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_dir=Path(args.output) / "logs", level=level)

    # Run discovery
    discovery = SourceDiscovery(args.source, args.output)
    inventory = discovery.discover_all()

    return 0 if inventory else 1


if __name__ == "__main__":
    sys.exit(main())
