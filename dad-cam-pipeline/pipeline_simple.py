#!/usr/bin/env python3
"""
pipeline_simple.py - Simplified Dad Cam Pipeline
=================================================

A streamlined 4-phase pipeline for processing legacy camcorder footage:

Phase 1: DISCOVER   - Find files, parse MOI timestamps, sort chronologically
Phase 2: TRANSCODE  - TOD/MTS → H.265 MOV, deinterlace, fix pixels, trim black
Phase 3: NORMALIZE  - Two-pass loudnorm to -14 LUFS, remove 60Hz hum
Phase 4: ASSEMBLE   - Concat with 0.5s audio fades → timeline edit

Outputs:
- clips/dad_cam_001.mov ... dad_cam_NNN.mov   (individual clips)
- clips/tripod_cam_001.mov ... NNN.mov        (tripod clips if present)
- timeline/dad_cam_main_timeline.mov          (continuous main camera edit)
- timeline/dad_cam_tripod_timeline.mov        (continuous tripod edit)

Usage:
    python pipeline_simple.py --source "/path/to/footage" --output "/path/to/output"

    # Skip already-completed phases:
    python pipeline_simple.py --source ... --output ... --skip-transcode --skip-audio
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib.utils import load_json, save_json
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig

# Import phase scripts
from scripts.discover import SourceDiscovery
from scripts.transcode import BatchTranscoder
from scripts.audio_process import AudioProcessor
from scripts.assemble_simple import SimpleAssembler, verify_timeline


class SimplePipeline:
    """
    Simplified 4-phase pipeline.

    Designed for reliability and simplicity over features.
    """

    def __init__(self, source_dir: str, output_dir: str):
        self.config = PipelineConfig()
        self.config.setup_paths(source_dir, output_dir)

        # Ensure timeline directory exists
        self.timeline_dir = self.config.output_dir / "timeline"
        self.timeline_dir.mkdir(parents=True, exist_ok=True)

        self.logger = None  # Set up after logging is configured
        self.start_time = None

    def run(
        self,
        skip_transcode: bool = False,
        skip_audio: bool = False,
        skip_assembly: bool = False
    ) -> dict:
        """
        Run the complete pipeline.

        Args:
            skip_transcode: Skip Phase 2 (transcoding)
            skip_audio: Skip Phase 3 (audio processing)
            skip_assembly: Skip Phase 4 (timeline assembly)

        Returns:
            Results dictionary with status of each phase
        """
        self.start_time = time.time()
        self.logger = get_logger()

        results = {
            "started_at": datetime.now().isoformat(),
            "source_dir": str(self.config.source_dir),
            "output_dir": str(self.config.output_dir),
            "phases": {},
            "clips": {},
            "timelines": {},
            "success": False,
            "total_time": 0
        }

        self._print_header()

        try:
            # Phase 1: Discovery
            results["phases"]["discover"] = self._run_discovery()

            # Phase 2: Transcode
            if not skip_transcode:
                results["phases"]["transcode"] = self._run_transcode()
            else:
                self.logger.info("Skipping Phase 2: Transcode (--skip-transcode)")

            # Phase 3: Audio Processing
            if not skip_audio:
                results["phases"]["audio"] = self._run_audio_processing()
            else:
                self.logger.info("Skipping Phase 3: Audio Processing (--skip-audio)")

            # Phase 4: Assembly
            if not skip_assembly:
                results["phases"]["assembly"] = self._run_assembly()
            else:
                self.logger.info("Skipping Phase 4: Assembly (--skip-assembly)")

            # Gather final stats
            results["clips"] = self._count_clips()
            results["timelines"] = self._verify_timelines()
            results["success"] = self._check_success(results)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results["error"] = str(e)

        # Record total time
        results["total_time"] = time.time() - self.start_time
        results["completed_at"] = datetime.now().isoformat()

        # Save results
        save_json(results, self.config.analysis_dir / "pipeline_results.json")

        # Print summary
        self._print_summary(results)

        return results

    def _print_header(self):
        """Print pipeline header."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  DAD CAM PIPELINE (Simplified)")
        self.logger.info("  Legacy Footage Processing")
        self.logger.info("=" * 60)
        self.logger.info(f"  Source: {self.config.source_dir}")
        self.logger.info(f"  Output: {self.config.output_dir}")
        self.logger.info("=" * 60)
        self.logger.info("")

    def _run_discovery(self) -> dict:
        """Phase 1: Discover source files."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: DISCOVERY")
        self.logger.info("=" * 60)

        start = time.time()

        discovery = SourceDiscovery(
            self.config.source_dir,
            self.config.output_dir
        )
        inventory = discovery.discover_all()

        elapsed = time.time() - start

        return {
            "success": bool(inventory),
            "main_clips": len(inventory.get("main_camera", [])),
            "tripod_clips": len(inventory.get("tripod_camera", [])),
            "time": elapsed
        }

    def _run_transcode(self) -> dict:
        """Phase 2: Transcode source files."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: TRANSCODE")
        self.logger.info("=" * 60)

        start = time.time()

        # Load inventory
        inventory_path = self.config.analysis_dir / "inventory.json"
        if not inventory_path.exists():
            self.logger.error("Inventory not found - run discovery first")
            return {"success": False, "error": "No inventory"}

        inventory = load_json(inventory_path)

        # Run transcoder
        transcoder = BatchTranscoder(self.config)

        with PhaseLogger("Transcoding", self.logger):
            results = transcoder.transcode_all(inventory)

        elapsed = time.time() - start

        return {
            "success": results.get("transcoded", 0) > 0,
            "transcoded": results.get("transcoded", 0),
            "skipped": results.get("skipped", 0),
            "failed": results.get("failed", 0),
            "time": elapsed
        }

    def _run_audio_processing(self) -> dict:
        """Phase 3: Process audio (normalize, remove hum)."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: AUDIO PROCESSING")
        self.logger.info("=" * 60)

        start = time.time()

        processor = AudioProcessor(self.config)

        with PhaseLogger("Audio Processing", self.logger):
            results = processor.process_all()

        elapsed = time.time() - start

        summary = results.get("summary", {})

        return {
            "success": summary.get("total_clips", 0) > 0,
            "clips_processed": summary.get("total_clips", 0),
            "hum_removed": summary.get("hum_detected", 0),
            "avg_lufs": summary.get("average_lufs", -14),
            "time": elapsed
        }

    def _run_assembly(self) -> dict:
        """Phase 4: Assemble timeline edits."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: TIMELINE ASSEMBLY")
        self.logger.info("=" * 60)

        start = time.time()

        assembler = SimpleAssembler(self.config)

        with PhaseLogger("Timeline Assembly", self.logger):
            results = assembler.assemble()

        elapsed = time.time() - start

        return {
            "success": bool(results.get("main_timeline") or results.get("tripod_timeline")),
            "main_timeline": results.get("main_timeline"),
            "tripod_timeline": results.get("tripod_timeline"),
            "durations": results.get("durations", {}),
            "errors": results.get("errors", []),
            "time": elapsed
        }

    def _count_clips(self) -> dict:
        """Count output clips."""
        clips_dir = self.config.clips_dir

        main_count = len(list(clips_dir.glob("dad_cam_*.mov")))
        tripod_count = len(list(clips_dir.glob("tripod_cam_*.mov")))

        return {
            "main": main_count,
            "tripod": tripod_count,
            "total": main_count + tripod_count
        }

    def _verify_timelines(self) -> dict:
        """Verify timeline outputs."""
        results = {}

        main_path = self.timeline_dir / "dad_cam_main_timeline.mov"
        if main_path.exists():
            results["main"] = verify_timeline(main_path)

        tripod_path = self.timeline_dir / "dad_cam_tripod_timeline.mov"
        if tripod_path.exists():
            results["tripod"] = verify_timeline(tripod_path)

        return results

    def _check_success(self, results: dict) -> bool:
        """Determine if pipeline succeeded."""
        # Must have clips
        if results["clips"]["total"] == 0:
            return False

        # Must have at least one timeline
        timelines = results.get("timelines", {})
        if not timelines:
            return False

        # Check for audio/video sync in timelines
        for name, verification in timelines.items():
            if not verification.get("duration_match", False):
                self.logger.warning(f"{name} timeline has audio/video mismatch")
                # Don't fail on this, just warn

        return True

    def _print_summary(self, results: dict):
        """Print final summary."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info("")

        # Clips
        clips = results.get("clips", {})
        self.logger.info(f"CLIPS:")
        self.logger.info(f"  Main camera:   {clips.get('main', 0)} clips")
        self.logger.info(f"  Tripod camera: {clips.get('tripod', 0)} clips")
        self.logger.info("")

        # Timelines
        timelines = results.get("timelines", {})
        self.logger.info(f"TIMELINES:")

        for name, verification in timelines.items():
            if verification.get("exists"):
                dur_min = verification.get("video_duration", 0) / 60
                size_gb = verification.get("size_gb", 0)
                gap = verification.get("gap_seconds", 0)
                status = "OK" if verification.get("duration_match") else f"GAP: {gap:.1f}s"

                self.logger.info(f"  {name.title()}: {dur_min:.1f} min, {size_gb:.2f} GB [{status}]")

        self.logger.info("")

        # Timing
        total_time = results.get("total_time", 0)
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")

        # Status
        if results.get("success"):
            self.logger.info("")
            self.logger.info("STATUS: SUCCESS")
        else:
            self.logger.info("")
            self.logger.info("STATUS: COMPLETED WITH ISSUES")
            if results.get("error"):
                self.logger.info(f"Error: {results['error']}")

        self.logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dad Cam Pipeline - Legacy Footage Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline run:
  python pipeline_simple.py --source "/path/to/footage" --output "/path/to/output"

  # Skip transcoding (use existing clips):
  python pipeline_simple.py -s /footage -o /output --skip-transcode

  # Only run discovery and transcode:
  python pipeline_simple.py -s /footage -o /output --skip-audio --skip-assembly
        """
    )

    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source directory containing TOD/MTS files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for clips and timelines"
    )
    parser.add_argument(
        "--skip-transcode",
        action="store_true",
        help="Skip Phase 2 (transcoding) - use existing clips"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip Phase 3 (audio processing)"
    )
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="Skip Phase 4 (timeline assembly)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_dir=output_dir / "logs", level=level)

    logger = get_logger()
    logger.info(f"Logging to: {output_dir / 'logs'}")

    # Run pipeline
    pipeline = SimplePipeline(args.source, args.output)
    results = pipeline.run(
        skip_transcode=args.skip_transcode,
        skip_audio=args.skip_audio,
        skip_assembly=args.skip_assembly
    )

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
