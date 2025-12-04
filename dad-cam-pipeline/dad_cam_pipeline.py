#!/usr/bin/env python3
"""
Dad Cam Pipeline - Master Orchestrator
=======================================

Automated video processing pipeline for legacy camcorder footage.

Phases:
1. Discovery - Scan sources, extract metadata, establish order
2. Analysis - Detect stuck pixels, black frames, stability
3. Transcode - Convert to H.265 with fixes applied
4. Audio - Hum removal and loudness normalization
5. Sync - Multicam synchronization (optional)
6. Assembly - Concatenate, generate FCPXML

Usage:
    python dad_cam_pipeline.py --source "/path/to/footage" --output "/path/to/output"

Options:
    --source, -s      Source directory with raw footage
    --output, -o      Output directory for processed files
    --parallel, -j    Number of parallel transcode jobs
    --skip-transcode  Skip transcoding if clips already exist
    --skip-multicam   Skip multicam synchronization
    --dry-run         Analyze only, don't process
    --verbose, -v     Enable verbose logging
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.utils import load_json, save_json, format_duration
from lib.logging_utils import setup_logger, get_logger, PhaseLogger
from config.settings import PipelineConfig

# Import phase scripts
from scripts.discover import SourceDiscovery
from scripts.analyze import VideoAnalyzer
from scripts.detect_shots import ShotDetector
from scripts.transcode import BatchTranscoder
from scripts.audio_process import AudioProcessor
from scripts.sync_multicam import MulticamSynchronizer
from scripts.audio_mix import AudioMixAnalyzer
from scripts.multicam_edit import MulticamEditor
from scripts.assemble import TimelineAssembler


class DadCamPipeline:
    """Master pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.start_time = None
        self.results = {}

    def run(self) -> int:
        """
        Execute the complete pipeline.

        Returns:
            Exit code (0 = success)
        """
        self.start_time = datetime.now()

        self._print_banner()

        try:
            # Phase 1-3: Discovery, Analysis, Shot Detection
            if not self.config.skip_analysis:
                # Phase 1: Discovery
                if not self._run_discovery():
                    return 1

                # Phase 2: Analysis (stuck pixels, black frames, stability)
                if not self._run_analysis():
                    return 1

                # Phase 3: Shot Detection (TransNet V2)
                if not self._run_shot_detection():
                    self.logger.warning("Shot detection failed - continuing with basic switching")
            else:
                self.logger.info("Skipping phases 1-3 (--skip-analysis) - using existing analysis data")

            # Check for dry run
            if self.config.dry_run:
                self.logger.info("")
                self.logger.info("Dry run complete - no files processed")
                return 0

            # Phase 4: Transcoding
            if not self.config.skip_transcode:
                if not self._run_transcode():
                    return 1
            else:
                self.logger.info("Skipping transcode (--skip-transcode)")

            # Phase 5: Audio Processing
            if not self.config.skip_audio:
                if not self._run_audio_process():
                    return 1
            else:
                self.logger.info("Skipping audio processing (--skip-audio)")

            # Phase 6: Multicam Sync
            if not self.config.skip_multicam:
                self._run_multicam_sync()

            # Phase 7: Audio Mix Decisions
            if not self._run_audio_mix():
                self.logger.warning("Audio mix analysis failed - using camera audio only")

            # Phase 8: Multicam Edit Decisions
            if not self._run_multicam_edit():
                self.logger.warning("Multicam edit failed - using single camera")

            # Phase 9: Assembly
            if not self._run_assembly():
                return 1

            # Final summary
            self._print_summary()

            return 0

        except KeyboardInterrupt:
            self.logger.warning("\nPipeline interrupted by user")
            return 130
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return 1

    def _print_banner(self):
        """Print pipeline banner."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  DAD CAM PIPELINE")
        self.logger.info("  Legacy Footage Processing & Assembly")
        self.logger.info("=" * 60)
        self.logger.info(f"  Source: {self.config.source_dir}")
        self.logger.info(f"  Output: {self.config.output_dir}")
        self.logger.info(f"  Parallel jobs: {self.config.parallel_jobs}")
        self.logger.info("=" * 60)
        self.logger.info("")

    def _run_discovery(self) -> bool:
        """Run discovery phase."""
        with PhaseLogger("PHASE 1: Discovery", self.logger) as phase:
            try:
                discovery = SourceDiscovery(
                    self.config.source_dir,
                    self.config.output_dir
                )
                inventory = discovery.discover_all()

                if not inventory:
                    phase.error("No source files found")
                    return False

                self.results["inventory"] = inventory
                return True

            except Exception as e:
                phase.error(f"Discovery failed: {e}")
                return False

    def _run_analysis(self) -> bool:
        """Run analysis phase."""
        with PhaseLogger("PHASE 2: Analysis", self.logger) as phase:
            try:
                # Load inventory
                inventory_path = self.config.analysis_dir / "inventory.json"
                inventory = load_json(inventory_path)

                analyzer = VideoAnalyzer(self.config)
                analysis = analyzer.analyze_all(inventory)

                self.results["analysis"] = analysis
                return True

            except Exception as e:
                phase.error(f"Analysis failed: {e}")
                return False

    def _run_shot_detection(self) -> bool:
        """Run shot boundary detection phase."""
        with PhaseLogger("PHASE 3: Shot Detection", self.logger) as phase:
            try:
                inventory = load_json(self.config.analysis_dir / "inventory.json")

                detector = ShotDetector(self.config)
                results = detector.detect_all(inventory)

                self.results["shots"] = results
                return True

            except Exception as e:
                phase.error(f"Shot detection failed: {e}")
                return False

    def _run_transcode(self) -> bool:
        """Run transcode phase."""
        with PhaseLogger("PHASE 4: Transcoding", self.logger) as phase:
            try:
                # Load required data
                inventory = load_json(self.config.analysis_dir / "inventory.json")

                stuck_pixels_path = self.config.analysis_dir / "stuck_pixels.json"
                stuck_pixels = load_json(stuck_pixels_path) if stuck_pixels_path.exists() else {}

                black_frames_path = self.config.analysis_dir / "black_frames.json"
                black_frames = load_json(black_frames_path) if black_frames_path.exists() else {}

                vignette_path = self.config.analysis_dir / "vignette.json"
                vignette = load_json(vignette_path) if vignette_path.exists() else {}

                transcoder = BatchTranscoder(self.config)
                results = transcoder.transcode_all(inventory, stuck_pixels, black_frames, vignette)

                self.results["transcode"] = results

                if transcoder.failed > 0:
                    phase.warn(f"{transcoder.failed} clips failed to transcode")

                return True

            except Exception as e:
                phase.error(f"Transcode failed: {e}")
                return False

    def _run_audio_process(self) -> bool:
        """Run audio processing phase."""
        with PhaseLogger("PHASE 5: Audio Processing", self.logger) as phase:
            try:
                processor = AudioProcessor(self.config)
                results = processor.process_all()

                self.results["audio"] = results
                return True

            except Exception as e:
                phase.error(f"Audio processing failed: {e}")
                return False

    def _run_multicam_sync(self) -> bool:
        """Run multicam sync phase."""
        with PhaseLogger("PHASE 6: Multicam Sync", self.logger) as phase:
            try:
                inventory = load_json(self.config.analysis_dir / "inventory.json")

                # Check if we have multiple sources
                tripod_clips = inventory.get("tripod_camera", [])
                audio_files = inventory.get("audio_sources", [])

                if not tripod_clips and not audio_files:
                    phase.step("No additional sources for multicam - skipping")
                    return True

                syncer = MulticamSynchronizer(self.config)
                results = syncer.sync_sources(inventory)

                self.results["sync"] = results
                return True

            except Exception as e:
                phase.error(f"Multicam sync failed: {e}")
                return False

    def _run_audio_mix(self) -> bool:
        """Run audio mix decisions phase."""
        with PhaseLogger("PHASE 7: Audio Mix Analysis", self.logger) as phase:
            try:
                inventory = load_json(self.config.analysis_dir / "inventory.json")

                # Check if we have external audio sources
                audio_files = inventory.get("audio_sources", [])
                if not audio_files:
                    phase.step("No external audio sources - using camera audio")
                    return True

                sync_path = self.config.analysis_dir / "sync_offsets.json"
                sync_data = load_json(sync_path) if sync_path.exists() else {}

                analyzer = AudioMixAnalyzer(self.config)
                results = analyzer.analyze_and_decide(inventory, sync_data)

                self.results["audio_mix"] = results
                return True

            except Exception as e:
                phase.error(f"Audio mix analysis failed: {e}")
                return False

    def _run_multicam_edit(self) -> bool:
        """Run multicam edit decisions phase."""
        with PhaseLogger("PHASE 8: Multicam Edit Decisions", self.logger) as phase:
            try:
                inventory = load_json(self.config.analysis_dir / "inventory.json")

                # Check if we have multiple cameras
                tripod_clips = inventory.get("tripod_camera", [])
                if not tripod_clips:
                    phase.step("No tripod camera - single camera edit")
                    return True

                # Load analysis data
                stability_path = self.config.analysis_dir / "stability_scores.json"
                stability_data = load_json(stability_path) if stability_path.exists() else {"clips": []}

                shot_path = self.config.analysis_dir / "shot_boundaries.json"
                shot_data = load_json(shot_path) if shot_path.exists() else {"clips": []}

                sync_path = self.config.analysis_dir / "sync_offsets.json"
                sync_data = load_json(sync_path) if sync_path.exists() else {}

                editor = MulticamEditor(self.config)
                results = editor.generate_decisions(inventory, stability_data, shot_data, sync_data)

                self.results["multicam_edit"] = results
                return True

            except Exception as e:
                phase.error(f"Multicam edit decisions failed: {e}")
                return False

    def _run_assembly(self) -> bool:
        """Run assembly phase."""
        with PhaseLogger("PHASE 9: Assembly", self.logger) as phase:
            try:
                # Load sync data if available
                sync_path = self.config.analysis_dir / "sync_offsets.json"
                sync_data = load_json(sync_path) if sync_path.exists() else None

                assembler = TimelineAssembler(self.config)
                results = assembler.assemble_all(sync_data)

                self.results["assembly"] = results
                return True

            except Exception as e:
                phase.error(f"Assembly failed: {e}")
                return False

    def _print_summary(self):
        """Print final summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"  Total time: {format_duration(elapsed)}")
        self.logger.info("")

        # Output locations
        self.logger.info("  Outputs:")

        clips_count = len(list(self.config.clips_dir.glob("*.mov")))
        self.logger.info(f"    Clips: {self.config.clips_dir} ({clips_count} files)")

        master_files = list(self.config.master_dir.glob("*.mov"))
        if master_files:
            self.logger.info(f"    Master: {master_files[0]}")

        fcpxml_files = list(self.config.project_dir.glob("*.fcpxml"))
        for f in fcpxml_files:
            self.logger.info(f"    Project: {f}")

        self.logger.info("")
        self.logger.info("=" * 60)

        # Save complete results
        save_json(self.results, self.config.analysis_dir / "pipeline_results.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dad Cam Pipeline - Legacy Footage Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python dad_cam_pipeline.py -s "/path/to/footage" -o "./output"

  # Analyze only (dry run)
  python dad_cam_pipeline.py -s "/path/to/footage" -o "./output" --dry-run

  # Skip transcoding (use existing clips)
  python dad_cam_pipeline.py -s "/path/to/footage" -o "./output" --skip-transcode
        """
    )

    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source directory containing raw footage"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=None,
        help="Number of parallel transcode jobs (default: CPU/2)"
    )
    parser.add_argument(
        "--skip-transcode",
        action="store_true",
        help="Skip transcoding phase"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip phases 1-3 (use existing analysis data)"
    )
    parser.add_argument(
        "--skip-multicam",
        action="store_true",
        help="Skip multicam synchronization"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio processing phase"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, don't process files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate source directory
    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(
        name="dad_cam_pipeline",
        log_dir=output_dir / "logs",
        level=level
    )

    # Create configuration
    config = PipelineConfig()
    config.setup_paths(str(source_dir), str(output_dir))

    if args.parallel:
        config.parallel_jobs = args.parallel
    config.skip_transcode = args.skip_transcode
    config.skip_analysis = args.skip_analysis
    config.skip_audio = args.skip_audio
    config.skip_multicam = args.skip_multicam
    config.dry_run = args.dry_run

    # Run pipeline
    pipeline = DadCamPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
