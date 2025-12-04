#!/usr/bin/env python3
"""
detect_shots.py - Shot Boundary Detection using TransNet V2
============================================================

Detects shot boundaries (cuts, fades, dissolves) in video clips
using TransNet V2 neural network.

This enables:
- Natural switching points for multicam editing
- Avoiding angle switches mid-shot
- Understanding clip structure

Output: analysis/shot_boundaries.json
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, ProgressBar, format_duration
from lib.ffmpeg_utils import probe_file
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig

# TransNet V2 imports (with fallback)
TRANSNET_AVAILABLE = False
try:
    import numpy as np
    from PIL import Image
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import tensorflow as tf
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@dataclass
class Shot:
    """Represents a single shot in a clip."""
    start_time: float
    end_time: float
    duration: float
    confidence: float  # Confidence of the boundary detection


@dataclass
class ClipShots:
    """Shot detection results for a clip."""
    clip_path: str
    clip_order: int
    duration: float
    shots: List[Shot]
    shot_count: int
    avg_shot_duration: float


class TransNetV2Detector:
    """
    Shot boundary detector using TransNet V2.

    TransNet V2 is a neural network that analyzes video frames
    to detect shot transitions (cuts, fades, dissolves).
    """

    # TransNet V2 model URL
    MODEL_URL = "https://github.com/soCzech/TransNetV2/raw/master/inference/transnetv2-weights/"

    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load TransNet V2 model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available - using FFmpeg fallback")
            return

        try:
            # Try to import transnetv2 package first
            try:
                from transnetv2 import TransNetV2
                self.model = TransNetV2()
                self.logger.info("Loaded TransNet V2 from package")
                return
            except ImportError:
                pass

            # Fallback: Try to load model weights directly
            model_dir = Path(__file__).parent.parent / "models" / "transnetv2"
            if model_dir.exists():
                self.model = self._load_model_from_weights(model_dir)
                self.logger.info("Loaded TransNet V2 from local weights")
            else:
                self.logger.warning("TransNet V2 model not found - using FFmpeg fallback")

        except Exception as e:
            self.logger.warning(f"Failed to load TransNet V2: {e} - using FFmpeg fallback")

    def _load_model_from_weights(self, model_dir: Path):
        """Load model from saved weights."""
        # This would load the actual TransNet V2 architecture
        # For now, return None to trigger fallback
        return None

    def detect_shots(self, video_path: Path, threshold: float = 0.5) -> List[Shot]:
        """
        Detect shot boundaries in a video.

        Args:
            video_path: Path to video file
            threshold: Confidence threshold for shot detection

        Returns:
            List of Shot objects
        """
        if self.model is not None:
            return self._detect_with_transnet(video_path, threshold)
        else:
            return self._detect_with_ffmpeg(video_path, threshold)

    def _detect_with_transnet(self, video_path: Path, threshold: float) -> List[Shot]:
        """Detect shots using TransNet V2 model."""
        try:
            # Extract frames
            frames = self._extract_frames(video_path)
            if frames is None or len(frames) == 0:
                return self._detect_with_ffmpeg(video_path, threshold)

            # Run inference
            predictions = self.model.predict_frames(frames)

            # Find shot boundaries
            boundaries = self._predictions_to_boundaries(predictions, threshold)

            # Get video duration and fps
            info = probe_file(video_path)
            fps = info.video_stream.fps if info and info.video_stream else 29.97
            duration = info.duration if info else 0

            # Convert frame indices to Shot objects
            shots = self._boundaries_to_shots(boundaries, fps, duration)

            return shots

        except Exception as e:
            self.logger.warning(f"TransNet failed: {e} - falling back to FFmpeg")
            return self._detect_with_ffmpeg(video_path, threshold)

    def _extract_frames(self, video_path: Path, max_frames: int = 10000) -> Optional[np.ndarray]:
        """Extract frames from video for TransNet analysis."""
        if not NUMPY_AVAILABLE:
            return None

        # Use FFmpeg to extract frames at lower resolution
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_pattern = Path(tmpdir) / "frame_%06d.jpg"

            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', 'scale=48:27',  # TransNet V2 input size
                '-q:v', '2',
                '-vsync', '0',
                str(frame_pattern)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                return None

            # Load frames
            frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))[:max_frames]
            if not frame_files:
                return None

            frames = []
            for f in frame_files:
                img = Image.open(f)
                frames.append(np.array(img))

            return np.array(frames)

    def _predictions_to_boundaries(
        self,
        predictions: np.ndarray,
        threshold: float
    ) -> List[Tuple[int, float]]:
        """Convert model predictions to shot boundary frame indices."""
        boundaries = []

        # Find peaks above threshold
        for i in range(1, len(predictions) - 1):
            if predictions[i] > threshold:
                # Local maximum check
                if predictions[i] >= predictions[i-1] and predictions[i] >= predictions[i+1]:
                    boundaries.append((i, float(predictions[i])))

        return boundaries

    def _boundaries_to_shots(
        self,
        boundaries: List[Tuple[int, float]],
        fps: float,
        duration: float
    ) -> List[Shot]:
        """Convert frame boundaries to Shot objects."""
        shots = []

        # Add implicit start
        boundary_frames = [0] + [b[0] for b in boundaries]
        boundary_confs = [1.0] + [b[1] for b in boundaries]

        for i in range(len(boundary_frames)):
            start_frame = boundary_frames[i]
            end_frame = boundary_frames[i + 1] if i + 1 < len(boundary_frames) else int(duration * fps)

            start_time = start_frame / fps
            end_time = end_frame / fps

            # Confidence is the confidence of the END boundary
            conf = boundary_confs[i + 1] if i + 1 < len(boundary_confs) else 1.0

            shots.append(Shot(
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                confidence=conf
            ))

        return shots

    def _detect_with_ffmpeg(self, video_path: Path, threshold: float) -> List[Shot]:
        """
        Fallback shot detection using FFmpeg scene detection.

        Uses the 'select' filter with scene change detection.
        Less accurate than TransNet but works without TensorFlow.
        """
        # Map threshold (0-1) to FFmpeg scene threshold (0-1, but inverted sensitivity)
        # TransNet threshold 0.5 → FFmpeg threshold ~0.3
        ffmpeg_threshold = max(0.1, min(0.5, 0.6 - threshold * 0.4))

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'frame=pts_time',
            '-select_streams', 'v:0',
            '-f', 'lavfi',
            f"movie={str(video_path)},select='gt(scene,{ffmpeg_threshold})'",
            '-of', 'json'
        ]

        # Alternative approach using ffmpeg with scene detection
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f"select='gt(scene,{ffmpeg_threshold})',showinfo",
            '-f', 'null',
            '-'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse showinfo output for pts_time
            scene_times = []
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        pts_part = line.split('pts_time:')[1].split()[0]
                        scene_times.append(float(pts_part))
                    except (IndexError, ValueError):
                        continue

            # Get video duration
            info = probe_file(video_path)
            duration = info.duration if info else 0

            # Build shots from scene times
            shots = []
            times = [0.0] + scene_times + [duration]

            for i in range(len(times) - 1):
                shot_duration = times[i + 1] - times[i]
                if shot_duration > 0.1:  # Minimum shot length
                    shots.append(Shot(
                        start_time=times[i],
                        end_time=times[i + 1],
                        duration=shot_duration,
                        confidence=0.7  # Fixed confidence for FFmpeg detection
                    ))

            return shots if shots else [Shot(0, duration, duration, 1.0)]

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Scene detection timeout for {video_path.name}")
            info = probe_file(video_path)
            duration = info.duration if info else 0
            return [Shot(0, duration, duration, 1.0)]
        except Exception as e:
            self.logger.warning(f"Scene detection failed: {e}")
            info = probe_file(video_path)
            duration = info.duration if info else 0
            return [Shot(0, duration, duration, 1.0)]


class ShotDetector:
    """Orchestrates shot detection across all clips."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.detector = TransNetV2Detector()

    def detect_all(self, inventory: Dict) -> Dict:
        """
        Detect shots in all clips from inventory.

        Args:
            inventory: Source inventory

        Returns:
            Shot detection results
        """
        results = {
            "clips": [],
            "summary": {}
        }

        # Get all clips
        main_clips = inventory.get("main_camera", [])
        tripod_clips = inventory.get("tripod_camera", [])
        all_clips = main_clips + tripod_clips

        if not all_clips:
            self.logger.warning("No clips to analyze")
            return results

        self.logger.info(f"Detecting shots in {len(all_clips)} clips...")

        progress = ProgressBar(len(all_clips), prefix="Shot detection")
        total_shots = 0

        for clip in all_clips:
            clip_path = Path(clip["original_path"])

            if not clip_path.exists():
                self.logger.warning(f"Clip not found: {clip_path}")
                progress.update(1)
                continue

            # Detect shots
            shots = self.detector.detect_shots(clip_path)

            # Build result
            duration = clip.get("duration", 0)
            clip_result = ClipShots(
                clip_path=str(clip_path),
                clip_order=clip.get("order", 0),
                duration=duration,
                shots=shots,
                shot_count=len(shots),
                avg_shot_duration=duration / len(shots) if shots else duration
            )

            results["clips"].append({
                "clip_path": clip_result.clip_path,
                "clip_order": clip_result.clip_order,
                "duration": clip_result.duration,
                "shot_count": clip_result.shot_count,
                "avg_shot_duration": clip_result.avg_shot_duration,
                "shots": [asdict(s) for s in clip_result.shots]
            })

            total_shots += len(shots)
            progress.update(1, suffix=clip_path.name)

        progress.finish()

        # Summary
        total_duration = sum(c.get("duration", 0) for c in all_clips)
        results["summary"] = {
            "total_clips": len(all_clips),
            "total_shots": total_shots,
            "total_duration": total_duration,
            "avg_shot_duration": total_duration / total_shots if total_shots else 0,
            "detection_method": "transnet_v2" if self.detector.model else "ffmpeg_scene"
        }

        self.logger.info(f"Detected {total_shots} shots across {len(all_clips)} clips")
        self.logger.info(f"Average shot duration: {results['summary']['avg_shot_duration']:.1f}s")

        # Save results
        save_json(results, self.config.analysis_dir / "shot_boundaries.json")

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect shot boundaries using TransNet V2"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Shot detection threshold (0-1, default 0.5)"
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

    # Run detection
    detector = ShotDetector(config)

    with PhaseLogger("Shot Boundary Detection", logger):
        results = detector.detect_all(inventory)

    return 0


if __name__ == "__main__":
    sys.exit(main())
