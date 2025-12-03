#!/usr/bin/env python3
"""
02_analyze.py - Video Analysis for Technical Issues
====================================================

Analyzes source footage to detect:
1. Stuck/hot/dead pixels that persist across clips
2. Black frames at start/end of clips (for trimming)
3. Camera stability scores (for multicam switching decisions)
4. Vignette (corner darkening) per clip - zoom level affects intensity

Output:
- analysis/stuck_pixels.json
- analysis/black_frames.json
- analysis/stability_scores.json
- analysis/vignette.json
"""

import sys
import subprocess
import json
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import struct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils import load_json, save_json, ProgressBar
from lib.ffmpeg_utils import (
    probe_file, detect_black_frames, detect_silence,
    generate_vidstab_analysis, get_duration
)
from lib.logging_utils import get_logger, PhaseLogger, setup_logger
from config.settings import PipelineConfig


@dataclass
class StuckPixel:
    """Represents a detected stuck pixel."""
    x: int
    y: int
    pixel_type: str  # "hot" or "dead"
    mean_value: float
    variance: float
    contrast: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlackFrameRegion:
    """Represents a black frame region in a clip."""
    clip_path: str
    clip_order: int
    start_time: float
    end_time: float
    duration: float
    position: str  # "start", "end", or "middle"
    has_audio: bool  # True if audio present during black frames
    trim_recommended: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilitySegment:
    """Represents a stability segment in a clip."""
    start_time: float
    end_time: float
    avg_motion: float
    max_motion: float
    stability_score: float  # 0.0 = very shaky, 1.0 = rock solid
    classification: str  # "stable", "moderate", "shaky"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VignetteAnalysis:
    """Represents vignette analysis for a clip."""
    clip_path: str
    clip_order: int
    vignette_score: float  # 0.0 = none, 1.0 = severe
    corner_brightness: float
    center_brightness: float
    recommended_crop_percent: int  # 0-5%
    severity: str  # "none", "mild", "moderate", "severe"
    crop_filter: str  # FFmpeg crop filter string

    def to_dict(self) -> dict:
        return asdict(self)


class VideoAnalyzer:
    """Analyzes video files for technical issues."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.analysis_dir = config.analysis_dir

    def analyze_all(self, inventory: Dict) -> Dict:
        """
        Run all analyses on inventory files.

        Args:
            inventory: Inventory from discovery phase

        Returns:
            Combined analysis results
        """
        results = {
            "stuck_pixels": None,
            "black_frames": None,
            "stability_scores": None,
            "vignette": None,
        }

        # Stuck pixel detection
        with PhaseLogger("Stuck Pixel Detection", self.logger) as phase:
            main_clips = inventory.get("main_camera", [])
            if main_clips:
                phase.step(f"Analyzing {len(main_clips)} clips for stuck pixels...")
                stuck_pixels = self._detect_stuck_pixels(main_clips)
                results["stuck_pixels"] = stuck_pixels

                # Save results
                save_json(stuck_pixels, self.analysis_dir / "stuck_pixels.json")

                if stuck_pixels["pixels"]:
                    phase.warn(f"Found {len(stuck_pixels['pixels'])} stuck pixel(s)")
                    for p in stuck_pixels["pixels"]:
                        phase.substep(f"{p['pixel_type'].upper()} pixel at ({p['x']}, {p['y']}) "
                                    f"- contrast: {p['contrast']:.1f}")
                else:
                    phase.success("No stuck pixels detected")

        # Black frame detection
        with PhaseLogger("Black Frame Detection", self.logger) as phase:
            all_clips = inventory.get("main_camera", []) + inventory.get("tripod_camera", [])
            phase.step(f"Scanning {len(all_clips)} clips for black frames...")

            black_frames = self._detect_all_black_frames(all_clips)
            results["black_frames"] = black_frames

            # Save results
            save_json(black_frames, self.analysis_dir / "black_frames.json")

            clips_with_black = len([c for c in black_frames["clips"] if c["regions"]])
            if clips_with_black:
                phase.warn(f"{clips_with_black} clip(s) have black frames to trim")
            else:
                phase.success("No black frames detected")

        # Stability analysis (main camera only)
        with PhaseLogger("Stability Analysis", self.logger) as phase:
            main_clips = inventory.get("main_camera", [])
            if main_clips:
                phase.step(f"Analyzing camera stability for {len(main_clips)} clips...")
                stability = self._analyze_stability(main_clips)
                results["stability_scores"] = stability

                # Save results
                save_json(stability, self.analysis_dir / "stability_scores.json")

                # Calculate overall stats
                all_segments = []
                for clip in stability["clips"]:
                    all_segments.extend(clip.get("segments", []))

                if all_segments:
                    avg_score = sum(s["stability_score"] for s in all_segments) / len(all_segments)
                    phase.success(f"Average stability score: {avg_score:.2f}")

        # Vignette detection (all clips - zoom affects intensity)
        with PhaseLogger("Vignette Detection", self.logger) as phase:
            all_clips = inventory.get("main_camera", []) + inventory.get("tripod_camera", [])
            if all_clips:
                phase.step(f"Analyzing vignette for {len(all_clips)} clips...")
                vignette = self._analyze_vignette(all_clips)
                results["vignette"] = vignette

                # Save results
                save_json(vignette, self.analysis_dir / "vignette.json")

                # Report summary
                summary = vignette["summary"]
                if summary["clips_with_vignette"] > 0:
                    phase.warn(f"{summary['clips_with_vignette']} clip(s) have vignette")
                    for severity in ["severe", "moderate", "mild"]:
                        count = summary["severity_breakdown"].get(severity, 0)
                        if count > 0:
                            phase.substep(f"{severity.capitalize()}: {count} clip(s)")
                else:
                    phase.success("No significant vignette detected")

        return results

    def _detect_stuck_pixels(self, clips: List[Dict]) -> Dict:
        """
        Detect stuck pixels by analyzing frames across multiple clips.

        Strategy:
        1. Sample N frames from M clips spread across the timeline
        2. For each pixel, compute variance across all samples
        3. Flag pixels with very low variance but extreme brightness
        4. Filter by visibility (contrast with neighbors)
        """
        settings = self.config.stuck_pixel

        # Select clips to sample (evenly distributed)
        sample_clip_count = min(settings.sample_clips, len(clips))
        step = max(1, len(clips) // sample_clip_count)
        selected_clips = [clips[i] for i in range(0, len(clips), step)][:sample_clip_count]

        self.logger.info(f"  Sampling from {len(selected_clips)} clips...")

        # Collect pixel statistics
        pixel_stats = self._collect_pixel_stats(
            selected_clips,
            frames_per_clip=settings.sample_frames // sample_clip_count
        )

        if pixel_stats is None:
            return {"pixels": [], "delogo_filter": "", "analysis_details": {}}

        # Analyze for stuck pixels
        stuck_pixels = self._find_stuck_pixels(
            pixel_stats,
            variance_threshold=settings.variance_threshold,
            hot_threshold=settings.hot_pixel_threshold,
            dead_threshold=settings.dead_pixel_threshold,
            contrast_threshold=settings.contrast_threshold,
            border_margin=settings.border_margin
        )

        # Generate delogo filter string
        delogo_filter = self._generate_delogo_filter(stuck_pixels)

        return {
            "pixels": [p.to_dict() for p in stuck_pixels],
            "delogo_filter": delogo_filter,
            "analysis_details": {
                "clips_sampled": len(selected_clips),
                "frames_analyzed": pixel_stats.get("total_frames", 0),
                "threshold_settings": {
                    "variance": settings.variance_threshold,
                    "contrast": settings.contrast_threshold,
                    "hot_brightness": settings.hot_pixel_threshold,
                    "dead_brightness": settings.dead_pixel_threshold,
                }
            }
        }

    def _collect_pixel_stats(self, clips: List[Dict], frames_per_clip: int) -> Optional[Dict]:
        """
        Collect pixel statistics from sampled frames using FFmpeg.

        Uses FFmpeg to extract frames and compute per-pixel mean/variance.
        """
        try:
            import numpy as np
        except ImportError:
            self.logger.warning("NumPy not available - skipping stuck pixel detection")
            return None

        all_frames = []
        width, height = None, None

        for clip in clips:
            clip_path = clip["original_path"]
            duration = clip.get("duration", 0)

            if duration <= 0:
                continue

            # Sample frames evenly across clip
            for i in range(frames_per_clip):
                timestamp = duration * (i + 0.5) / frames_per_clip

                with tempfile.NamedTemporaryFile(suffix='.raw', delete=True) as tmp:
                    # Extract single frame as raw RGB
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(timestamp),
                        '-i', clip_path,
                        '-vframes', '1',
                        '-f', 'rawvideo',
                        '-pix_fmt', 'rgb24',
                        '-s', '360x270',  # Downscale for speed
                        tmp.name
                    ]

                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode == 0 and Path(tmp.name).exists():
                        try:
                            data = np.fromfile(tmp.name, dtype=np.uint8)
                            if len(data) == 360 * 270 * 3:
                                frame = data.reshape((270, 360, 3))
                                all_frames.append(frame)
                                if width is None:
                                    width, height = 360, 270
                        except Exception:
                            pass

        if not all_frames:
            return None

        # Stack frames and compute statistics
        stacked = np.stack(all_frames, axis=0).astype(np.float32)
        mean = np.mean(stacked, axis=0)
        variance = np.var(stacked, axis=0)

        # Average across color channels
        mean_gray = np.mean(mean, axis=2)
        variance_gray = np.mean(variance, axis=2)

        return {
            "mean": mean_gray,
            "variance": variance_gray,
            "width": width,
            "height": height,
            "total_frames": len(all_frames),
            "scale_factor": 4  # We downscaled by 4x
        }

    def _find_stuck_pixels(
        self,
        stats: Dict,
        variance_threshold: float,
        hot_threshold: int,
        dead_threshold: int,
        contrast_threshold: float,
        border_margin: int
    ) -> List[StuckPixel]:
        """Find stuck pixels from computed statistics."""
        try:
            import numpy as np
            from scipy.ndimage import uniform_filter
        except ImportError:
            return []

        mean = stats["mean"]
        variance = stats["variance"]
        scale = stats["scale_factor"]
        height, width = mean.shape

        # Margin in scaled coordinates
        margin = border_margin // scale

        # Find low-variance pixels
        low_var_mask = variance < variance_threshold

        # Find extreme brightness pixels
        hot_mask = mean > hot_threshold
        dead_mask = mean < dead_threshold

        # Stuck = low variance AND (hot OR dead)
        stuck_mask = low_var_mask & (hot_mask | dead_mask)

        # Apply border exclusion
        stuck_mask[:margin, :] = False
        stuck_mask[-margin:, :] = False
        stuck_mask[:, :margin] = False
        stuck_mask[:, -margin:] = False

        # Compute local mean for contrast calculation
        local_mean = uniform_filter(mean, size=5)
        contrast = np.abs(mean - local_mean)

        # Only keep pixels with sufficient contrast (visible stuck pixels)
        visible_mask = stuck_mask & (contrast > contrast_threshold)

        # Extract coordinates and create StuckPixel objects
        stuck_pixels = []
        coords = np.argwhere(visible_mask)

        for y, x in coords:
            # Scale coordinates back to original resolution
            orig_x = x * scale
            orig_y = y * scale

            pixel = StuckPixel(
                x=int(orig_x),
                y=int(orig_y),
                pixel_type="hot" if mean[y, x] > 128 else "dead",
                mean_value=float(mean[y, x]),
                variance=float(variance[y, x]),
                contrast=float(contrast[y, x]),
                confidence=min(1.0, contrast[y, x] / 100)
            )
            stuck_pixels.append(pixel)

        return stuck_pixels

    def _generate_delogo_filter(self, pixels: List[StuckPixel]) -> str:
        """Generate FFmpeg delogo filter string for stuck pixels."""
        if not pixels:
            return ""

        filters = []
        for p in pixels:
            # Use 3x3 area centered on pixel
            filters.append(f"delogo=x={p.x-1}:y={p.y-1}:w=3:h=3")

        return ",".join(filters)

    def _detect_all_black_frames(self, clips: List[Dict]) -> Dict:
        """Detect black frames at start/end of all clips."""
        settings = self.config.black_frame
        results = []

        progress = ProgressBar(len(clips), prefix="  Black frames")

        for clip in clips:
            clip_path = clip["original_path"]
            clip_order = clip.get("order", 0)
            duration = clip.get("duration", 0)

            regions = []

            # Detect black frames
            black_regions = detect_black_frames(
                Path(clip_path),
                threshold=settings.brightness_threshold / 255.0,
                min_duration=settings.min_duration_frames / 30.0
            )

            # Detect silence for cross-reference
            silence_regions = detect_silence(
                Path(clip_path),
                threshold_db=settings.silence_threshold_db,
                min_duration=0.1
            )

            for br in black_regions:
                start = br["start"]
                end = br["end"]
                dur = br["duration"]

                # Determine position
                if start < settings.scan_seconds:
                    position = "start"
                elif duration - end < settings.scan_seconds:
                    position = "end"
                else:
                    position = "middle"

                # Check if there's audio during black frames
                has_audio = not any(
                    sr["start"] <= start and sr["end"] >= end
                    for sr in silence_regions
                )

                # Recommend trimming if at start/end and silent
                trim_recommended = (position in ("start", "end")) and not has_audio

                region = BlackFrameRegion(
                    clip_path=clip_path,
                    clip_order=clip_order,
                    start_time=start,
                    end_time=end,
                    duration=dur,
                    position=position,
                    has_audio=has_audio,
                    trim_recommended=trim_recommended
                )
                regions.append(region)

            results.append({
                "clip_path": clip_path,
                "clip_order": clip_order,
                "regions": [r.to_dict() for r in regions]
            })

            progress.update(1)

        progress.finish()

        return {
            "clips": results,
            "summary": {
                "total_clips": len(clips),
                "clips_with_black_frames": len([r for r in results if r["regions"]]),
                "total_regions": sum(len(r["regions"]) for r in results),
            }
        }

    def _analyze_stability(self, clips: List[Dict]) -> Dict:
        """Analyze camera stability for main camera clips."""
        settings = self.config.stability
        results = []

        # For efficiency, sample every Nth clip for detailed analysis
        # and interpolate for others
        sample_interval = max(1, len(clips) // 20)  # Analyze ~20 clips max

        progress = ProgressBar(len(clips), prefix="  Stability")

        for i, clip in enumerate(clips):
            clip_path = clip["original_path"]
            clip_order = clip.get("order", 0)
            duration = clip.get("duration", 0)

            segments = []

            # Only do full analysis on sampled clips
            if i % sample_interval == 0:
                segments = self._analyze_clip_stability(clip_path, duration)
            else:
                # Quick estimate based on file (placeholder score)
                segments = [{
                    "start_time": 0,
                    "end_time": duration,
                    "avg_motion": 0,
                    "max_motion": 0,
                    "stability_score": 0.7,  # Default moderate
                    "classification": "moderate"
                }]

            results.append({
                "clip_path": clip_path,
                "clip_order": clip_order,
                "duration": duration,
                "segments": segments,
                "overall_score": self._compute_overall_stability(segments)
            })

            progress.update(1)

        progress.finish()

        return {
            "clips": results,
            "summary": {
                "total_clips": len(clips),
                "avg_stability": sum(r["overall_score"] for r in results) / len(results) if results else 0,
            }
        }

    def _analyze_clip_stability(self, clip_path: str, duration: float) -> List[Dict]:
        """
        Analyze stability of a single clip using vidstab.

        Returns list of stability segments.
        """
        settings = self.config.stability

        with tempfile.NamedTemporaryFile(suffix='.trf', delete=False) as trf_file:
            trf_path = Path(trf_file.name)

        try:
            # Run vidstab detection (analyze first 30 seconds max)
            analyze_duration = min(30, duration)

            cmd = [
                'ffmpeg', '-y',
                '-t', str(analyze_duration),
                '-i', clip_path,
                '-vf', f'vidstabdetect=shakiness={settings.shakiness}:accuracy={settings.accuracy}:result={trf_path}',
                '-f', 'null', '-'
            ]

            result = subprocess.run(cmd, capture_output=True)

            if result.returncode != 0 or not trf_path.exists():
                return [{
                    "start_time": 0,
                    "end_time": duration,
                    "avg_motion": 0,
                    "max_motion": 0,
                    "stability_score": 0.7,
                    "classification": "moderate"
                }]

            # Parse transform file
            motions = self._parse_vidstab_trf(trf_path)

            if not motions:
                return [{
                    "start_time": 0,
                    "end_time": duration,
                    "avg_motion": 0,
                    "max_motion": 0,
                    "stability_score": 0.7,
                    "classification": "moderate"
                }]

            # Segment by stability
            segments = self._segment_by_stability(motions, duration, settings)

            return segments

        finally:
            if trf_path.exists():
                trf_path.unlink()

    def _parse_vidstab_trf(self, trf_path: Path) -> List[float]:
        """Parse vidstab transform file to extract motion magnitudes."""
        import math

        motions = []

        try:
            with open(trf_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Format: frame dx dy da (translation x, y, rotation)
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            dx = float(parts[1])
                            dy = float(parts[2])
                            motion = math.sqrt(dx*dx + dy*dy)
                            motions.append(motion)
                        except (ValueError, IndexError):
                            pass
        except Exception as e:
            self.logger.debug(f"Could not parse transform file: {e}")

        return motions

    def _segment_by_stability(
        self,
        motions: List[float],
        duration: float,
        settings
    ) -> List[Dict]:
        """Segment clip by stability scores."""
        if not motions:
            return []

        fps = 30  # Assume 30fps
        window_frames = int(settings.min_segment_seconds * fps)

        segments = []
        frame_duration = duration / len(motions) if motions else 0

        # Simple segmentation: average motion over windows
        for start_frame in range(0, len(motions), window_frames):
            end_frame = min(start_frame + window_frames, len(motions))
            window_motions = motions[start_frame:end_frame]

            if not window_motions:
                continue

            avg_motion = sum(window_motions) / len(window_motions)
            max_motion = max(window_motions)

            # Convert motion to stability score (inverse relationship)
            # Motion of 0 = score 1.0, motion of 10+ = score ~0
            stability_score = 1.0 / (1.0 + avg_motion * 0.5)

            # Classify
            if stability_score >= settings.stable_threshold:
                classification = "stable"
            elif stability_score >= settings.unstable_threshold:
                classification = "moderate"
            else:
                classification = "shaky"

            segments.append({
                "start_time": start_frame * frame_duration,
                "end_time": end_frame * frame_duration,
                "avg_motion": avg_motion,
                "max_motion": max_motion,
                "stability_score": stability_score,
                "classification": classification
            })

        return segments

    def _compute_overall_stability(self, segments: List[Dict]) -> float:
        """Compute overall stability score from segments."""
        if not segments:
            return 0.7  # Default moderate

        total_duration = sum(s["end_time"] - s["start_time"] for s in segments)
        if total_duration == 0:
            return 0.7

        weighted_score = sum(
            s["stability_score"] * (s["end_time"] - s["start_time"])
            for s in segments
        )

        return weighted_score / total_duration

    def _analyze_vignette(self, clips: List[Dict]) -> Dict:
        """
        Analyze vignette (corner darkening) for all clips.

        Each clip is analyzed because zoom level affects vignette intensity.
        Returns per-clip analysis with recommended crop percentages.
        """
        settings = self.config.vignette
        results = []

        progress = ProgressBar(len(clips), prefix="  Vignette")

        for clip in clips:
            clip_path = clip["original_path"]
            clip_order = clip.get("order", 0)
            duration = clip.get("duration", 0)

            analysis = self._detect_clip_vignette(clip_path, duration, settings)
            analysis["clip_path"] = clip_path
            analysis["clip_order"] = clip_order

            results.append(analysis)
            progress.update(1)

        progress.finish()

        # Calculate summary statistics
        clips_with_vignette = [r for r in results if r["severity"] != "none"]
        severity_counts = {"none": 0, "mild": 0, "moderate": 0, "severe": 0}
        for r in results:
            severity_counts[r["severity"]] += 1

        return {
            "clips": results,
            "summary": {
                "total_clips": len(clips),
                "clips_with_vignette": len(clips_with_vignette),
                "severity_breakdown": severity_counts,
                "avg_vignette_score": sum(r["vignette_score"] for r in results) / len(results) if results else 0,
            }
        }

    def _detect_clip_vignette(self, clip_path: str, duration: float, settings) -> Dict:
        """
        Detect vignette in a single clip by comparing corner brightness to center.

        Algorithm:
        1. Sample N frames evenly across clip
        2. For each frame, measure average brightness in corners vs center
        3. Compute ratio (lower = more vignette)
        4. Average across frames
        5. Classify severity and recommend crop
        """
        try:
            import numpy as np
        except ImportError:
            return self._default_vignette_result()

        if duration <= 0:
            return self._default_vignette_result()

        corner_brightnesses = []
        center_brightnesses = []

        # Sample frames across clip
        for i in range(settings.sample_frames):
            timestamp = duration * (i + 0.5) / settings.sample_frames

            with tempfile.NamedTemporaryFile(suffix='.raw', delete=True) as tmp:
                # Extract single frame as grayscale
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(timestamp),
                    '-i', clip_path,
                    '-vframes', '1',
                    '-f', 'rawvideo',
                    '-pix_fmt', 'gray',
                    '-s', '320x240',  # Downscale for speed
                    tmp.name
                ]

                result = subprocess.run(cmd, capture_output=True)

                if result.returncode == 0 and Path(tmp.name).exists():
                    try:
                        data = np.fromfile(tmp.name, dtype=np.uint8)
                        if len(data) == 320 * 240:
                            frame = data.reshape((240, 320))

                            # Measure corners (all 4 corners)
                            corner_size = settings.corner_sample_size
                            corners = [
                                frame[:corner_size, :corner_size],           # Top-left
                                frame[:corner_size, -corner_size:],          # Top-right
                                frame[-corner_size:, :corner_size],          # Bottom-left
                                frame[-corner_size:, -corner_size:],         # Bottom-right
                            ]
                            corner_avg = np.mean([np.mean(c) for c in corners])

                            # Measure center (middle region)
                            h, w = frame.shape
                            center_region = frame[h//3:2*h//3, w//3:2*w//3]
                            center_avg = np.mean(center_region)

                            corner_brightnesses.append(corner_avg)
                            center_brightnesses.append(center_avg)
                    except Exception:
                        pass

        if not corner_brightnesses or not center_brightnesses:
            return self._default_vignette_result()

        # Calculate average ratio
        avg_corner = np.mean(corner_brightnesses)
        avg_center = np.mean(center_brightnesses)

        # Avoid division by zero
        if avg_center < 1:
            avg_center = 1

        ratio = avg_corner / avg_center

        # Classify severity
        if ratio >= settings.mild_threshold:
            severity = "none"
            crop_percent = 0
        elif ratio >= settings.moderate_threshold:
            severity = "mild"
            crop_percent = settings.mild_crop_percent
        elif ratio >= settings.severe_threshold:
            severity = "moderate"
            crop_percent = settings.moderate_crop_percent
        else:
            severity = "severe"
            crop_percent = settings.severe_crop_percent

        # Clamp to max
        crop_percent = min(crop_percent, settings.max_crop_percent)

        # Vignette score: 0 = none, 1 = severe
        vignette_score = max(0, min(1, 1 - ratio))

        # Generate crop filter (crops equal amount from all sides)
        crop_filter = self._generate_crop_filter(crop_percent)

        return {
            "vignette_score": float(vignette_score),
            "corner_brightness": float(avg_corner),
            "center_brightness": float(avg_center),
            "brightness_ratio": float(ratio),
            "recommended_crop_percent": crop_percent,
            "severity": severity,
            "crop_filter": crop_filter,
        }

    def _default_vignette_result(self) -> Dict:
        """Return default vignette analysis result."""
        return {
            "vignette_score": 0.0,
            "corner_brightness": 0.0,
            "center_brightness": 0.0,
            "brightness_ratio": 1.0,
            "recommended_crop_percent": 0,
            "severity": "none",
            "crop_filter": "",
        }

    def _generate_crop_filter(self, crop_percent: int) -> str:
        """
        Generate FFmpeg crop filter string.

        Crops equal percentage from all sides while maintaining aspect ratio.
        For 1440x1080 input with 4:3 SAR, we need to maintain the display aspect.
        """
        if crop_percent <= 0:
            return ""

        # Calculate crop amounts (percentage from each side)
        # crop=w:h:x:y - crops to w×h starting at x,y
        # For symmetric crop: new_w = w * (1 - 2*crop), same for height
        # Example: 5% crop means 5% off each side = 90% remaining
        remaining = (100 - 2 * crop_percent) / 100.0

        # Use FFmpeg expressions to handle any input size
        filter_str = f"crop=iw*{remaining:.4f}:ih*{remaining:.4f}"

        return filter_str


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze video files for technical issues"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory (containing analysis/inventory.json)"
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

    # Load inventory
    inventory_path = output_dir / "analysis" / "inventory.json"
    if not inventory_path.exists():
        get_logger().error(f"Inventory not found: {inventory_path}")
        get_logger().error("Run 01_discover.py first")
        return 1

    inventory = load_json(inventory_path)

    # Create config
    config = PipelineConfig()
    config.output_dir = output_dir
    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer = VideoAnalyzer(config)
    results = analyzer.analyze_all(inventory)

    return 0


if __name__ == "__main__":
    sys.exit(main())
