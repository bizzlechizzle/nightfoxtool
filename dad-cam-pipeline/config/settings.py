"""
Dad Cam Pipeline - Configuration Settings
==========================================
Central configuration for all pipeline scripts.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class TranscodeSettings:
    """Video transcoding configuration."""
    # Output codec settings - H.265 with Mac hardware decode support
    video_codec: str = "libx265"
    video_preset: str = "medium"
    video_crf: int = 18
    pixel_format: str = "yuv420p"  # 4:2:0 8-bit (matches source)
    video_tag: str = "hvc1"  # Apple QuickTime compatibility
    video_profile: str = "main"  # Main profile for 8-bit

    # Color metadata
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    colorspace: str = "bt709"

    # Deinterlacing
    deinterlace: bool = True
    deinterlace_filter: str = "yadif=0:-1:0"  # 0=send frame, -1=auto parity, 0=all frames

    # Frame rate
    output_fps: str = "30000/1001"  # 29.97p

    # Audio settings
    audio_codec: str = "aac"
    audio_bitrate: str = "256k"

    # Container
    container: str = "mov"
    movflags: str = "+faststart"

    # x265 specific params
    x265_params: str = ""


@dataclass
class AudioSettings:
    """Audio processing configuration."""
    # Loudness normalization
    target_lufs: float = -14.0
    target_tp: float = -1.5  # True peak
    target_lra: float = 11.0  # Loudness range

    # Hum removal
    hum_frequency: int = 60  # Hz (60 for US, 50 for EU)
    hum_harmonics: int = 5  # Number of harmonics to remove
    hum_detection_threshold: float = -40.0  # dB relative to speech band

    # Notch filter widths (increase for higher harmonics)
    notch_base_width: float = 2.0
    notch_width_increment: float = 1.0


@dataclass
class StuckPixelSettings:
    """Stuck pixel detection configuration."""
    # Sampling
    sample_frames: int = 50  # Number of frames to sample
    sample_clips: int = 10   # Number of clips to sample from

    # Detection thresholds
    variance_threshold: float = 5.0  # Max variance for "stuck" pixel
    hot_pixel_threshold: int = 240   # Min brightness for hot pixel
    dead_pixel_threshold: int = 15   # Max brightness for dead pixel

    # Visibility threshold - only fix if clearly visible
    contrast_threshold: float = 30.0  # Min contrast vs neighbors to fix

    # Border exclusion (vignette zone)
    border_margin: int = 20  # Pixels from edge to ignore


@dataclass
class BlackFrameSettings:
    """Black frame detection configuration."""
    brightness_threshold: int = 10  # Max avg brightness for "black"
    min_duration_frames: int = 3    # Min consecutive frames (0.1s @ 30fps)
    scan_seconds: float = 5.0       # Seconds to scan at start/end

    # Audio cross-reference
    silence_threshold_db: float = -50.0  # dB threshold for silence


@dataclass
class StabilitySettings:
    """Camera stability analysis configuration."""
    # vidstab detection settings
    shakiness: int = 10      # 1-10, higher = more sensitive
    accuracy: int = 15       # Analysis accuracy

    # Segment classification
    stable_threshold: float = 0.6    # Score above = stable
    unstable_threshold: float = 0.4  # Score below = shaky

    # Minimum segment duration
    min_segment_seconds: float = 3.0


@dataclass
class VignetteSettings:
    """Vignette detection configuration."""
    # Sampling
    sample_frames: int = 5         # Frames to sample per clip
    corner_sample_size: int = 50   # Pixels from corner to sample

    # Detection thresholds
    # Ratio of corner brightness to center (lower = more vignette)
    mild_threshold: float = 0.85   # Below this = mild vignette
    moderate_threshold: float = 0.70  # Below this = moderate
    severe_threshold: float = 0.55    # Below this = severe

    # Crop limits
    max_crop_percent: int = 5      # Never crop more than 5%
    mild_crop_percent: int = 1     # Crop for mild vignette
    moderate_crop_percent: int = 3  # Crop for moderate vignette
    severe_crop_percent: int = 5   # Crop for severe vignette


@dataclass
class SyncSettings:
    """Multicam synchronization configuration."""
    # Audio extraction
    sample_duration: float = 60.0  # Seconds of audio to analyze
    sample_rate: int = 8000        # Downsampled rate for correlation

    # Cross-correlation
    min_confidence: float = 0.5    # Min correlation peak ratio

    # Drift detection
    drift_check_interval: float = 1800.0  # Check every 30 minutes
    max_drift_frames_per_hour: int = 2    # Alert if drift exceeds this


@dataclass
class AssemblySettings:
    """Timeline assembly configuration."""
    # J/L cuts
    audio_overlap_seconds: float = 0.5  # Audio overlap duration
    crossfade_duration: float = 0.5     # Audio crossfade length
    crossfade_curve: str = "tri"        # Crossfade curve type

    # Timeline settings
    start_timecode: str = "01:00:00:00"  # NDF timecode

    # Multicam switching
    switch_hysteresis_seconds: float = 5.0  # Min time between switches
    prefer_handheld_threshold: float = 0.6  # Stability threshold


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # Paths (will be set at runtime)
    source_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Output subdirectories
    clips_subdir: str = "clips"
    master_subdir: str = "master"
    multicam_subdir: str = "multicam"
    project_subdir: str = "project"
    analysis_subdir: str = "analysis"
    logs_subdir: str = "logs"

    # Processing options
    parallel_jobs: int = 2  # Conservative default for H.265 encoding
    skip_transcode: bool = False
    skip_analysis: bool = False
    skip_audio: bool = False
    skip_multicam: bool = False
    dry_run: bool = False

    # Sub-configurations
    transcode: TranscodeSettings = field(default_factory=TranscodeSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    stuck_pixel: StuckPixelSettings = field(default_factory=StuckPixelSettings)
    black_frame: BlackFrameSettings = field(default_factory=BlackFrameSettings)
    stability: StabilitySettings = field(default_factory=StabilitySettings)
    vignette: VignetteSettings = field(default_factory=VignetteSettings)
    sync: SyncSettings = field(default_factory=SyncSettings)
    assembly: AssemblySettings = field(default_factory=AssemblySettings)

    # File naming
    output_prefix: str = "dad_cam"

    def setup_paths(self, source: str, output: str) -> None:
        """Initialize paths and create output directories."""
        self.source_dir = Path(source)
        self.output_dir = Path(output)

        # Create output subdirectories
        for subdir in [self.clips_subdir, self.master_subdir, self.multicam_subdir,
                       self.project_subdir, self.analysis_subdir, self.logs_subdir]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def clips_dir(self) -> Path:
        return self.output_dir / self.clips_subdir

    @property
    def master_dir(self) -> Path:
        return self.output_dir / self.master_subdir

    @property
    def multicam_dir(self) -> Path:
        return self.output_dir / self.multicam_subdir

    @property
    def project_dir(self) -> Path:
        return self.output_dir / self.project_subdir

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / self.analysis_subdir

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / self.logs_subdir


# Default configuration instance
default_config = PipelineConfig()
